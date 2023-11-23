import os
import pathlib
import warnings
from typing import Any, Dict, Union

import gymnasium as gym
import hydra.utils
import numpy as np
import torch
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from omegaconf import OmegaConf
from torchmetrics import MeanMetric, SumMetric

from sheeprl.algos.muzero.agent import MuzeroAgent
from sheeprl.algos.muzero.loss import policy_loss, reward_loss, value_loss
from sheeprl.algos.muzero.utils import MCTS, test, visit_softmax_temperature
from sheeprl.data.buffers_np import Trajectory, TrajectoryReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import dotdict, nstep_returns


def apply_temperature(logits, temperature):
    """Returns `logits / temperature`, supporting also temperature=0.

    Taken from https://github.com/deepmind/mctx/blob/main/mctx/_src/policies.py#L409
    """
    # The max subtraction prevents +inf after dividing by a small temperature.
    logits = logits - np.max(logits, keepdims=True, axis=-1)
    tiny = np.finfo(logits.dtype).tiny
    return logits / max(tiny, temperature)


def train(
    fabric: Fabric,
    agent: Union[torch.nn.Module, _FabricModule],
    optimizer: torch.optim.Optimizer,
    data: Dict[str, np.ndarray],
    aggregator: MetricAggregator,
    cfg: Dict[str, Any],
):
    target_rewards = data["rewards"]
    target_values = data["returns"]
    target_policies = data["policies"]
    obs = np.concatenate(data[k] for k in cfg.mlp_keys.encoder)
    # preprocessed_obs = {k: torch.as_tensor(data[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder} TODO
    observations: torch.Tensor = torch.as_tensor(obs, dtype=torch.float32)  # shape should be (L, N, C, H, W)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)

    hidden_states, policy_0, value_0 = agent.initial_inference(observations[0])  # in shape should be (N, C, H, W)
    # Policy loss
    pg_loss = policy_loss(policy_0, target_policies[0])
    # Value loss
    v_loss = value_loss(value_0, target_values[0])
    # Reward loss
    r_loss = torch.tensor(0.0, device=observations.device)
    entropy = torch.distributions.Categorical(logits=policy_0.detach()).entropy().unsqueeze(0)

    for sequence_idx in range(1, cfg.algo.chunk_sequence_len):
        hidden_states, rewards, policies, values = agent.recurrent_inference(
            actions[sequence_idx], hidden_states
        )  # action should be (1, N, 1)
        # Policy loss
        pg_loss += policy_loss(policies[0], target_policies[sequence_idx])
        # Value loss
        v_loss += value_loss(values[0], target_values[sequence_idx])
        # Reward loss
        r_loss += reward_loss(rewards[0], target_rewards[sequence_idx])
        entropy += torch.distributions.Categorical(logits=policies.detach()).entropy()

    # Equation (1) in the paper, the regularization loss is handled by `weight_decay` in the optimizer
    loss = (pg_loss + v_loss + r_loss) / cfg.algo.chunk_sequence_len

    optimizer.zero_grad(set_to_none=True)
    fabric.backward(loss)
    optimizer.step()

    # Update metrics
    aggregator.update("Loss/policy_loss", pg_loss.detach())
    aggregator.update("Loss/value_loss", v_loss.detach())
    aggregator.update("Loss/reward_loss", r_loss.detach())
    aggregator.update("Loss/total_loss", loss.detach())
    aggregator.update("Gradient/gradient_norm", agent.gradient_norm())
    aggregator.update("Info/policy_entropy", entropy.mean())


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    ## to take from hydra
    assert cfg.env.num_envs == 1, "Only one environment is supported"

    # Initialize Fabric
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint # TODO fix
    if cfg.checkpoint.resume_from:
        root_dir = cfg.root_dir
        run_name = cfg.run_name
        state = fabric.load(cfg.checkpoint.resume_from)
        ckpt_path = pathlib.Path(cfg.checkpoint.resume_from)
        cfg = dotdict(OmegaConf.load(ckpt_path.parent.parent.parent / ".hydra" / "config.yaml"))
        cfg.checkpoint.resume_from = str(ckpt_path)
        cfg.per_rank_batch_size = state["batch_size"] // fabric.world_size
        cfg.root_dir = root_dir
        cfg.run_name = run_name

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, cfg)
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)

    # Environment setup # TODO add support to parallel envs
    env = make_env(
        cfg,
        cfg.seed + rank * cfg.env.num_envs + 0,
        rank * cfg.env.num_envs,
        logger.log_dir if rank == 0 else None,
        "train",
        vector_env_idx=0,
    )()

    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action space is supported"
    obs_shapes = {k: env.observation_space.spaces[k] for k in cfg.mlp_keys.encoder}
    input_dims = [np.prod(s) for s in obs_shapes.values()]
    num_actions = env.action_space.n

    # Create the model
    # TODO initialize using cfg for each model
    full_support_size = 2 * cfg.algo.support_size + 1
    agent = MuzeroAgent(cfg, input_dims, num_actions, full_support_size)

    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters())

    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Metrics
    with device:  # TODO BETTER LIKE THIS OR WITH .TO(DEVICE)?
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(),
                "Game/ep_len_avg": MeanMetric(),
                "Time/step_per_second": MeanMetric(),
                "Loss/value_loss": MeanMetric(),
                "Loss/policy_loss": MeanMetric(),
                "Loss/reward_loss": MeanMetric(),
                "Loss/total_loss": MeanMetric(),
                "Gradient/gradient_norm": MeanMetric(),
                "Info/policy_entropy": MeanMetric(),
            }
        )

    # Local data
    buffer_size = cfg.buffer.size // int(fabric.world_size) if not cfg.dry_run else 1
    rb = TrajectoryReplayBuffer(max_num_trajectories=buffer_size, memmap=cfg.buffer.memmap)

    # Initialize MCTS
    mcts = MCTS(
        agent,
        cfg.algo.num_simulations,
        gamma=cfg.algo.gamma,
        dirichlet_alpha=cfg.algo.dirichlet_alpha,
        exploration_fraction=cfg.algo.exploration_fraction,
        support_size=cfg.algo.support_size,
        pbc_base=cfg.algo.pbc_base,
        pbc_init=cfg.algo.pbc_init,
    )

    # Global variables
    last_train = 0
    train_step = 0
    start_step = state["update"] // fabric.world_size if cfg.checkpoint.resume_from else 1
    policy_step = state["update"] * cfg.env.num_envs * cfg.algo.rollout_steps if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    num_episodes_per_update = int(cfg.env.num_envs * world_size)
    num_updates = cfg.total_steps // num_episodes_per_update if not cfg.dry_run else 1

    cfg.algo.learning_starts // int(fabric.world_size) if not cfg.dry_run else 0  # TODO check

    # Warning for log and checkpoint every
    if cfg.metric.log_every % num_episodes_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"num_episodes_per_update value ({num_episodes_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % num_episodes_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"num_episodes_per_update value ({num_episodes_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    # Linear learning rate scheduler
    if cfg.algo.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)
        if cfg.checkpoint.resume_from:
            scheduler.load_state_dict(state["scheduler"])

    env_steps = 0
    warm_up = True
    for update_step in range(start_step, num_updates + 1):
        with torch.no_grad():
            # reset the episode at every update
            # Get the first environment observation and start the optimization
            obs: dict[str, np.ndarray] = env.reset(seed=cfg.seed)[0]
            for k, v in obs.items():
                preprocessed_obs = {
                    k: torch.as_tensor(v[np.newaxis], dtype=torch.float32, device=device) for k in cfg.mlp_keys.encoder
                }
            rew_sum = 0.0

            steps_data = None
            for trajectory_step in range(0, cfg.algo.max_trajectory_len):
                policy_step += 1
                with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
                    if not warm_up:
                        # start MCTS
                        node = mcts.search(preprocessed_obs)

                        # Select action based on the visit count distribution and the temperature
                        visits_count = np.array([child.visit_count for child in node.children])
                        temperature = visit_softmax_temperature(training_steps=update_step)
                        visit_probs = visits_count / cfg.algo.num_simulations
                        visit_probs = np.where(visit_probs > 0, visit_probs, 1 / visit_probs.shape[-1])
                        tiny = np.finfo(visit_probs.dtype).tiny
                        visit_logits = np.log(np.maximum(visit_probs, tiny))
                        logits = apply_temperature(visit_logits, temperature)
                        action = np.random.choice(
                            np.arange(visit_probs.shape[-1]), p=np.exp(logits) / np.sum(np.exp(logits))
                        )
                        value = node.value()
                    else:
                        action = env.action_space.sample()
                        visit_probs = np.ones((1, num_actions)) / num_actions
                        value = 0.0

                    # Single environment step
                    next_obs, reward, done, truncated, info = env.step(action)
                rew_sum += reward

                # Store the current step data
                trajectory_step_data = Trajectory(
                    {
                        "policies": visit_probs.reshape(1, num_actions),
                        "actions": action.reshape(1, 1),
                        "rewards": np.array(reward).reshape(1, 1),
                        "values": np.array(value).reshape(1, 1),
                        "dones": np.array(done).reshape(1, 1),
                    }
                )
                for k in obs:
                    trajectory_step_data[k] = obs[k].reshape(1, *obs_shapes[k])
                if steps_data is None:
                    steps_data = trajectory_step_data
                else:
                    # append the last trajectory_step_data values to each key of steps_data
                    for key in steps_data.keys():
                        steps_data[key] = np.concatenate([steps_data[key], trajectory_step_data[key]], axis=0)

                if done or truncated:
                    break
                else:
                    obs = next_obs

            aggregator.update("Rewards/rew_avg", rew_sum)
            aggregator.update("Game/ep_len_avg", trajectory_step)
            # print("Finished episode")
            if len(steps_data) >= cfg.algo.chunk_sequence_len:
                steps_data["returns"] = nstep_returns(
                    steps_data["rewards"],
                    steps_data["values"],
                    steps_data["dones"],
                    cfg.algo.nstep_horizon,
                    cfg.algo.gamma,
                )
                steps_data["weights"] = np.abs(steps_data["returns"] - steps_data["values"]) ** cfg.algo.priority_alpha
                rb.add(trajectory=steps_data)
            env_steps += trajectory_step

        if len(rb) >= cfg.algo.learning_starts:
            warm_up = False
            print("UPDATING")
            for _ in range(cfg.algo.update_epochs):
                data = rb.sample(cfg.algo.chunks_per_batch, cfg.algo.chunk_sequence_len)

                with timer("Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute)):
                    train(fabric, agent, optimizer, data, aggregator, cfg)
                train_step += world_size

            if cfg.algo.anneal_lr:
                fabric.log("Info/learning_rate", scheduler.get_last_lr()[0], policy_step)
                scheduler.step()
            else:
                fabric.log("Info/learning_rate", cfg.algo.optimizer.lr, policy_step)

            # Log metrics
            if policy_step - last_log >= cfg.metric.log_every or update_step == num_updates or cfg.dry_run:
                # Sync distributed metrics
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

                # Sync distributed timers
                timer_metrics = timer.compute()
                fabric.log(
                    "Time/sps_train",
                    (train_step - last_train) / timer_metrics["Time/train_time"],
                    policy_step,
                )
                fabric.log(
                    "Time/sps_env_interaction",
                    ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                    / timer_metrics["Time/env_interaction_time"],
                    policy_step,
                )
                timer.reset()

                # Reset counters
                last_log = policy_step
                last_train = train_step

            # Checkpoint model
            if (
                (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every)
                or cfg.dry_run
                or update_step == num_updates
            ):
                last_checkpoint = policy_step
                state = {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
                    "update": update_step * world_size,
                    "batch_size": cfg.algo.chunks_per_batch * fabric.world_size,
                    "last_log": last_log,
                    "last_checkpoint": last_checkpoint,
                }
                ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
                fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    env.close()
    if fabric.is_global_zero:
        test(agent.module, mcts, fabric, cfg)
