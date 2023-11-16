import hydra.utils

USE_C = True

import os
import time
from datetime import datetime
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from torchmetrics import MeanMetric

if USE_C:
    from sheeprl._node import Node
else:
    from sheeprl.algos.muzero.mcts_utils import Node

from sheeprl.algos.muzero.agent import MlpDynamics, MuzeroAgent, Predictor
from sheeprl.algos.muzero.loss import policy_loss, reward_loss, value_loss
from sheeprl.algos.muzero.utils import MCTS, make_env, test, visit_softmax_temperature
from sheeprl.data.buffers_np import Trajectory, TrajectoryReplayBuffer
from sheeprl.models.models import MLP
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import nstep_returns


def apply_temperature(logits, temperature):
    """Returns `logits / temperature`, supporting also temperature=0.

    Taken from https://github.com/deepmind/mctx/blob/main/mctx/_src/policies.py#L409
    """
    # The max subtraction prevents +inf after dividing by a small temperature.
    logits = logits - np.max(logits, keepdims=True, axis=-1)
    tiny = np.finfo(logits.dtype).tiny
    return logits / max(tiny, temperature)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    ## to take from hydra
    learning_starts = 128  # TODO confronta con ppo

    # Initialize Fabric
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Set logger only on rank-0 but share the logger directory: since we don't know
    # what is happening during the `fabric.save()` method, at least we assure that all
    # ranks save under the same named folder.
    # As a plus, rank-0 sets the time uniquely for everyone
    world_collective = TorchCollective()
    if fabric.world_size > 1:
        world_collective.setup()
        world_collective.create_group()
    if rank == 0:
        root_dir = (
            cfg.root_dir
            if cfg.root_dir is not None
            else os.path.join("logs", "muzero", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        run_name = (
            cfg.run_name if cfg.run_name is not None else f"{cfg.env.id}_{cfg.algo.name}_{cfg.seed}_{int(time.time())}"
        )
        logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
        fabric._loggers = [logger]
        log_dir = logger.log_dir
        # fabric.logger.log_hyperparams(asdict(args))
        if fabric.world_size > 1:
            world_collective.broadcast_object_list([log_dir], src=0)
    else:
        data = [""]
        world_collective.broadcast_object_list(data, src=0)
        log_dir = data[0]
        os.makedirs(log_dir, exist_ok=True)

    # Environment setup
    env = make_env(cfg.env.id, seed=cfg.seed + rank, idx=rank, capture_video=False, run_name=run_name)()
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action space is supported"
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Create the model
    full_support_size = 2 * cfg.algo.support_size + 1
    agent = MuzeroAgent(
        representation=MLP(
            input_dims=env.observation_space.shape,
            hidden_sizes=tuple(),
            output_dim=cfg.algo.embedding_size,
            activation=torch.nn.ELU,
        ),
        dynamics=MlpDynamics(
            num_actions=env.action_space.n, embedding_size=cfg.algo.support_size, full_support_size=full_support_size
        ),
        prediction=Predictor(
            embedding_size=cfg.algo.support_size, num_actions=env.action_space.n, full_support_size=full_support_size
        ),
    )

    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, parameters=agent.parameters())
    optimizer = fabric.setup_optimizers(optimizer)

    # Metrics
    with device:
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
    start_time = time.perf_counter()
    num_updates = int(cfg.total_steps // int(fabric.world_size)) if not cfg.dry_run else 1  # TODO confronta con ppo
    learning_starts = learning_starts // int(fabric.world_size) if not cfg.dry_run else 0

    env_steps = 0
    warm_up = True
    for update_step in range(1, num_updates + 1):
        with torch.no_grad():
            # reset the episode at every update
            # Get the first environment observation and start the optimization
            obs: np.ndarray = env.reset(seed=cfg.seed)[0]
            rew_sum = 0.0

            steps_data = None
            for trajectory_step in range(0, cfg.max_trajectory_len):
                if not warm_up:
                    node = Node(0.0)

                    # start MCTS
                    mcts.search(
                        node,
                        obs,
                    )

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
                        "observations": obs.reshape(1, *obs_shape),
                        "rewards": np.array(reward).reshape(1, 1),
                        "values": np.array(value).reshape(1, 1),
                        "dones": np.array(done).reshape(1, 1),
                    }
                )
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

            # fabric.print(f"Rank-{rank}: update_step={update_step}, reward={rew_sum}")
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

        if len(rb) >= learning_starts:
            warm_up = False
            print("UPDATING")
            for _ in range(cfg.algo.update_epochs):
                # We sample one time to reduce the communications between processes
                data = rb.sample(cfg.algo.chunks_per_batch, cfg.algo.chunk_sequence_len)

                target_rewards = data["rewards"]
                target_values = data["returns"]
                target_policies = data["policies"]
                observations: torch.Tensor = torch.as_tensor(
                    data["observations"], dtype=torch.float32
                )  # shape should be (L, N, C, H, W)
                actions = torch.as_tensor(data["actions"], dtype=torch.float32)

                hidden_states, policy_0, value_0 = agent.initial_inference(
                    observations[0]
                )  # in shape should be (N, C, H, W)
                # Policy loss
                pg_loss = policy_loss(policy_0, target_policies[0])
                # Value loss
                v_loss = value_loss(value_0, target_values[0])
                # Reward loss
                r_loss = torch.tensor(0.0, device=device)
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

        aggregator.update("Time/step_per_second", int(update_step / (time.perf_counter() - start_time)))
        fabric.log_dict(aggregator.compute(), env_steps)
        aggregator.reset()

        if (
            (cfg.checkpoint.every > 0 and update_step % cfg.checkpoint.every == 0)
            or cfg.dry_run
            or update_step == num_updates
        ):  # TODO confronta con ppo
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "args": asdict(args),
                "update_step": update_step,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{update_step}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    env.close()

    if fabric.is_global_zero:
        test_env = make_env(
            cfg.env.id,
            None,
            0,
            True,
            fabric.logger.log_dir,
            "test",
            vector_env_idx=0,
        )()
        test(agent, test_env, fabric)  # , args)


if __name__ == "__main__":
    main()
