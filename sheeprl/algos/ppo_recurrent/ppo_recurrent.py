import copy
import os
import pathlib
import warnings

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torchmetrics import MeanMetric, SumMetric

from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent
from sheeprl.algos.ppo_recurrent.utils import test
from sheeprl.data.buffers import SequentialReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay


def train(
    fabric: Fabric,
    agent: RecurrentPPOAgent,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    cfg: DictConfig,
):
    for i in range(cfg.algo.update_epochs):
        entropies = []
        new_values = []
        new_logprobs = []
        sequence = data[i]
        for k in cfg.cnn_keys.encoder:
            sequence[k] = sequence[k] / 255.0 - 0.5
        for step in sequence:
            step = step.unsqueeze(0)
            _, logprobs, entropy, values, _ = agent(
                {k: step[k] for k in set(cfg.cnn_keys.encoder + cfg.mlp_keys.encoder)},
                prev_actions=step["prev_actions"],
                prev_hx=step["prev_states"],
                actions=step["actions"],
            )
            entropies.append(entropy)
            new_values.append(values)
            new_logprobs.append(logprobs)

        normalized_advantages = sequence["advantages"]
        if cfg.algo.normalize_advantages and len(normalized_advantages) > 1:
            normalized_advantages = normalize_tensor(normalized_advantages)

        # Policy loss
        pg_loss = policy_loss(
            torch.cat(new_logprobs, dim=0),
            sequence["logprobs"],
            normalized_advantages,
            cfg.algo.clip_coef,
            "mean",
        )

        # Value loss
        v_loss = value_loss(
            torch.cat(new_values, dim=0),
            sequence["values"],
            sequence["returns"],
            cfg.algo.clip_coef,
            cfg.algo.clip_vloss,
            "mean",
        )

        # Entropy loss
        ent_loss = entropy_loss(torch.cat(entropies, dim=0), cfg.algo.loss_reduction)

        # Equation (9) in the paper
        loss = pg_loss + cfg.algo.vf_coef * v_loss + cfg.algo.ent_coef * ent_loss

        optimizer.zero_grad(set_to_none=True)
        fabric.backward(loss)
        if cfg.algo.max_grad_norm > 0.0:
            fabric.clip_gradients(agent, optimizer, max_norm=cfg.algo.max_grad_norm)
        optimizer.step()

        # Update metrics
        aggregator.update("Loss/policy_loss", pg_loss.detach())
        aggregator.update("Loss/value_loss", v_loss.detach())
        aggregator.update("Loss/entropy_loss", ent_loss.detach())


@register_algorithm()
def main(fabric: Fabric, cfg: DictConfig):
    initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)
    initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)

    if "minedojo" in cfg.env.wrapper._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by PPO Recurrent agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    if cfg.buffer.share_data:
        warnings.warn(
            "The script has been called with `buffer.share_data=True`: with recurrent PPO only gradients are shared"
        )

    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        root_dir = cfg.root_dir
        run_name = cfg.run_name
        state = fabric.load(cfg.checkpoint.resume_from)
        ckpt_path = pathlib.Path(cfg.checkpoint.resume_from)
        cfg = OmegaConf.load(ckpt_path.parent.parent.parent / ".hydra" / "config.yaml")
        cfg.checkpoint.resume_from = str(ckpt_path)
        cfg.per_rank_batch_size = state["batch_size"] // fabric.world_size
        cfg.root_dir = root_dir
        cfg.run_name = run_name

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, cfg)
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank * cfg.env.num_envs,
                logger.log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    observation_space = envs.single_observation_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if cfg.cnn_keys.encoder + cfg.mlp_keys.encoder == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: "
            "`cnn_keys.encoder=[rgb]` or `mlp_keys.encoder=[state]`"
        )
    fabric.print("Encoder CNN keys:", cfg.cnn_keys.encoder)
    fabric.print("Encoder MLP keys:", cfg.mlp_keys.encoder)
    obs_keys = cfg.cnn_keys.encoder + cfg.mlp_keys.encoder

    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(envs.single_action_space, gym.spaces.MultiDiscrete)
    actions_dim = (
        envs.single_action_space.shape
        if is_continuous
        else (envs.single_action_space.nvec.tolist() if is_multidiscrete else [envs.single_action_space.n])
    )

    # Define the agent and the optimizer
    agent = RecurrentPPOAgent(
        actions_dim=actions_dim,
        obs_space=observation_space,
        encoder_cfg=cfg.algo.encoder,
        rnn_cfg=cfg.algo.rnn,
        actor_cfg=cfg.algo.actor,
        critic_cfg=cfg.algo.critic,
        cnn_keys=cfg.cnn_keys.encoder,
        mlp_keys=cfg.mlp_keys.encoder,
        is_continuous=is_continuous,
        num_envs=cfg.env.num_envs,
        screen_size=cfg.env.screen_size,
        device=device,
    )
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters())

    # Load the state from the checkpoint
    if cfg.checkpoint.resume_from:
        agent.load_state_dict(state["agent"])
        optimizer.load_state_dict(state["optimizer"])

    # Setup agent and optimizer with Fabric
    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Create a metric aggregator to log the metrics
    aggregator = MetricAggregator(
        {
            "Rewards/rew_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Game/ep_len_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/value_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/policy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/entropy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
        }
    ).to(device)

    # Local data
    rb = SequentialReplayBuffer(
        cfg.algo.rollout_steps,
        cfg.env.num_envs,
        device=device,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    step_data = TensorDict({}, batch_size=[1, cfg.env.num_envs], device=device)

    # Check that `rollout_steps` = k * `per_rank_sequence_length`
    if cfg.algo.rollout_steps % cfg.per_rank_sequence_length != 0:
        pass

    # Global variables
    last_train = 0
    train_step = 0
    start_step = state["update"] // fabric.world_size if cfg.checkpoint.resume_from else 1
    policy_step = state["update"] * cfg.env.num_envs * cfg.algo.rollout_steps if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * cfg.algo.rollout_steps * world_size)
    num_updates = cfg.total_steps // policy_steps_per_update if not cfg.dry_run else 1

    # Warning for log and checkpoint every
    if cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    # Linear learning rate scheduler
    if cfg.algo.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)
        if cfg.checkpoint.resume_from:
            scheduler.load_state_dict(state["scheduler"])

    # Get the first environment observation and start the optimization
    o = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    obs = {}
    for k in obs_keys:
        torch_obs = torch.as_tensor(o[k], device=fabric.device)
        if k in cfg.cnn_keys.encoder:
            torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
        elif k in cfg.mlp_keys.encoder:
            torch_obs = torch_obs.float()
        step_data[k] = torch_obs[None]
        obs[k] = torch_obs

    # Get the resetted recurrent states from the agent
    prev_states = agent.initial_states
    prev_actions = torch.zeros(1, cfg.env.num_envs, sum(actions_dim), device=fabric.device)

    for update in range(start_step, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs * world_size

            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
                with torch.no_grad():
                    # Sample an action given the observation received by the environment
                    normalized_obs = {
                        k: obs[k][None] / 255 - 0.5 if k in cfg.cnn_keys.encoder else obs[k][None] for k in obs_keys
                    }
                    actions, logprobs, _, values, states = agent.module(
                        normalized_obs, prev_actions=prev_actions, prev_hx=prev_states
                    )
                    if is_continuous:
                        real_actions = torch.cat(actions, -1).cpu().numpy()
                    else:
                        real_actions = np.concatenate([act.argmax(dim=-1).cpu().numpy() for act in actions], axis=-1)
                    actions = torch.cat(actions, -1)

                # Single environment step
                next_obs, rewards, dones, truncated, info = envs.step(real_actions.reshape(envs.action_space.shape))
                truncated_envs = np.nonzero(truncated)[0]
                if len(truncated_envs) > 0:
                    real_next_obs = {}
                    for final_obs in info["final_observation"]:
                        if final_obs is not None:
                            for k, v in final_obs.items():
                                torch_v = torch.as_tensor(v, dtype=torch.float32, device=device)
                                if k in cfg.cnn_keys.encoder:
                                    torch_v = torch_v / 255.0 - 0.5
                                real_next_obs[k] = torch_v[None]
                    with torch.no_grad():
                        feat = agent.module.feature_extractor(real_next_obs)
                        real_states = agent.module.rnn(torch.cat((feat, actions), dim=-1), states)
                        vals = agent.module.get_values(real_states).cpu().numpy()
                        rewards[truncated_envs] += vals
                dones = np.logical_or(dones, truncated)
                dones = torch.as_tensor(dones, dtype=torch.float32, device=device).view(1, cfg.env.num_envs, -1)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(1, cfg.env.num_envs, -1)

            step_data["dones"] = dones
            step_data["values"] = values
            step_data["states"] = states
            step_data["actions"] = actions
            step_data["rewards"] = rewards
            step_data["logprobs"] = logprobs
            step_data["prev_states"] = prev_states
            step_data["prev_actions"] = prev_actions
            if cfg.buffer.memmap:
                step_data["returns"] = torch.zeros_like(rewards)
                step_data["advantages"] = torch.zeros_like(rewards)

            # Append data to buffer
            rb.add(step_data)

            # Update actions
            prev_actions = actions

            # Update the observation
            for k in obs_keys:
                if k in cfg.cnn_keys.encoder:
                    torch_obs = torch.as_tensor(obs[k], device=device)
                    torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
                elif k in cfg.mlp_keys.encoder:
                    torch_obs = torch.as_tensor(obs[k], device=device, dtype=torch.float32)
                step_data[k] = torch_obs[None]
                next_obs[k] = torch_obs
            obs = next_obs

            # Reset the states if the episode is done
            if cfg.algo.reset_recurrent_state_on_done:
                prev_states = (1 - dones) * states
            else:
                prev_states = states

            if "final_info" in info:
                for i, agent_ep_info in enumerate(info["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        aggregator.update("Rewards/rew_avg", ep_rew)
                        aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.no_grad():
            normalized_obs = {
                k: obs[k][None] / 255 - 0.5 if k in cfg.cnn_keys.encoder else obs[k][None] for k in obs_keys
            }
            feat = agent.module.feature_extractor(normalized_obs)
            next_states = agent.module.rnn(torch.cat((feat, actions), dim=-1), states)
            next_values = agent.module.get_values(next_states)
            returns, advantages = gae(
                rb["rewards"],
                rb["values"],
                rb["dones"],
                next_values,
                cfg.algo.rollout_steps,
                cfg.algo.gamma,
                cfg.algo.gae_lambda,
            )

            # Add returns and advantages to the buffer
            rb["returns"] = returns.float()
            rb["advantages"] = advantages.float()

        # Get the training data as a TensorDict
        local_data = rb.sample(
            batch_size=cfg.per_rank_batch_size,
            sequence_length=cfg.per_rank_sequence_length,
            n_samples=cfg.algo.update_epochs,
        )

        with timer("Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute)):
            train(fabric, agent, optimizer, local_data, aggregator, cfg)
        train_step += world_size

        if cfg.algo.anneal_lr:
            fabric.log("Info/learning_rate", scheduler.get_last_lr()[0], policy_step)
            scheduler.step()
        else:
            fabric.log("Info/learning_rate", cfg.algo.optimizer.lr, policy_step)

        fabric.log("Info/clip_coef", cfg.algo.clip_coef, policy_step)
        if cfg.algo.anneal_clip_coef:
            cfg.algo.clip_coef = polynomial_decay(
                update, initial=initial_clip_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        fabric.log("Info/ent_coef", cfg.algo.ent_coef, policy_step)
        if cfg.algo.anneal_ent_coef:
            cfg.algo.ent_coef = polynomial_decay(
                update, initial=initial_ent_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        # Log metrics
        if policy_step - last_log >= cfg.metric.log_every or update == num_updates or cfg.dry_run:
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
            or update == num_updates
        ):
            last_checkpoint = policy_step
            ckpt_state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
                "update": update * world_size,
                "batch_size": cfg.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=ckpt_state)

    envs.close()
    if fabric.is_global_zero:
        test(agent.module, fabric, cfg)
