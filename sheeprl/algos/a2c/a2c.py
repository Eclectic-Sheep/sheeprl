import os
import warnings
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import SumMetric

from sheeprl.algos.a2c.agent import build_agent
from sheeprl.algos.a2c.loss import policy_loss, value_loss
from sheeprl.algos.a2c.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import gae


def train(
    fabric: Fabric,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    cfg: Dict[str, Any],
):
    """Train the agent on the data collected from the environment."""
    indexes = list(range(data.shape[0]))
    if cfg.buffer.share_data:
        sampler = DistributedSampler(
            indexes,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=cfg.seed,
        )
    else:
        sampler = RandomSampler(indexes)
    sampler = BatchSampler(sampler, batch_size=cfg.algo.per_rank_batch_size, drop_last=False)

    optimizer.zero_grad(set_to_none=True)
    if cfg.buffer.share_data:
        sampler.sampler.set_epoch(0)
    for i, batch_idxes in enumerate(sampler):
        batch = data[batch_idxes]
        obs = {k: batch[k] for k in cfg.algo.mlp_keys.encoder}

        # is_accumulating is True for every i except for the last one
        is_accumulating = i < len(sampler) - 1

        with fabric.no_backward_sync(agent, enabled=is_accumulating):
            _, logprobs, values = agent(obs, torch.split(batch["actions"], agent.actions_dim, dim=-1))

            # Policy loss
            pg_loss = policy_loss(
                logprobs,
                batch["advantages"],
                cfg.algo.loss_reduction,
            )

            # Value loss
            v_loss = value_loss(
                values,
                batch["returns"],
                cfg.algo.loss_reduction,
            )

            loss = pg_loss + v_loss
            fabric.backward(loss)

        if not is_accumulating:
            if cfg.algo.max_grad_norm > 0.0:
                fabric.clip_gradients(agent, optimizer, max_norm=cfg.algo.max_grad_norm)
            optimizer.step()

        # Update metrics
        if aggregator and not aggregator.disabled:
            aggregator.update("Loss/policy_loss", pg_loss.detach())
            aggregator.update("Loss/value_loss", v_loss.detach())


@register_algorithm(decoupled=False)
def main(fabric: Fabric, cfg: Dict[str, Any]):
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank * cfg.env.num_envs,
                log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    observation_space = envs.single_observation_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.algo.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `algo.mlp_keys.encoder=[state]`")
    for k in cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder:
        if k in observation_space.keys() and len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the A2C agent. "
                f"The observation with key '{k}' has shape {observation_space[k].shape}. "
                f"Provided environment: {cfg.env.id}"
            )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
    obs_keys = cfg.algo.mlp_keys.encoder

    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(envs.single_action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        envs.single_action_space.shape
        if is_continuous
        else (envs.single_action_space.nvec.tolist() if is_multidiscrete else [envs.single_action_space.n])
    )

    # Create the agent model: this should be a torch.nn.Module to be accelerated with Fabric
    # Given that the environment has been created with the `make_env` method, the agent
    # forward method must accept as input a dictionary like {"obs1_name": obs1, "obs2_name": obs2, ...}.
    # The agent should be able to process both image and vector-like observations.
    agent = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["agent"] if cfg.checkpoint.resume_from else None,
    )
    fabric.print(agent.module)

    # the optimizer and set up it with Fabric
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")

    # Create a metric aggregator to log the metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    rb = ReplayBuffer(cfg.algo.rollout_steps, cfg.env.num_envs, device=device, memmap=cfg.buffer.memmap)
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=device)

    # Global variables
    last_log = 0
    last_train = 0
    train_step = 0
    policy_step = 0
    last_checkpoint = 0
    policy_steps_per_update = int(cfg.env.num_envs * cfg.algo.rollout_steps * world_size)
    num_updates = cfg.algo.total_steps // policy_steps_per_update if not cfg.dry_run else 1

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

    # Get the first environment observation and start the optimization
    o = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    next_obs = {}
    for k in o.keys():
        if k in obs_keys:
            torch_obs = torch.as_tensor(o[k], dtype=torch.float32, device=fabric.device)
            step_data[k] = torch_obs
            next_obs[k] = torch_obs

    for update in range(1, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs * world_size

            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
                with torch.no_grad():
                    # Sample an action given the observation received by the environment
                    # This calls the `forward` method of the PyTorch module, escaping from Fabric
                    # because we don't want this to be a synchronization point
                    actions, _, values = agent.module(next_obs)
                    if is_continuous:
                        real_actions = torch.cat(actions, -1).cpu().numpy()
                    else:
                        real_actions = np.concatenate([act.argmax(dim=-1).cpu().numpy() for act in actions], axis=-1)
                    actions = torch.cat(actions, -1)

                # Single environment step
                o, rewards, done, truncated, info = envs.step(real_actions.reshape(envs.action_space.shape))

            dones = np.logical_or(done, truncated)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=device).view(cfg.env.num_envs, -1)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(cfg.env.num_envs, -1)

            # Update the step data
            step_data["dones"] = dones
            step_data["values"] = values
            step_data["actions"] = actions
            step_data["rewards"] = rewards
            if cfg.buffer.memmap:
                step_data["returns"] = torch.zeros_like(rewards, dtype=torch.float32)
                step_data["advantages"] = torch.zeros_like(rewards, dtype=torch.float32)

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            # Update the observation and done
            obs = {}
            for k in o.keys():
                if k in obs_keys:
                    torch_obs = torch.as_tensor(o[k], dtype=torch.float32, device=fabric.device)
                    step_data[k] = torch_obs
                    obs[k] = torch_obs
            next_obs = obs

            if cfg.metric.log_level > 0 and "final_info" in info:
                for i, agent_ep_info in enumerate(info["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        if aggregator and "Rewards/rew_avg" in aggregator:
                            aggregator.update("Rewards/rew_avg", ep_rew)
                        if aggregator and "Game/ep_len_avg" in aggregator:
                            aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.no_grad():
            next_values = agent.module.get_value(next_obs)
            returns, advantages = gae(
                rb["rewards"].to(torch.float64),
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

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        # Train the agent
        train(fabric, agent, optimizer, local_data, aggregator, cfg)

        # Log metrics
        if policy_step - last_log >= cfg.metric.log_every or update == num_updates or cfg.dry_run:
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics:
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
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_step": update,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    envs.close()
    if fabric.is_global_zero:
        test(agent.module, fabric, cfg, log_dir)
