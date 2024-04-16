import os
import warnings
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import SumMetric

from sheeprl.algos.a2c.agent import A2CAgent, build_agent
from sheeprl.algos.a2c.loss import policy_loss, value_loss
from sheeprl.algos.a2c.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import gae, save_configs


def train(
    fabric: Fabric,
    agent: A2CAgent,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, torch.Tensor],
    aggregator: MetricAggregator,
    cfg: Dict[str, Any],
):
    """Train the agent on the data collected from the environment."""

    # Prepare the sampler
    # If we are in the distributed setting, we need to use a DistributedSampler, which
    # will shuffle the data at each epoch and will ensure that each process will get
    # a different part of the data
    indexes = list(range(next(iter(data.values())).shape[0]))
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

    # Train the agent
    # Even though in the Spinning-Up A2C algorithm implementation
    # (https://spinningup.openai.com/en/latest/algorithms/vpg.html) the policy gradient is estimated
    # by taking the mean over all the sequences collected
    # of the sum of the actions log-probabilities gradients' multiplied by the advantages,
    # we do not do that, instead we take the overall sum (or mean, depending on the loss reduction).
    # This is achieved by accumulating the gradients and calling the backward method only at the end.
    for i, batch_idxes in enumerate(sampler):
        batch = {k: v[batch_idxes] for k, v in data.items()}
        obs = {k: v for k, v in batch.items() if k in cfg.algo.mlp_keys.encoder}

        # is_accumulating is True for every i except for the last one
        is_accumulating = i < len(sampler) - 1

        with fabric.no_backward_sync(agent.feature_extractor, enabled=is_accumulating), fabric.no_backward_sync(
            agent.actor, enabled=is_accumulating
        ), fabric.no_backward_sync(agent.critic, enabled=is_accumulating):
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
    agent, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["agent"] if cfg.checkpoint.resume_from else None,
    )

    # the optimizer and set up it with Fabric
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Create a metric aggregator to log the metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    rb = ReplayBuffer(
        cfg.buffer.size,
        cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        obs_keys=obs_keys,
    )

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
    step_data = {}
    next_obs = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    for k in obs_keys:
        step_data[k] = next_obs[k][np.newaxis]

    for update in range(1, num_updates + 1):
        with torch.inference_mode():
            for _ in range(0, cfg.algo.rollout_steps):
                policy_step += cfg.env.num_envs * world_size

                # Measure environment interaction time: this considers both the model forward
                # to get the action given the observation and the time taken into the environment
                with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                    # Sample an action given the observation received by the environment
                    # This calls the `forward` method of the PyTorch module, escaping from Fabric
                    # because we don't want this to be a synchronization point
                    torch_obs = {k: torch.as_tensor(next_obs[k], dtype=torch.float32, device=device) for k in obs_keys}
                    actions, _, values = player(torch_obs)
                    if is_continuous:
                        real_actions = torch.cat(actions, -1).cpu().numpy()
                    else:
                        real_actions = torch.cat([act.argmax(dim=-1) for act in actions], axis=-1).cpu().numpy()
                    actions = torch.cat(actions, -1).cpu().numpy()

                    # Single environment step
                    obs, rewards, terminated, truncated, info = envs.step(real_actions.reshape(envs.action_space.shape))
                    truncated_envs = np.nonzero(truncated)[0]
                    if len(truncated_envs) > 0:
                        real_next_obs = {
                            k: torch.empty(
                                len(truncated_envs),
                                *observation_space[k].shape,
                                dtype=torch.float32,
                                device=device,
                            )
                            for k in obs_keys
                        }
                        for i, truncated_env in enumerate(truncated_envs):
                            for k, v in info["final_observation"][truncated_env].items():
                                torch_v = torch.as_tensor(v, dtype=torch.float32, device=device)
                                if k in cfg.algo.cnn_keys.encoder:
                                    torch_v = torch_v.view(-1, *v.shape[-2:])
                                    torch_v = torch_v / 255.0 - 0.5
                                real_next_obs[k][i] = torch_v
                        vals = player.get_values(real_next_obs).cpu().numpy()
                        rewards[truncated_envs] += cfg.algo.gamma * vals.reshape(rewards[truncated_envs].shape)
                    dones = np.logical_or(terminated, truncated).reshape(cfg.env.num_envs, -1).astype(np.uint8)
                    rewards = rewards.reshape(cfg.env.num_envs, -1)

                # Update the step data
                step_data["dones"] = dones[np.newaxis]
                step_data["values"] = values.cpu().numpy()[np.newaxis]
                step_data["actions"] = actions[np.newaxis]
                step_data["rewards"] = rewards[np.newaxis]
                if cfg.buffer.memmap:
                    step_data["returns"] = np.zeros_like(rewards, shape=(1, *rewards.shape))
                    step_data["advantages"] = np.zeros_like(rewards, shape=(1, *rewards.shape))

                # Append data to buffer
                rb.add(step_data, validate_args=cfg.buffer.validate_args)

                # Update the observation and dones
                next_obs = {}
                for k in obs_keys:
                    _obs = obs[k]
                    step_data[k] = _obs[np.newaxis]
                    next_obs[k] = _obs

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

        # Transform the data into PyTorch Tensors
        local_data = rb.to_tensor(dtype=None, device=device, from_numpy=cfg.buffer.from_numpy)

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.inference_mode():
            torch_obs = {k: torch.as_tensor(next_obs[k], dtype=torch.float32, device=device) for k in obs_keys}
            next_values = player.get_values(torch_obs)
            returns, advantages = gae(
                local_data["rewards"].to(torch.float64),
                local_data["values"],
                local_data["dones"],
                next_values,
                cfg.algo.rollout_steps,
                cfg.algo.gamma,
                cfg.algo.gae_lambda,
            )

            # Add returns and advantages to the buffer
            local_data["returns"] = returns.float()
            local_data["advantages"] = advantages.float()

        # Train the agent
        with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
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
            or (update == num_updates and cfg.checkpoint.save_last)
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
    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.ppo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"agent": agent}
        register_model(fabric, log_models, cfg, models_to_log)
