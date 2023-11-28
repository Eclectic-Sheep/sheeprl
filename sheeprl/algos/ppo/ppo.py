from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, Union

import gymnasium as gym
import hydra
import mlflow
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from mlflow.models.model import ModelInfo
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch import nn
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import SumMetric

from sheeprl.algos.ppo.agent import build_agent
from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.utils import normalize_obs, test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay, register_model, unwrap_fabric


def train(
    fabric: Fabric,
    agent: Union[nn.Module, _FabricModule],
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator | None,
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

    for epoch in range(cfg.algo.update_epochs):
        if cfg.buffer.share_data:
            sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            batch = data[batch_idxes]
            normalized_obs = normalize_obs(
                batch, cfg.algo.cnn_keys.encoder, cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder
            )
            _, logprobs, entropy, new_values = agent(
                normalized_obs, torch.split(batch["actions"], agent.actions_dim, dim=-1)
            )

            if cfg.algo.normalize_advantages:
                batch["advantages"] = normalize_tensor(batch["advantages"])

            # Policy loss
            pg_loss = policy_loss(
                logprobs,
                batch["logprobs"],
                batch["advantages"],
                cfg.algo.clip_coef,
                cfg.algo.loss_reduction,
            )

            # Value loss
            v_loss = value_loss(
                new_values,
                batch["values"],
                batch["returns"],
                cfg.algo.clip_coef,
                cfg.algo.clip_vloss,
                cfg.algo.loss_reduction,
            )

            # Entropy loss
            ent_loss = entropy_loss(entropy, cfg.algo.loss_reduction)

            # Equation (9) in the paper
            loss = pg_loss + cfg.algo.vf_coef * v_loss + cfg.algo.ent_coef * ent_loss

            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            if cfg.algo.max_grad_norm > 0.0:
                fabric.clip_gradients(agent, optimizer, max_norm=cfg.algo.max_grad_norm)
            optimizer.step()

            # Update metrics
            if aggregator and not aggregator.disabled:
                aggregator.update("Loss/policy_loss", pg_loss.detach())
                aggregator.update("Loss/value_loss", v_loss.detach())
                aggregator.update("Loss/entropy_loss", ent_loss.detach())


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    if "minedojo" in cfg.env.wrapper._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by PPO agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)
    initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)

    # Initialize Fabric
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)

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
    if cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: "
            "`cnn_keys.encoder=[rgb]` or `mlp_keys.encoder=[state]`"
        )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(envs.single_action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        envs.single_action_space.shape
        if is_continuous
        else (envs.single_action_space.nvec.tolist() if is_multidiscrete else [envs.single_action_space.n])
    )
    # Create the actor and critic models
    agent = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["agent"] if cfg.checkpoint.resume_from else None,
    )

    # Define the optimizer
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters())

    local_vars = locals()

    # Load the state from the checkpoint
    if cfg.checkpoint.resume_from:
        optimizer.load_state_dict(state["optimizer"])

    # Setup agent and optimizer with Fabric
    optimizer = fabric.setup_optimizers(optimizer)

    # Create a metric aggregator to log the metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator).to(device)

    # Local data
    if cfg.buffer.size < cfg.algo.rollout_steps:
        raise ValueError(
            f"The size of the buffer ({cfg.buffer.size}) cannot be lower "
            f"than the rollout steps ({cfg.algo.rollout_steps})"
        )
    rb = ReplayBuffer(
        cfg.buffer.size,
        cfg.env.num_envs,
        device=device,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        obs_keys=obs_keys,
    )
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=device)

    # Global variables
    last_train = 0
    train_step = 0
    start_step = state["update"] // fabric.world_size if cfg.checkpoint.resume_from else 1
    policy_step = state["update"] * cfg.env.num_envs * cfg.algo.rollout_steps if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * cfg.algo.rollout_steps * world_size)
    num_updates = cfg.algo.total_steps // policy_steps_per_update if not cfg.dry_run else 1

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_update != 0:
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
    obs = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    next_obs = {}
    for k in obs_keys:
        torch_obs = torch.as_tensor(obs[k]).to(fabric.device)
        if k in cfg.algo.cnn_keys.encoder:
            torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
        elif k in cfg.algo.mlp_keys.encoder:
            torch_obs = torch_obs.float()
        step_data[k] = torch_obs
        next_obs[k] = torch_obs

    for update in range(start_step, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs * world_size

            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
                with torch.no_grad():
                    # Sample an action given the observation received by the environment
                    normalized_obs = normalize_obs(next_obs, cfg.algo.cnn_keys.encoder, obs_keys)
                    actions, logprobs, _, values = agent.module(normalized_obs)
                    if is_continuous:
                        real_actions = torch.cat(actions, -1).cpu().numpy()
                    else:
                        real_actions = np.concatenate([act.argmax(dim=-1).cpu().numpy() for act in actions], axis=-1)
                    actions = torch.cat(actions, -1)

                # Single environment step
                obs, rewards, dones, truncated, info = envs.step(real_actions.reshape(envs.action_space.shape))
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
                                torch_v = torch_v.view(len(truncated_envs), -1, *torch_obs.shape[-2:]) / 255.0 - 0.5
                            real_next_obs[k][i] = torch_v
                    with torch.no_grad():
                        vals = agent.module.get_value(real_next_obs).cpu().numpy()
                        rewards[truncated_envs] += vals.reshape(rewards[truncated_envs].shape)
                dones = np.logical_or(dones, truncated)
                dones = torch.as_tensor(dones, dtype=torch.float32, device=device).view(cfg.env.num_envs, -1)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(cfg.env.num_envs, -1)

            # Update the step data
            step_data["dones"] = dones
            step_data["values"] = values
            step_data["actions"] = actions
            step_data["logprobs"] = logprobs
            step_data["rewards"] = rewards
            if cfg.buffer.memmap:
                step_data["returns"] = torch.zeros_like(rewards)
                step_data["advantages"] = torch.zeros_like(rewards)

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            # Update the observation and dones
            next_obs = {}
            for k in obs_keys:
                if k in cfg.algo.cnn_keys.encoder:
                    torch_obs = torch.as_tensor(obs[k], device=device)
                    torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
                elif k in cfg.algo.mlp_keys.encoder:
                    torch_obs = torch.as_tensor(obs[k], device=device, dtype=torch.float32)
                step_data[k] = torch_obs
                next_obs[k] = torch_obs

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
            normalized_obs = normalize_obs(next_obs, cfg.algo.cnn_keys.encoder, obs_keys)
            next_values = agent.module.get_value(normalized_obs)
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

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        if cfg.buffer.share_data and fabric.world_size > 1:
            # Gather all the tensors from all the world and reshape them
            gathered_data = fabric.all_gather(local_data.to_dict())  # Fabric does not work with TensorDict
            gathered_data = make_tensordict(gathered_data).view(-1)
        else:
            gathered_data = local_data

        with timer("Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute)):
            train(fabric, agent, optimizer, gathered_data, aggregator, cfg)
        train_step += world_size

        if cfg.metric.log_level > 0:
            # Log lr and coefficients
            if cfg.algo.anneal_lr:
                fabric.log("Info/learning_rate", scheduler.get_last_lr()[0], policy_step)
            else:
                fabric.log("Info/learning_rate", cfg.algo.optimizer.lr, policy_step)
            fabric.log("Info/clip_coef", cfg.algo.clip_coef, policy_step)
            fabric.log("Info/ent_coef", cfg.algo.ent_coef, policy_step)

            # Log metrics
            if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or update == num_updates):
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

        # Update lr and coefficients
        if cfg.algo.anneal_lr:
            scheduler.step()
        if cfg.algo.anneal_clip_coef:
            cfg.algo.clip_coef = polynomial_decay(
                update, initial=initial_clip_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )
        if cfg.algo.anneal_ent_coef:
            cfg.algo.ent_coef = polynomial_decay(
                update, initial=initial_ent_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        # Checkpoint model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            update == num_updates and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
                "update": update * world_size,
                "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    envs.close()
    if fabric.is_global_zero:
        test(agent.module, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:

        def log_models(
            run_id: str, experiment_id: str | None = None, run_name: str | None = None
        ) -> Dict[str, ModelInfo]:
            with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=True) as _:
                model_info = {}
                unwrapped_models = {}
                for k in cfg.model_manager.models.keys():
                    unwrapped_models[k] = unwrap_fabric(local_vars[k])
                    model_info[k] = mlflow.pytorch.log_model(unwrapped_models[k], artifact_path=k)
                mlflow.log_dict(cfg, "config.json")
            return model_info

        register_model(fabric, log_models, cfg)
