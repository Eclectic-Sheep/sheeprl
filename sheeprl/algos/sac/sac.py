from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torchmetrics import SumMetric

from sheeprl.algos.sac.agent import SACAgent, build_agent
from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.algos.sac.utils import prepare_obs, test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs


def train(
    fabric: Fabric,
    agent: SACAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    update: int,
    cfg: Dict[str, Any],
    policy_steps_per_iter: int,
    group: Optional[CollectibleGroup] = None,
):
    # Update the soft-critic
    next_target_qf_value = agent.get_next_target_q_values(
        data["next_observations"], data["rewards"], data["terminated"], cfg.algo.gamma
    )
    qf_values = agent.get_q_values(data["observations"], data["actions"])
    qf_loss = critic_loss(qf_values, next_target_qf_value, agent.num_critics)
    qf_optimizer.zero_grad(set_to_none=True)
    fabric.backward(qf_loss)
    qf_optimizer.step()

    # Update the target networks with EMA
    if update % (cfg.algo.critic.target_network_frequency // policy_steps_per_iter + 1) == 0:
        agent.qfs_target_ema()

    # Update the actor
    actions, logprobs = agent.get_actions_and_log_probs(data["observations"])
    qf_values = agent.get_q_values(data["observations"], actions)
    min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
    actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
    actor_optimizer.zero_grad(set_to_none=True)
    fabric.backward(actor_loss)
    actor_optimizer.step()

    # Update the entropy value
    alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
    alpha_optimizer.zero_grad(set_to_none=True)
    fabric.backward(alpha_loss)
    agent.log_alpha.grad = fabric.all_reduce(agent.log_alpha.grad, group=group)
    alpha_optimizer.step()

    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/value_loss", qf_loss)
        aggregator.update("Loss/policy_loss", actor_loss)
        aggregator.update("Loss/alpha_loss", alpha_loss)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    if "minedojo" in cfg.env.wrapper._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by SAC agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    if len(cfg.algo.cnn_keys.encoder) > 0:
        warnings.warn("SAC algorithm cannot allow to use images as observations, the CNN keys will be ignored")
        cfg.algo.cnn_keys.encoder = []

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
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the SAC agent")
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.algo.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
    for k in cfg.algo.mlp_keys.encoder:
        if len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the SAC agent. "
                f"The observation with key '{k}' has shape {observation_space[k].shape}. "
                f"Provided environment: {cfg.env.id}"
            )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    # Define the agent and the optimizer and setup sthem with Fabric
    agent, player = build_agent(
        fabric, cfg, observation_space, action_space, state["agent"] if cfg.checkpoint.resume_from else None
    )

    # Optimizers
    qf_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer,
        params=agent.qfs.parameters(),
        _convert_="all",
    )
    actor_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer,
        params=agent.actor.parameters(),
        _convert_="all",
    )
    alpha_optimizer = hydra.utils.instantiate(
        cfg.algo.alpha.optimizer,
        params=[agent.log_alpha],
        _convert_="all",
    )
    if cfg.checkpoint.resume_from:
        qf_optimizer.load_state_dict(state["qf_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        alpha_optimizer.load_state_dict(state["alpha_optimizer"])
    qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        qf_optimizer, actor_optimizer, alpha_optimizer
    )

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Create a metric aggregator to log the metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * world_size) if not cfg.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], ReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")

    # Global variables
    last_train = 0
    train_step = 0
    start_iter = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["iter_num"] // fabric.world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
    total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
    prefill_steps = learning_starts - int(learning_starts > 0)
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
        learning_starts += start_iter
        prefill_steps += start_iter

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )
    if cfg.checkpoint.every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )

    step_data = {}
    # Get the first environment observation and start the optimization
    obs = envs.reset(seed=cfg.seed)[0]

    per_rank_gradient_steps = 0
    cumulative_per_rank_gradient_steps = 0
    for iter_num in range(start_iter, total_iters + 1):
        policy_step += policy_steps_per_iter

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
            if iter_num <= learning_starts:
                actions = envs.action_space.sample()
            else:
                # Sample an action given the observation received by the environment
                with torch.inference_mode():
                    torch_obs = prepare_obs(fabric, obs, num_envs=cfg.env.num_envs)
                    actions = player(torch_obs)
                    actions = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions.reshape(envs.action_space.shape))
            rewards = rewards.reshape(cfg.env.num_envs, -1)

        if cfg.metric.log_level > 0 and "final_info" in infos:
            for i, agent_ep_info in enumerate(infos["final_info"]):
                if agent_ep_info is not None:
                    ep_rew = agent_ep_info["episode"]["r"]
                    ep_len = agent_ep_info["episode"]["l"]
                    if aggregator and not aggregator.disabled:
                        aggregator.update("Rewards/rew_avg", ep_rew)
                        aggregator.update("Game/ep_len_avg", ep_len)
                    fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Save the real next observation
        real_next_obs = copy.deepcopy(next_obs)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v
        real_next_obs = np.concatenate([real_next_obs[k] for k in cfg.algo.mlp_keys.encoder], axis=-1).astype(
            np.float32
        )

        step_data["terminated"] = terminated.reshape(1, cfg.env.num_envs, -1).astype(np.uint8)
        step_data["truncated"] = truncated.reshape(1, cfg.env.num_envs, -1).astype(np.uint8)
        step_data["actions"] = actions.reshape(1, cfg.env.num_envs, -1)
        step_data["observations"] = np.concatenate([obs[k] for k in cfg.algo.mlp_keys.encoder], axis=-1)[np.newaxis]
        if not cfg.buffer.sample_next_obs:
            step_data["next_observations"] = real_next_obs[np.newaxis]
        step_data["rewards"] = rewards[np.newaxis]
        rb.add(step_data, validate_args=cfg.buffer.validate_args)

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        if iter_num >= learning_starts:
            per_rank_gradient_steps = (
                ratio((policy_step - prefill_steps + policy_steps_per_iter) / world_size)
                if not cfg.run_benchmarks
                else 1
            )
            if per_rank_gradient_steps > 0:
                # We sample one time to reduce the communications between processes
                sample = rb.sample_tensors(
                    batch_size=per_rank_gradient_steps * cfg.algo.per_rank_batch_size,
                    sample_next_obs=cfg.buffer.sample_next_obs,
                    dtype=None,
                    device=device,
                    from_numpy=cfg.buffer.from_numpy,
                )  # [G*B]
                gathered_data: Dict[str, torch.Tensor] = fabric.all_gather(sample)  # [World, G*B]
                for k, v in gathered_data.items():
                    gathered_data[k] = v.float()  # [G*B*World]
                    if fabric.world_size > 1:
                        gathered_data[k] = gathered_data[k].flatten(start_dim=0, end_dim=2)
                    else:
                        gathered_data[k] = gathered_data[k].flatten(start_dim=0, end_dim=1)
                idxes_to_sample = list(range(next(iter(gathered_data.values())).shape[0]))
                if world_size > 1:
                    dist_sampler: DistributedSampler = DistributedSampler(
                        idxes_to_sample,
                        num_replicas=world_size,
                        rank=fabric.global_rank,
                        shuffle=True,
                        seed=cfg.seed,
                        drop_last=False,
                    )
                    sampler: BatchSampler = BatchSampler(
                        sampler=dist_sampler, batch_size=cfg.algo.per_rank_batch_size, drop_last=False
                    )
                else:
                    sampler = BatchSampler(
                        sampler=idxes_to_sample, batch_size=cfg.algo.per_rank_batch_size, drop_last=False
                    )

                # Start training
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    for batch_idxes in sampler:
                        batch = {k: v[batch_idxes] for k, v in gathered_data.items()}
                        train(
                            fabric,
                            agent,
                            actor_optimizer,
                            qf_optimizer,
                            alpha_optimizer,
                            batch,
                            aggregator,
                            iter_num,
                            cfg,
                            policy_steps_per_iter,
                        )
                        cumulative_per_rank_gradient_steps += 1
                    train_step += world_size

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or iter_num == total_iters):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
            )

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics and timer_metrics["Time/train_time"] > 0:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
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
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            iter_num == total_iters and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "agent": agent.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "ratio": ratio.state_dict(),
                "iter_num": iter_num * fabric.world_size,
                "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"agent": agent}
        register_model(fabric, log_models, cfg, models_to_log)
