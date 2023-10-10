import copy
import os
import pathlib
import warnings
from math import prod
from typing import Any, Dict, Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric, SumMetric

from sheeprl.algos.sac.agent import SACActor, SACAgent, SACCritic
from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.algos.sac.utils import test
from sheeprl.data.buffers_np import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import dotdict


def train(
    fabric: Fabric,
    agent: SACAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    data: Dict[str, torch.Tensor],
    aggregator: MetricAggregator,
    update: int,
    cfg: Dict[str, Any],
    policy_steps_per_update: int,
    group: Optional[CollectibleGroup] = None,
):
    # Update the soft-critic
    next_target_qf_value = agent.get_next_target_q_values(
        data["next_observations"], data["rewards"], data["dones"], cfg.algo.gamma
    )
    qf_values = agent.get_q_values(data["observations"], data["actions"])
    qf_loss = critic_loss(qf_values, next_target_qf_value, agent.num_critics)
    qf_optimizer.zero_grad(set_to_none=True)
    fabric.backward(qf_loss)
    qf_optimizer.step()
    aggregator.update("Loss/value_loss", qf_loss)

    # Update the target networks with EMA
    if update % (cfg.algo.critic.target_network_frequency // policy_steps_per_update + 1) == 0:
        agent.qfs_target_ema()

    # Update the actor
    actions, logprobs = agent.get_actions_and_log_probs(data["observations"])
    qf_values = agent.get_q_values(data["observations"], actions)
    min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
    actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
    actor_optimizer.zero_grad(set_to_none=True)
    fabric.backward(actor_loss)
    actor_optimizer.step()
    aggregator.update("Loss/policy_loss", actor_loss)

    # Update the entropy value
    alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
    alpha_optimizer.zero_grad(set_to_none=True)
    fabric.backward(alpha_loss)
    agent.log_alpha.grad = fabric.all_reduce(agent.log_alpha.grad, group=group)
    alpha_optimizer.step()
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
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
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

    if len(cfg.cnn_keys.encoder) > 0:
        warnings.warn("SAC algorithm cannot allow to use images as observations, the CNN keys will be ignored")
        cfg.cnn_keys.encoder = []

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, cfg)
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)

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
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the SAC agent")
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
    for k in cfg.mlp_keys.encoder:
        if len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the SAC agent. "
                f"Provided environment: {cfg.env.id}"
            )

    # Define the agent and the optimizer and setup sthem with Fabric
    act_dim = prod(action_space.shape)
    obs_dim = sum([prod(observation_space[k].shape) for k in cfg.mlp_keys.encoder])
    actor = SACActor(
        observation_dim=obs_dim,
        action_dim=act_dim,
        distribution_cfg=cfg.distribution,
        hidden_size=cfg.algo.actor.hidden_size,
        action_low=action_space.low,
        action_high=action_space.high,
    )
    critics = [
        SACCritic(observation_dim=obs_dim + act_dim, hidden_size=cfg.algo.critic.hidden_size, num_critics=1)
        for _ in range(cfg.algo.critic.n)
    ]
    target_entropy = -act_dim
    agent = SACAgent(actor, critics, target_entropy, alpha=cfg.algo.alpha.alpha, tau=cfg.algo.tau, device=fabric.device)
    if cfg.checkpoint.resume_from:
        agent.load_state_dict(state["agent"])
    agent.actor = fabric.setup_module(agent.actor)
    agent.critics = [fabric.setup_module(critic) for critic in agent.critics]

    # Optimizers
    qf_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=agent.qfs.parameters())
    actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=agent.actor.parameters())
    alpha_optimizer = hydra.utils.instantiate(cfg.algo.alpha.optimizer, params=[agent.log_alpha])
    if cfg.checkpoint.resume_from:
        qf_optimizer.load_state_dict(state["qf_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        alpha_optimizer.load_state_dict(state["alpha_optimizer"])
    qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        qf_optimizer, actor_optimizer, alpha_optimizer
    )

    # Create a metric aggregator to log the metrics
    aggregator = MetricAggregator(
        {
            "Rewards/rew_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Game/ep_len_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/value_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/policy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/alpha_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
        }
    ).to(device)

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
    step_data = {}

    # Global variables
    last_train = 0
    train_step = 0
    start_step = state["update"] // fabric.world_size if cfg.checkpoint.resume_from else 1
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * fabric.world_size)
    policy_steps_per_update = int(cfg.env.num_envs * world_size)
    num_updates = int(cfg.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from and not cfg.buffer.checkpoint:
        learning_starts += start_step

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
    obs = envs.reset(seed=cfg.seed)[0]
    obs = np.concatenate([obs[k] for k in cfg.mlp_keys.encoder], axis=-1)

    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * world_size

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
            if update <= learning_starts:
                actions = envs.action_space.sample()
            else:
                # Sample an action given the observation received by the environment
                with torch.no_grad():
                    torch_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    actions, _ = agent.actor.module(torch_obs)
                    actions = actions.cpu().numpy()
            next_obs, rewards, dones, truncated, infos = envs.step(actions)
            next_obs = np.concatenate([next_obs[k] for k in cfg.mlp_keys.encoder], axis=-1)
            dones = np.logical_or(dones, truncated).reshape(cfg.env.num_envs, -1).astype(np.uint8)
            rewards = rewards.reshape(cfg.env.num_envs, -1)

        if "final_info" in infos:
            for i, agent_ep_info in enumerate(infos["final_info"]):
                if agent_ep_info is not None:
                    ep_rew = agent_ep_info["episode"]["r"]
                    ep_len = agent_ep_info["episode"]["l"]
                    aggregator.update("Rewards/rew_avg", ep_rew)
                    aggregator.update("Game/ep_len_avg", ep_len)
                    fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Save the real next observation
        real_next_obs = copy.deepcopy(next_obs)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    real_next_obs[idx] = np.concatenate([v for v in final_obs.values()], axis=-1)

        step_data["dones"] = dones[np.newaxis]
        step_data["actions"] = actions[np.newaxis]
        step_data["observations"] = obs[np.newaxis]
        if not cfg.buffer.sample_next_obs:
            step_data["next_observations"] = real_next_obs[np.newaxis]
        step_data["rewards"] = rewards[np.newaxis]
        rb.add(step_data, validate_args=False)

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        if update >= learning_starts:
            training_steps = learning_starts if update == learning_starts else 1

            # We sample one time to reduce the communications between processes
            sample = rb.sample_tensors(
                batch_size=training_steps * cfg.algo.per_rank_gradient_steps * cfg.per_rank_batch_size,
                sample_next_obs=cfg.buffer.sample_next_obs,
                dtype=None,
                device=device,
            )  # [G*B]
            gathered_data: Dict[str, torch.Tensor] = fabric.all_gather(sample)  # [World, G*B]
            for k, v in gathered_data.items():
                gathered_data[k] = v.float()  # [G*B*World]
                if fabric.world_size > 1:
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
                    sampler=dist_sampler, batch_size=cfg.per_rank_batch_size, drop_last=False
                )
            else:
                sampler = BatchSampler(
                    sampler=idxes_to_sample,
                    batch_size=cfg.per_rank_batch_size,
                    drop_last=False,
                )

            # Start training
            with timer("Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute)):
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
                        update,
                        cfg,
                        policy_steps_per_update,
                    )
                train_step += world_size

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
            state = {
                "agent": agent.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "update": update * fabric.world_size,
                "batch_size": cfg.per_rank_batch_size * fabric.world_size,
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
    if fabric.is_global_zero:
        test(agent.actor.module, fabric, cfg)
