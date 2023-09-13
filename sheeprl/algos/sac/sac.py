import os
import pathlib
import time
import warnings
from math import prod
from typing import Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric

from sheeprl.algos.sac.agent import SACActor, SACAgent, SACCritic
from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.algos.sac.utils import test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm


def train(
    fabric: Fabric,
    agent: SACAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    update: int,
    cfg: DictConfig,
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
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize Fabric
    fabric = Fabric(callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
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
        cfg.run_name = f"resume_from_checkpoint_{run_name}"

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, cfg, "sac")
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg.env.id,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank,
                cfg.env.capture_video,
                logger.log_dir if rank == 0 else None,
                "train",
                mask_velocities=False,
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the SAC agent")
    if len(envs.single_observation_space.shape) > 1:
        raise ValueError(
            "Only environments with vector-only observations are supported by the SAC agent. "
            f"Provided environment: {cfg.env.id}"
        )

    # Define the agent and the optimizer and setup sthem with Fabric
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = prod(envs.single_observation_space.shape)
    actor = SACActor(
        observation_dim=obs_dim,
        action_dim=act_dim,
        hidden_size=cfg.algo.actor.hidden_size,
        action_low=envs.single_action_space.low,
        action_high=envs.single_action_space.high,
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

    # Metrics
    aggregator = MetricAggregator(
        {
            "Rewards/rew_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Game/ep_len_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Time/step_per_second": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/value_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/policy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/alpha_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
        }
    )
    aggregator.to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        device=device,
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
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=device)

    # Global variables
    start_step = state["update"] // fabric.world_size if cfg.checkpoint.resume_from else 1
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    start_time = time.perf_counter()
    policy_steps_per_update = int(cfg.env.num_envs * fabric.world_size)
    num_updates = int(cfg.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from and not cfg.buffer.checkpoint:
        learning_starts += start_step

    # Warning for log and checkpoint every
    if cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The log every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    # Get the first environment observation and start the optimization
    obs = torch.tensor(
        envs.reset(seed=cfg.seed)[0],
        dtype=torch.float32,
        device=device,
    )  # [N_envs, N_obs]

    for update in range(start_step, num_updates + 1):
        if update < learning_starts:
            actions = envs.action_space.sample()
        else:
            # Sample an action given the observation received by the environment
            with torch.no_grad():
                actions, _ = agent.actor.module(obs)
                actions = actions.cpu().numpy()
        next_obs, rewards, dones, truncated, infos = envs.step(actions)
        dones = np.logical_or(dones, truncated)

        policy_step += policy_steps_per_update

        if "final_info" in infos:
            for i, agent_final_info in enumerate(infos["final_info"]):
                if agent_final_info is not None and "episode" in agent_final_info:
                    fabric.print(
                        f"Rank-0: policy_step={policy_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                    )
                    aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                    aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

        # Save the real next observation
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    real_next_obs[idx] = final_obs

        with device:
            real_next_obs = torch.tensor(real_next_obs, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32).view(cfg.env.num_envs, -1)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(cfg.env.num_envs, -1)
            dones = torch.tensor(dones, dtype=torch.float32).view(cfg.env.num_envs, -1)

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        if not cfg.buffer.sample_next_obs:
            step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = torch.tensor(next_obs, device=device)

        # Train the agent
        if update >= learning_starts - 1:
            training_steps = learning_starts if update == learning_starts - 1 else 1
            for _ in range(training_steps):
                # We sample one time to reduce the communications between processes
                sample = rb.sample(
                    cfg.algo.per_rank_gradient_steps * cfg.per_rank_batch_size,
                    sample_next_obs=cfg.buffer.sample_next_obs,
                )  # [G*B, 1]
                gathered_data = fabric.all_gather(sample.to_dict())  # [G*B, World, 1]
                gathered_data = make_tensordict(gathered_data).view(-1)  # [G*B*World]
                if fabric.world_size > 1:
                    dist_sampler: DistributedSampler = DistributedSampler(
                        range(len(gathered_data)),
                        num_replicas=fabric.world_size,
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
                        sampler=range(len(gathered_data)), batch_size=cfg.per_rank_batch_size, drop_last=False
                    )
                for batch_idxes in sampler:
                    train(
                        fabric,
                        agent,
                        actor_optimizer,
                        qf_optimizer,
                        alpha_optimizer,
                        gathered_data[batch_idxes],
                        aggregator,
                        update,
                        cfg,
                        policy_steps_per_update,
                    )
        aggregator.update("Time/step_per_second", int(policy_step / (time.perf_counter() - start_time)))
        if policy_step - last_log >= cfg.metric.log_every or cfg.dry_run:
            last_log = policy_step
            fabric.log_dict(aggregator.compute(), policy_step)
            aggregator.reset()

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
        test_env = make_env(
            cfg.env.id,
            None,
            0,
            cfg.env.capture_video,
            fabric.logger.log_dir,
            "test",
            mask_velocities=False,
            vector_env_idx=0,
        )()
        test(agent.actor.module, test_env, fabric, cfg)


if __name__ == "__main__":
    main()
