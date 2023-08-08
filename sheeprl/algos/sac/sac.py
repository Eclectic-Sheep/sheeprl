import os
import time
from dataclasses import asdict
from math import prod
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.optim import Adam, Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric

from sheeprl.algos.sac.agent import SACActor, SACAgent, SACCritic
from sheeprl.algos.sac.args import SACArgs
from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.algos.sac.utils import test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm


def train(
    fabric: Fabric,
    agent: SACAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    global_step: int,
    args: SACArgs,
    group: Optional[CollectibleGroup] = None,
):
    # Update the soft-critic
    next_target_qf_value = agent.get_next_target_q_values(
        data["next_observations"],
        data["rewards"],
        data["dones"],
        args.gamma,
    )
    qf_values = agent.get_q_values(data["observations"], data["actions"])
    qf_loss = critic_loss(qf_values, next_target_qf_value, agent.num_critics)
    qf_optimizer.zero_grad(set_to_none=True)
    fabric.backward(qf_loss)
    qf_optimizer.step()
    aggregator.update("Loss/value_loss", qf_loss)

    # Update the target networks with EMA
    if global_step % args.target_network_frequency == 0:
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
def main():
    parser = HfArgumentParser(SACArgs)
    args: SACArgs = parser.parse_args_into_dataclasses()[0]

    # Initialize Fabric
    fabric = Fabric(callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, args, "sac")
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(asdict(args))

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if args.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args.capture_video,
                logger.log_dir if rank == 0 else None,
                "train",
                mask_velocities=False,
                vector_env_idx=i,
            )
            for i in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the SAC agent")
    if len(envs.single_observation_space.shape) > 1:
        raise ValueError(
            f"Only environments with vector-only observations are supported by the SAC agent. Provided environment: {args.env_id}"
        )

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = prod(envs.single_observation_space.shape)
    actor = fabric.setup_module(
        SACActor(
            observation_dim=obs_dim,
            action_dim=act_dim,
            hidden_size=args.actor_hidden_size,
            action_low=envs.single_action_space.low,
            action_high=envs.single_action_space.high,
        )
    )
    critics = [
        fabric.setup_module(
            SACCritic(observation_dim=obs_dim + act_dim, hidden_size=args.critic_hidden_size, num_critics=1)
        )
        for _ in range(args.num_critics)
    ]
    target_entropy = -act_dim
    agent = SACAgent(actor, critics, target_entropy, alpha=args.alpha, tau=args.tau, device=fabric.device)

    # Optimizers
    qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        Adam(agent.qfs.parameters(), lr=args.q_lr, eps=1e-4),
        Adam(agent.actor.parameters(), lr=args.policy_lr, eps=1e-4),
        Adam([agent.log_alpha], lr=args.alpha_lr, eps=1e-4),
    )

    # Metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(),
                "Game/ep_len_avg": MeanMetric(),
                "Time/step_per_second": MeanMetric(),
                "Loss/value_loss": MeanMetric(),
                "Loss/policy_loss": MeanMetric(),
                "Loss/alpha_loss": MeanMetric(),
            }
        )

    # Local data
    buffer_size = args.buffer_size // int(args.num_envs * fabric.world_size) if not args.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        args.num_envs,
        device=device,
        memmap=args.memmap_buffer,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    start_time = time.perf_counter()
    num_updates = int(args.total_steps // (args.num_envs * fabric.world_size)) if not args.dry_run else 1
    args.learning_starts = args.learning_starts // int(args.num_envs * fabric.world_size) if not args.dry_run else 0

    with device:
        # Get the first environment observation and start the optimization
        obs = torch.tensor(envs.reset(seed=args.seed)[0], dtype=torch.float32)  # [N_envs, N_obs]

    for global_step in range(1, num_updates + 1):
        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            # Sample an action given the observation received by the environment
            with torch.no_grad():
                actions, _ = actor.module(obs)
                actions = actions.cpu().numpy()
        next_obs, rewards, dones, truncated, infos = envs.step(actions)
        dones = np.logical_or(dones, truncated)

        if "final_info" in infos:
            for i, agent_final_info in enumerate(infos["final_info"]):
                if agent_final_info is not None and "episode" in agent_final_info:
                    fabric.print(
                        f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
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
            actions = torch.tensor(actions, dtype=torch.float32).view(args.num_envs, -1)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(args.num_envs, -1)
            dones = torch.tensor(dones, dtype=torch.float32).view(args.num_envs, -1)

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        if not args.sample_next_obs:
            step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        if global_step >= args.learning_starts - 1:
            training_steps = args.learning_starts if global_step == args.learning_starts - 1 else 1
            for _ in range(training_steps):
                # We sample one time to reduce the communications between processes
                sample = rb.sample(
                    args.gradient_steps * args.per_rank_batch_size, sample_next_obs=args.sample_next_obs
                )  # [G*B, 1]
                gathered_data = fabric.all_gather(sample.to_dict())  # [G*B, World, 1]
                gathered_data = make_tensordict(gathered_data).view(-1)  # [G*B*World]
                if fabric.world_size > 1:
                    dist_sampler: DistributedSampler = DistributedSampler(
                        range(len(gathered_data)),
                        num_replicas=fabric.world_size,
                        rank=fabric.global_rank,
                        shuffle=True,
                        seed=args.seed,
                        drop_last=False,
                    )
                    sampler: BatchSampler = BatchSampler(
                        sampler=dist_sampler, batch_size=args.per_rank_batch_size, drop_last=False
                    )
                else:
                    sampler = BatchSampler(
                        sampler=range(len(gathered_data)), batch_size=args.per_rank_batch_size, drop_last=False
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
                        global_step,
                        args,
                    )
        aggregator.update("Time/step_per_second", int(global_step / (time.perf_counter() - start_time)))
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint model
        if (
            (args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0)
            or args.dry_run
            or global_step == num_updates
        ):
            state = {
                "agent": agent.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "args": asdict(args),
                "global_step": global_step,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{global_step}_{fabric.global_rank}.ckpt")
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if args.checkpoint_buffer else None,
            )

    envs.close()
    if fabric.is_global_zero:
        test_env = make_env(
            args.env_id,
            None,
            0,
            args.capture_video,
            fabric.logger.log_dir,
            "test",
            mask_velocities=False,
            vector_env_idx=0,
        )()
        test(actor.module, test_env, fabric, args)


if __name__ == "__main__":
    main()
