import os
import time
from dataclasses import asdict
from datetime import datetime
from math import prod

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from tensordict import TensorDict, make_tensordict
from torch.optim import Adam, Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric

from sheeprl.algos.droq.agent import DROQAgent, DROQCritic
from sheeprl.algos.droq.args import DROQArgs
from sheeprl.algos.sac.agent import SACActor
from sheeprl.algos.sac.loss import entropy_loss, policy_loss
from sheeprl.algos.sac.sac import test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import make_env


def train(
    fabric: Fabric,
    agent: DROQAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    rb: ReplayBuffer,
    aggregator: MetricAggregator,
    args: DROQArgs,
):
    # Sample a minibatch in a distributed way: Line 5 - Algorithm 2
    # We sample one time to reduce the communications between processes
    sample = rb.sample(args.gradient_steps * args.per_rank_batch_size)
    gathered_data = fabric.all_gather(sample.to_dict())
    gathered_data = make_tensordict(gathered_data).view(-1)
    if fabric.world_size > 1:
        dist_sampler: DistributedSampler = DistributedSampler(
            range(len(gathered_data)),
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        sampler: BatchSampler = BatchSampler(sampler=dist_sampler, batch_size=args.per_rank_batch_size, drop_last=False)
    else:
        sampler = BatchSampler(sampler=range(len(gathered_data)), batch_size=args.per_rank_batch_size, drop_last=False)

    # Update the soft-critic
    for batch_idxes in sampler:
        data = gathered_data[batch_idxes]
        next_target_qf_value = agent.get_next_target_q_values(
            data["next_observations"],
            data["rewards"],
            data["dones"],
            args.gamma,
        )
        for qf_value_idx in range(agent.num_critics):
            # Line 8 - Algorithm 2
            qf_loss = F.mse_loss(
                agent.get_ith_q_value(data["observations"], data["actions"], qf_value_idx), next_target_qf_value
            )
            qf_optimizer.zero_grad(set_to_none=True)
            fabric.backward(qf_loss)
            qf_optimizer.step()
            aggregator.update("Loss/value_loss", qf_loss)

            # Update the target networks with EMA
            agent.qfs_target_ema(critic_idx=qf_value_idx)

    # Sample a different minibatch in a distributed way to update actor and alpha parameter
    sample = rb.sample(args.per_rank_batch_size)
    data = fabric.all_gather(sample.to_dict())
    data = make_tensordict(data).view(-1)
    if fabric.world_size > 1:
        sampler: DistributedSampler = DistributedSampler(
            range(len(data)),
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        data = data[next(iter(sampler))]

    # Update the actor
    actions, logprobs = agent.get_actions_and_log_probs(data["observations"])
    qf_values = agent.get_q_values(data["observations"], actions)
    min_qf_values = torch.mean(qf_values, dim=-1, keepdim=True)
    actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
    actor_optimizer.zero_grad(set_to_none=True)
    fabric.backward(actor_loss)
    actor_optimizer.step()
    aggregator.update("Loss/policy_loss", actor_loss)

    # Update the entropy value
    alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
    alpha_optimizer.zero_grad(set_to_none=True)
    fabric.backward(alpha_loss)
    agent.log_alpha.grad = fabric.all_reduce(agent.log_alpha.grad)
    alpha_optimizer.step()
    aggregator.update("Loss/alpha_loss", alpha_loss)


@register_algorithm()
def main():
    parser = HfArgumentParser(DROQArgs)
    args: DROQArgs = parser.parse_args_into_dataclasses()[0]

    # Initialize Fabric
    fabric = Fabric()
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

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
            args.root_dir
            if args.root_dir is not None
            else os.path.join("logs", "droq", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        run_name = (
            args.run_name
            if args.run_name is not None
            else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        )
        logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
        fabric._loggers = [logger]
        log_dir = logger.log_dir
        fabric.logger.log_hyperparams(asdict(args))
        if fabric.world_size > 1:
            world_collective.broadcast_object_list([log_dir], src=0)
    else:
        data = [None]
        world_collective.broadcast_object_list(data, src=0)
        log_dir = data[0]
        os.makedirs(log_dir, exist_ok=True)

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
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
        raise ValueError("only continuous action space is supported")

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = prod(envs.single_observation_space.shape)
    actor = fabric.setup_module(
        SACActor(
            observation_dim=obs_dim,
            action_dim=act_dim,
            action_low=envs.single_action_space.low,
            action_high=envs.single_action_space.high,
        )
    )
    critics = [
        fabric.setup_module(DROQCritic(observation_dim=obs_dim + act_dim, num_critics=1, dropout=args.dropout))
        for _ in range(args.num_critics)
    ]
    target_entropy = -act_dim
    agent = DROQAgent(actor, critics, target_entropy, alpha=args.alpha, tau=args.tau, device=fabric.device)

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
    rb = ReplayBuffer(buffer_size, args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    start_time = time.time()
    num_updates = int(args.total_steps // (args.num_envs * fabric.world_size)) if not args.dry_run else 1
    args.learning_starts = args.learning_starts // int(args.num_envs * fabric.world_size) if not args.dry_run else 0

    with device:
        # Get the first environment observation and start the optimization
        obs = torch.tensor(envs.reset(seed=args.seed)[0], dtype=torch.float32)  # [N_envs, N_obs]

    for global_step in range(1, num_updates + 1):
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
            rewards = torch.tensor(rewards, dtype=torch.float32).view(args.num_envs, -1)  # [N_envs, 1]
            dones = torch.tensor(dones, dtype=torch.float32).view(args.num_envs, -1)

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = real_next_obs

        # Train the agent
        if global_step > args.learning_starts:
            train(fabric, agent, actor_optimizer, qf_optimizer, alpha_optimizer, rb, aggregator, args)
        aggregator.update("Time/step_per_second", int(global_step / (time.time() - start_time)))
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint model
        if (args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0) or args.dry_run:
            state = {
                "agent": agent.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "args": asdict(args),
                "global_step": global_step,
            }
            if args.checkpoint_buffer:
                true_done = rb["dones"][(rb._pos - 1) % rb.buffer_size, :].clone()
                rb["dones"][(rb._pos - 1) % rb.buffer_size, :] = True
                state["rb"] = rb
                if fabric.world_size > 1:
                    # We need to collect the buffers from all the ranks
                    # The collective it is needed because the `gather_object` function is not implemented in Fabric
                    checkpoint_collective = TorchCollective()
                    # gloo is the torch.distributed backend that works on cpu
                    if rb.device == torch.device("cpu"):
                        backend = "gloo"
                    else:
                        backend = "nccl"
                    checkpoint_collective.create_group(backend=backend, ranks=list(range(fabric.world_size)))
                    gathered_rb = [None for _ in range(fabric.world_size)]
                    if fabric.global_rank == 0:
                        checkpoint_collective.gather_object(rb, gathered_rb)
                        state["rb"] = gathered_rb
                    else:
                        checkpoint_collective.gather_object(rb, None)
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{global_step}_{fabric.global_rank}.ckpt")
            fabric.save(ckpt_path, state)
            if args.checkpoint_buffer:
                rb["dones"][(rb._pos - 1) % rb.buffer_size, :] = true_done

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
