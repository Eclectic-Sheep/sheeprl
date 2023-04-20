"""
Proximal Policy Optimization (PPO) - Accelerated with Lightning Fabric

Author: Federico Belotti @belerico
Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
Based on the paper: https://arxiv.org/abs/1707.06347

Requirements:
- gymnasium[box2d]>=0.27.1
- moviepy
- lightning
- torchmetrics
- tensorboard


Run it with:
    lightning run model --accelerator=cpu --strategy=ddp --devices=2 train_fabric.py
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict

import gymnasium as gym
import torch
import torchmetrics
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from tensordict import TensorDict, make_tensordict
from torch import Tensor
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from fabricrl.algos.ppo.agent import PPOAgent
from fabricrl.algos.ppo.args import parse_args
from fabricrl.algos.ppo.utils import make_env
from fabricrl.data import ReplayBuffer
from fabricrl.utils.utils import linear_annealing


@torch.no_grad()
def test(agent: "PPOAgent", device: torch.device, logger: SummaryWriter, args: argparse.Namespace):
    env = make_env(args.env_id, args.seed, 0, args.capture_video, logger.log_dir, "test")()
    step = 0
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=device)
    while not done:
        # Act greedly through the environment
        action = agent.get_greedy_action(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=device)
        step += 1
    logger.add_scalar("Test/cumulative_reward", cumulative_rew, 0)
    env.close()


def train(
    fabric: Fabric,
    agent: PPOAgent,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, Tensor],
    global_step: int,
    args: argparse.Namespace,
):
    indexes = list(range(data["observations"].shape[0]))
    if args.share_data:
        sampler = DistributedSampler(
            indexes, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True, seed=args.seed
        )
    else:
        sampler = RandomSampler(indexes)
    sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)

    for epoch in range(args.update_epochs):
        if args.share_data:
            sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            loss = agent.training_step(data[batch_idxes])
            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(agent, optimizer, max_norm=args.max_grad_norm)
            optimizer.step()
        agent.on_train_epoch_end(global_step)


def main(args: argparse.Namespace):
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    logger = TensorBoardLogger(
        root_dir=os.path.join("logs", "fabric_logs", "ppo", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
        name=run_name,
    )

    # Initialize Fabric
    fabric = Fabric(loggers=logger)
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Log hyperparameters
    fabric.logger.experiment.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id, args.seed + rank * args.num_envs + i, rank, args.capture_video, logger.log_dir, "train"
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    agent: PPOAgent = PPOAgent(
        envs,
        act_fun=args.activation_function,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ortho_init=args.ortho_init,
        normalize_advantages=args.normalize_advantages,
    )
    optimizer = agent.configure_optimizers(args.learning_rate)
    agent, optimizer = fabric.setup(agent, optimizer)

    # Player metrics
    rew_avg = torchmetrics.MeanMetric().to(device)
    ep_len_avg = torchmetrics.MeanMetric().to(device)

    # Local data
    rb = ReplayBuffer(args.num_steps, args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_rollout = int(args.num_envs * args.num_steps * world_size)
    num_updates = args.total_timesteps // single_global_rollout

    with device:
        # Get the first environment observation and start the optimization
        next_obs = torch.tensor(envs.reset(seed=args.seed)[0])  # [N_envs, N_obs]
        next_done = torch.zeros(args.num_envs, 1)  # [N_envs, 1]

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            linear_annealing(optimizer, update, num_updates, args.learning_rate)
        fabric.log("Info/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        for _ in range(0, args.num_steps):
            global_step += args.num_envs * world_size

            # Sample an action given the observation received by the environment
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            step_data["dones"] = next_done
            step_data["values"] = value
            step_data["actions"] = action
            step_data["logprobs"] = logprob
            step_data["observations"] = next_obs

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            with device:
                next_obs = torch.tensor(next_obs)
                next_done = (
                    torch.logical_or(torch.tensor(done), torch.tensor(truncated)).view(args.num_envs, -1).float()
                )  # [N_envs, 1]

                # Save reward for the last (observation, action) pair
                step_data["rewards"] = torch.tensor(reward).view(args.num_envs, -1)  # [N_envs, 1]

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                        )
                        rew_avg(agent_final_info["episode"]["r"][0])
                        ep_len_avg(agent_final_info["episode"]["l"][0])

        # Sync the metrics
        rew_avg_reduced = rew_avg.compute()
        if not rew_avg_reduced.isnan():
            fabric.log("Rewards/rew_avg", rew_avg_reduced, global_step)
        ep_len_avg_reduced = ep_len_avg.compute()
        if not ep_len_avg_reduced.isnan():
            fabric.log("Game/ep_len_avg", ep_len_avg_reduced, global_step)
        rew_avg.reset()
        ep_len_avg.reset()

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        returns, advantages = agent.estimate_returns_and_advantages(
            rb["rewards"], rb["values"], rb["dones"], next_obs, next_done, args.num_steps, args.gamma, args.gae_lambda
        )

        # Add returns and advantages to the buffer
        rb["returns"] = returns.float()
        rb["advantages"] = advantages.float()

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        if args.share_data and fabric.world_size > 1:
            # Gather all the tensors from all the world and reshape them
            gathered_data = fabric.all_gather(
                local_data.to_dict()
            )  # Fabric does not work with TensorDict: I'll open them a PR!
            gathered_data = make_tensordict(gathered_data).view(-1)
        else:
            gathered_data = local_data

        # Train the agent
        train(fabric, agent, optimizer, gathered_data, global_step, args)
        fabric.log("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    if fabric.is_global_zero:
        test(agent.module, device, fabric.logger.experiment, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
