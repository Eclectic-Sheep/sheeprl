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
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.optim import Adam, Optimizer
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MeanMetric

from fabricrl.algos.droq.args import DROQArgs
from fabricrl.algos.ppo.utils import make_env
from fabricrl.algos.sac.agent import Actor, Critic, SACAgent
from fabricrl.algos.sac.loss import entropy_loss, policy_loss
from fabricrl.algos.sac.sac import test
from fabricrl.data.buffers import ReplayBuffer
from fabricrl.utils.metric import MetricAggregator
from fabricrl.utils.parser import HfArgumentParser

__all__ = ["main"]


def train(
    fabric: Fabric,
    agent: SACAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: DROQArgs,
):
    for _ in range(args.gradient_steps):
        # Update the soft-critic
        next_target_qf_value = agent.get_next_target_q_value(
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
            agent.qfs_target_ema()

    # Update the actor
    actions, logprobs = agent.get_action_and_log_prob(data["observations"])
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

    # Set logger only on rank-0
    if rank == 0:
        run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        logger = TensorBoardLogger(
            root_dir=os.path.join("logs", "droq", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")), name=run_name
        )
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(asdict(args))

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
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    actor = fabric.setup_module(Actor(envs))
    critics = [fabric.setup_module(Critic(envs)) for _ in range(args.num_critics)]
    target_entropy = -prod(envs.single_action_space.shape)
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
    rb = ReplayBuffer(args.buffer_size // int(args.num_envs * fabric.world_size), args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    start_time = time.time()
    num_updates = int(args.total_steps // (args.num_envs * fabric.world_size))
    args.learning_starts = args.learning_starts // int(args.num_envs * fabric.world_size)
    if args.learning_starts <= 1:
        args.learning_starts = 2

    with device:
        # Get the first environment observation and start the optimization
        obs = torch.tensor(envs.reset(seed=args.seed)[0])  # [N_envs, N_obs]

    for global_step in range(num_updates):
        # Sample an action given the observation received by the environment
        with torch.inference_mode():
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
            next_obs = torch.tensor(real_next_obs)
            actions = torch.tensor(actions).view(args.num_envs, -1)
            rewards = torch.tensor(rewards).view(args.num_envs, -1).float()  # [N_envs, 1]
            dones = torch.tensor(dones).view(args.num_envs, -1).float()

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        if global_step > args.learning_starts:
            local_data = rb.sample(args.batch_size // fabric.world_size)
            gathered_data = fabric.all_gather(local_data.to_dict())
            gathered_data = make_tensordict(gathered_data).view(-1)
            if fabric.world_size > 1:
                sampler = DistributedSampler(
                    range(len(gathered_data)),
                    num_replicas=fabric.world_size,
                    rank=fabric.global_rank,
                    shuffle=True,
                    seed=args.seed,
                    drop_last=False,
                )
                gathered_data = gathered_data[next(iter(sampler))]
            train(fabric, agent, actor_optimizer, qf_optimizer, alpha_optimizer, gathered_data, aggregator, args)
        aggregator.update("Time/step_per_second", int(global_step / (time.time() - start_time)))
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

    envs.close()
    if fabric.is_global_zero:
        test(actor.module, device, fabric.logger.experiment, args)


if __name__ == "__main__":
    main()
