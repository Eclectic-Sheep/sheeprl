# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch.optim import Optimizer

from fabricrl.algos.ppo.utils import make_env
from fabricrl.algos.sac.agent import SACAgent
from fabricrl.algos.sac.args import parse_args
from fabricrl.data.buffers import ReplayBuffer


def train(
    fabric: Fabric,
    agent: SACAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    data: TensorDictBase,
    global_step: int,
    args: argparse.Namespace,
):
    if global_step > args.learning_starts:
        alpha = agent.log_alpha.exp().item()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = agent.actor.get_action(data["next_observations"])
            qf_next_target = agent.qf_target(data["next_observations"], next_state_actions)
            min_qf_next_target = torch.min(qf_next_target, dim=-1, keepdim=True)[0] - alpha * next_state_log_pi
            next_qf_value = data["rewards"] + (~data["dones"]) * args.gamma * (min_qf_next_target)

        qf_values = agent.qf(data["observations"], data["actions"])
        qf_loss = (
            1
            / agent.num_critics
            * sum(
                F.mse_loss(qf_values[..., qf_value_idx].unsqueeze(-1), next_qf_value)
                for qf_value_idx in range(agent.num_critics)
            )
        )

        qf_optimizer.zero_grad()
        fabric.backward(qf_loss)
        qf_optimizer.step()

        if global_step % args.policy_frequency == 0:  # TD-3 Delayed update support
            for _ in range(args.policy_frequency):
                pi, log_pi, _ = agent.actor.get_action(data["observations"])
                qf_pi = agent.qf(data["observations"], pi)
                min_qf_pi = torch.min(qf_pi, dim=-1, keepdim=True)[0]
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                fabric.backward(actor_loss)
                actor_optimizer.step()

                with torch.no_grad():
                    _, log_pi, _ = agent.actor.get_action(data["observations"])
                alpha_loss = (-agent.log_alpha * (log_pi + agent.target_entropy)).mean()

                alpha_optimizer.zero_grad()
                fabric.backward(alpha_loss)
                alpha_optimizer.step()

        # update the target networks
        if global_step % args.target_network_frequency == 0:
            agent.qf_target_ema()


def main(args: argparse.Namespace):
    args.num_envs = 4

    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    logger = TensorBoardLogger(
        root_dir=os.path.join("logs", "sac", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
        name=run_name,
    )

    # Initialize Fabric
    fabric = Fabric(loggers=logger)
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    fabric.world_size
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
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args.capture_video,
                logger.log_dir,
                "train",
                mask_velocities=False,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    agent: SACAgent = fabric.setup_module(SACAgent(envs, num_critics=2, tau=args.tau))
    qf_optimizer = fabric.setup_optimizers(optim.Adam(agent.qf.parameters(), lr=args.q_lr))
    actor_optimizer = fabric.setup_optimizers(optim.Adam(list(agent.actor.parameters()), lr=args.policy_lr))
    alpha_optimizer = optim.Adam([agent.log_alpha], lr=args.q_lr)

    # Player metrics
    with device:
        rew_avg = torchmetrics.MeanMetric()
        ep_len_avg = torchmetrics.MeanMetric()

    # Local data
    rb = ReplayBuffer(args.buffer_size, args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    time.time()
    with device:
        # Get the first environment observation and start the optimization
        obs = torch.tensor(envs.reset(seed=args.seed)[0])  # [N_envs, N_obs]

    for global_step in range(args.total_timesteps):
        # Sample an action given the observation received by the environment
        # or play randomly
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _, _ = agent.actor.get_action(obs)
                actions = actions.cpu().numpy()
        next_obs, rewards, dones, truncated, infos = envs.step(actions)
        dones = np.logical_or(dones, truncated)

        if "final_info" in infos:
            for i, agent_final_info in enumerate(infos["final_info"]):
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

        # Save the real next observation
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    real_next_obs[idx] = final_obs

        with device:
            obs = torch.tensor(obs)
            next_obs = torch.tensor(real_next_obs)
            actions = torch.tensor(actions).view(args.num_envs, -1)
            rewards = torch.tensor(rewards).view(args.num_envs, -1).float()  # [N_envs, 1]
            dones = torch.tensor(dones).view(args.num_envs, -1)

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        data = rb.sample(args.batch_size)
        train(fabric, agent, actor_optimizer, qf_optimizer, alpha_optimizer, data, global_step, args)
    envs.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
