import argparse
import os
from typing import Optional

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from fabricrl.algos.ppo.agent import PPOAgent
from fabricrl.envs.wrappers import MaskVelocityWrapper


@torch.no_grad()
def test(agent: PPOAgent, device: torch.device, logger: SummaryWriter, args: argparse.Namespace):
    env = make_env(
        args.env_id, args.seed, 0, args.capture_video, logger.log_dir, "test", mask_velocities=args.mask_vel
    )()
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


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: Optional[str] = None,
    prefix: str = "",
    mask_velocities: bool = False,
):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array", continuous=True)
        if mask_velocities:
            env = MaskVelocityWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0 and run_name is not None:
                env = gym.wrappers.RecordVideo(
                    env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
                )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
