import os
from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

from fabricrl.algos.ppo.args import PPOArgs
from fabricrl.envs.wrappers import MaskVelocityWrapper


@torch.inference_mode()
def test(actor: nn.Module, device: torch.device, logger: SummaryWriter, args: PPOArgs):
    env = SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, logger.log_dir, "test", mask_velocities=args.mask_vel)]
    )
    step = 0
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=device)
    while not done:
        # Act greedly through the environment
        logits = actor(next_obs)
        action = F.softmax(logits, dim=-1).argmax(dim=-1)

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
        env = gym.make(env_id, render_mode="rgb_array")
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


def make_atari_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: Optional[str] = None,
    prefix: str = "",
    mask_velocities: bool = False,
):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
