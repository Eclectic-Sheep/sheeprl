import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from lightning import Fabric

from fabricrl.algos.ppo.args import PPOArgs
from fabricrl.envs.wrappers import MaskVelocityWrapper


@torch.no_grad()
def test(actor: nn.Module, envs: SyncVectorEnv, fabric: Fabric, args: PPOArgs):
    actor.train(False)
    done = False
    cumulative_rew = 0
    env = envs.envs[0]
    next_obs = torch.tensor(np.array(env.reset(seed=args.seed)[0]), device=fabric.device).unsqueeze(0)
    while not done:
        # Act greedly through the environment
        logits = actor(next_obs)
        action = F.softmax(logits, dim=-1).argmax(dim=-1)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(np.array(next_obs), device=fabric.device).unsqueeze(0)

        if args.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
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
