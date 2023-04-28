import os
from typing import Optional

import gymnasium as gym

from fabricrl.envs.wrappers import MaskVelocityWrapper


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
        env = gym.make(env_id, render_mode="rgb_array")  # , continuous=True)
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
