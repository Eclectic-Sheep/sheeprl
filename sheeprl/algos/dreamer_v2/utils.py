import os
from typing import TYPE_CHECKING, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor, nn
from torch.distributions import OneHotCategoricalStraightThrough

from sheeprl.utils.utils import get_dummy_env

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v2.agent import Player

from sheeprl.algos.dreamer_v2.args import DreamerV2Args
from sheeprl.envs.wrappers import ActionRepeat


def make_env(
    env_id: str,
    seed: int,
    rank: int,
    args: DreamerV2Args,
    run_name: Optional[str] = None,
    prefix: str = "",
) -> gym.Env:
    """
    Create the callable function to createenvironment and
    force the environment to return only pixels observations.

    Args:
        env_id (str): the id of the environment to initialize.
        seed (int): the seed to use.
        rank (int): the rank of the process.
        args (DreamerV2Args): the configs of the experiment.
        run_name (str, optional): the name of the run.
            Default to None.
        prefix (str): the prefix to add to the video folder.
            Default to "".

    Returns:
        The callable function that initializes the environment.
    """
    env_spec = ""
    _env_id = env_id.lower()
    if "dummy" in _env_id:
        env = get_dummy_env(_env_id)
    elif "dmc" in _env_id:
        from sheeprl.envs.dmc import DMCWrapper

        _, domain, task = _env_id.split("_")
        env = DMCWrapper(
            domain,
            task,
            from_pixels=True,
            height=64,
            width=64,
            frame_skip=args.action_repeat,
            seed=seed,
        )
    elif "minedojo" in _env_id:
        from sheeprl.envs.minedojo import MineDojoWrapper

        task_id = "_".join(env_id.split("_")[1:])
        start_position = (
            {
                "x": args.mine_start_position[0],
                "y": args.mine_start_position[1],
                "z": args.mine_start_position[2],
                "pitch": args.mine_start_position[3],
                "yaw": args.mine_start_position[4],
            }
            if args.mine_start_position is not None
            else None
        )
        env = MineDojoWrapper(
            task_id,
            height=64,
            width=64,
            pitch_limits=(args.mine_min_pitch, args.mine_max_pitch),
            seed=args.seed,
            start_position=start_position,
        )
    else:
        env_spec = gym.spec(env_id).entry_point
        if "mujoco" in env_spec:
            try:
                env = gym.make(env_id, render_mode="rgb_array", terminate_when_unhealthy=False)
            except:
                env = gym.make(env_id, render_mode="rgb_array")
            env.frame_skip = 0
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        if "atari" in env_spec:
            if args.atari_noop_max < 0:
                raise ValueError(
                    f"Negative value of atart_noop_max parameter ({args.atari_noop_max}), the minimum value allowed is 0"
                )
            env = gym.wrappers.AtariPreprocessing(
                env,
                noop_max=args.atari_noop_max,
                frame_skip=args.action_repeat,
                screen_size=64,
                grayscale_obs=args.grayscale_obs,
                scale_obs=False,
                terminal_on_life_loss=False,
                grayscale_newaxis=True,
            )

    # action repeat
    if "atari" not in env_spec:
        env = ActionRepeat(env, args.action_repeat)

    # create dict
    if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) < 3:
        env = gym.wrappers.PixelObservationWrapper(env, pixels_only=False, pixel_keys=("rgb",))
    elif isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 3:
        env = gym.wrappers.TransformObservation(env, lambda obs: {"rgb": obs})
        env.observation_space = gym.spaces.Dict({"rgb": env.observation_space})

    # resize image
    if "atari" not in env_spec and "dmc" not in env_id and "minedojo" not in env_id:
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: obs.update(
                {"rgb": cv2.resize(obs["rgb"], (64, 64), interpolation=cv2.INTER_AREA).reshape(64, 64, 3)}
            )
            or obs,
        )
        env.observation_space["rgb"] = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)

    # grayscale
    if args.grayscale_obs and "atari" not in env_spec:
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: obs.update({"rgb": np.expand_dims(cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2GRAY), -1)}) or obs,
        )
        env.observation_space["rgb"] = gym.spaces.Box(0, 255, (64, 64, 1), np.uint8)

    # channels first
    if "minedojo" not in env_id:
        env = gym.wrappers.TransformObservation(
            env, lambda obs: obs.update({"rgb": obs["rgb"].transpose(2, 0, 1)}) or obs
        )
        env.observation_space["rgb"] = gym.spaces.Box(
            0, 255, (env.observation_space["rgb"].shape[-1], *env.observation_space["rgb"].shape[:2]), np.uint8
        )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if args.max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps // args.action_repeat)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.capture_video and rank == 0 and run_name is not None:
        env = gym.experimental.wrappers.RecordVideoV0(
            env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
        )
        env.metadata["render_fps"] = env.frames_per_sec
    return env


def compute_stochastic_state(
    logits: Tensor,
    discrete: int = 32,
) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = OneHotCategoricalStraightThrough(logits=logits)
    return dist.rsample()
