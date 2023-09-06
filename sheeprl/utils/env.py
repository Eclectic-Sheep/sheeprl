import os
import warnings
from typing import Any, Dict, Optional

import cv2
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig

from sheeprl.envs.wrappers import ActionRepeat, FrameStack, MaskVelocityWrapper
from sheeprl.utils.imports import _IS_DIAMBRA_ARENA_AVAILABLE, _IS_DIAMBRA_AVAILABLE, _IS_DMC_AVAILABLE

if _IS_DIAMBRA_ARENA_AVAILABLE and _IS_DIAMBRA_AVAILABLE:
    from sheeprl.envs.diambra_wrapper import DiambraWrapper
if _IS_DMC_AVAILABLE:
    from sheeprl.envs.dmc import DMCWrapper


def make_env(
    env_id: str,
    seed: Optional[int],
    idx: int,
    capture_video: bool,
    run_name: Optional[str] = None,
    prefix: str = "",
    mask_velocities: bool = False,
    vector_env_idx: int = 0,
    action_repeat: int = 1,
):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if mask_velocities:
            env = MaskVelocityWrapper(env)
        env = ActionRepeat(env, action_repeat)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if vector_env_idx == 0 and idx == 0 and run_name is not None:
                env = gym.experimental.wrappers.RecordVideoV0(
                    env,
                    os.path.join(run_name, prefix + "_videos" if prefix else "videos"),
                    disable_logger=True,
                )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_dict_env(
    cfg: DictConfig,
    seed: int,
    rank: int,
    run_name: Optional[str] = None,
    prefix: str = "",
    vector_env_idx: int = 0,
) -> gym.Env:
    """
    Create the callable function to createenvironment and
    force the environment to return only pixels observations.

    Args:
        cfg (str): the configs of the environment to initialize.
        seed (int): the seed to use.
        rank (int): the rank of the process.
        run_name (str, optional): the name of the run.
            Default to None.
        prefix (str): the prefix to add to the video folder.
            Default to "".
        vector_env_idx (int): the index of the environment.

    Returns:
        The callable function that initializes the environment.
    """

    def thunk():
        try:
            env_spec = gym.spec(cfg.env.id).entry_point
        except Exception:
            env_spec = ""

        instantiate_kwargs = {}
        if "seed" in cfg.env.env:
            instantiate_kwargs["seed"] = seed
        if "rank" in cfg.env.env:
            instantiate_kwargs["rank"] = rank + vector_env_idx
        env = hydra.utils.instantiate(cfg.env.env, **instantiate_kwargs)
        if "mujoco" in env_spec:
            env.frame_skip = 0

        # action repeat
        if (
            cfg.env.action_repeat > 1
            and "atari" not in env_spec
            and (not _IS_DMC_AVAILABLE or not isinstance(env, DMCWrapper))
            and (not (_IS_DIAMBRA_ARENA_AVAILABLE and _IS_DIAMBRA_AVAILABLE) or not isinstance(env, DiambraWrapper))
        ):
            env = ActionRepeat(env, cfg.env.action_repeat)

        if "mask_velocities" in cfg.env and cfg.env.mask_velocities:
            env = MaskVelocityWrapper(env)

        # Create observation dict
        if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) < 2:
            if cfg.cnn_keys.encoder is not None and len(cfg.cnn_keys.encoder) > 0:
                if len(cfg.cnn_keys.encoder) > 1:
                    warnings.warn(
                        "Multiple cnn keys have been specified and only one pixel observation "
                        f"is allowed in {cfg.env.id}, "
                        f"only the first one is kept: {cfg.cnn_keys.encoder[0]}"
                    )
                env = gym.wrappers.PixelObservationWrapper(
                    env, pixels_only=len(cfg.mlp_keys.encoder) == 0, pixel_keys=(cfg.cnn_keys.encoder[0],)
                )
            else:
                if cfg.mlp_keys.encoder is not None and len(cfg.mlp_keys.encoder) > 0:
                    if len(cfg.mlp_keys.encoder) > 1:
                        warnings.warn(
                            "Multiple mlp keys have been specified and only one pixel observation "
                            f"is allowed in {cfg.env.id}, "
                            f"only the first one is kept: {cfg.mlp_keys.encoder[0]}"
                        )
                    mlp_key = cfg.mlp_keys.encoder[0]
                else:
                    mlp_key = "state"
                    cfg.mlp_keys.encoder = [mlp_key]
                env = gym.wrappers.TransformObservation(env, lambda obs: {mlp_key: obs})
                env.observation_space = gym.spaces.Dict({mlp_key: env.observation_space})
        elif isinstance(env.observation_space, gym.spaces.Box) and 2 <= len(env.observation_space.shape) <= 3:
            if cfg.cnn_keys.encoder is not None and len(cfg.cnn_keys.encoder) > 1:
                warnings.warn(
                    "Multiple cnn keys have been specified and only one pixel observation "
                    f"is allowed in {cfg.env.id}, "
                    f"only the first one is kept: {cfg.cnn_keys.encoder[0]}"
                )
                cnn_key = cfg.cnn_keys.encoder[0]
            else:
                cnn_key = "rgb"
                cfg.cnn_keys.encoder = [cnn_key]
            env = gym.wrappers.TransformObservation(env, lambda obs: {cnn_key: obs})
            env.observation_space = gym.spaces.Dict({cnn_key: env.observation_space})

        env_cnn_keys = set(
            [k for k in env.observation_space.spaces.keys() if len(env.observation_space[k].shape) in {2, 3}]
        )
        if cfg.cnn_keys.encoder is None:
            user_cnn_keys = set()
        else:
            user_cnn_keys = set(cfg.cnn_keys.encoder)
        cnn_keys = env_cnn_keys.intersection(user_cnn_keys)

        def transform_obs(obs: Dict[str, Any]):
            for k in cnn_keys:
                shape = obs[k].shape
                is_3d = len(shape) == 3
                is_grayscale = not is_3d or shape[0] == 1 or shape[-1] == 1
                channel_first = not is_3d or shape[0] in (1, 3)

                # to 3D image
                if not is_3d:
                    obs.update({k: np.expand_dims(obs[k], axis=0)})

                # channel last (opencv needs it)
                if channel_first:
                    obs.update({k: obs[k].transpose(1, 2, 0)})

                # resize
                if obs[k].shape[:-1] != (cfg.env.screen_size, cfg.env.screen_size):
                    obs.update(
                        {
                            k: cv2.resize(
                                obs[k], (cfg.env.screen_size, cfg.env.screen_size), interpolation=cv2.INTER_AREA
                            )
                        }
                    )

                # to grayscale
                if cfg.env.grayscale and not is_grayscale:
                    obs.update({k: cv2.cvtColor(obs[k], cv2.COLOR_RGB2GRAY)})

                # back to 3D
                if len(obs[k].shape) == 2:
                    obs.update({k: np.expand_dims(obs[k], axis=-1)})
                    if not cfg.env.grayscale:
                        obs.update({k: np.repeat(obs[k], 3, axis=-1)})

                # channel first (PyTorch default)
                obs.update({k: obs[k].transpose(2, 0, 1)})

            return obs

        env = gym.wrappers.TransformObservation(env, transform_obs)
        for k in cnn_keys:
            env.observation_space[k] = gym.spaces.Box(
                0, 255, (1 if cfg.env.grayscale else 3, cfg.env.screen_size, cfg.env.screen_size), np.uint8
            )

        if cnn_keys is not None and len(cnn_keys) > 0 and cfg.env.frame_stack > 1:
            if cfg.env.frame_stack_dilation <= 0:
                raise ValueError(
                    f"The frame stack dilation argument must be greater than zero, got: {cfg.env.frame_stack_dilation}"
                )
            env = FrameStack(env, cfg.env.frame_stack, cnn_keys, cfg.env.frame_stack_dilation)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if cfg.env.max_episode_steps and cfg.env.max_episode_steps > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.env.max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if cfg.env.capture_video and rank == 0 and vector_env_idx == 0 and run_name is not None:
            env = gym.experimental.wrappers.RecordVideoV0(
                env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
            )
            env.metadata["render_fps"] = env.frames_per_sec
        return env

    return thunk


def get_dummy_env(id: str):
    if "continuous" in id:
        from sheeprl.envs.dummy import ContinuousDummyEnv

        env = ContinuousDummyEnv()
    elif "multidiscrete" in id:
        from sheeprl.envs.dummy import MultiDiscreteDummyEnv

        env = MultiDiscreteDummyEnv()
    elif "discrete" in id:
        from sheeprl.envs.dummy import DiscreteDummyEnv

        env = DiscreteDummyEnv()
    else:
        raise ValueError(f"Unrecognized dummy environment: {id}")
    return env
