import os
import warnings
from typing import Any, Callable, Dict, Optional

import cv2
import gymnasium as gym
import hydra
import numpy as np

from sheeprl.envs.wrappers import (
    ActionRepeat,
    FrameStack,
    GrayscaleRenderWrapper,
    MaskVelocityWrapper,
    RewardAsObservationWrapper,
)
from sheeprl.utils.imports import _IS_DIAMBRA_ARENA_AVAILABLE, _IS_DIAMBRA_AVAILABLE, _IS_DMC_AVAILABLE

if _IS_DIAMBRA_ARENA_AVAILABLE and _IS_DIAMBRA_AVAILABLE:
    from sheeprl.envs.diambra import DiambraWrapper
if _IS_DMC_AVAILABLE:
    pass


def make_env(
    cfg: Dict[str, Any],
    seed: int,
    rank: int,
    run_name: Optional[str] = None,
    prefix: str = "",
    vector_env_idx: int = 0,
) -> Callable[[], gym.Env]:
    """
    Create the callable function to create environment and
    force the environment to return an observation space of type
    gymnasium.spaces.Dict.

    Args:
        cfg (Dict[str, Any]): the configs of the environment to initialize.
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

    def thunk() -> gym.Env:
        try:
            env_spec = gym.spec(cfg.env.id).entry_point
        except Exception:
            env_spec = ""

        if "diambra" in cfg.env.wrapper._target_ and not cfg.env.sync_env:
            if cfg.env.wrapper.diambra_settings.pop("splash_screen", True):
                warnings.warn(
                    "You must set the `splash_screen` setting to `False` when using the `AsyncVectorEnv` "
                    "in `DIAMBRA` environments. The specified `splash_screen` setting is ignored and set "
                    "to `False`."
                )
            cfg.env.wrapper.diambra_settings.splash_screen = False

        instantiate_kwargs = {}
        if "seed" in cfg.env.wrapper:
            instantiate_kwargs["seed"] = seed
        if "rank" in cfg.env.wrapper:
            instantiate_kwargs["rank"] = rank + vector_env_idx
        env = hydra.utils.instantiate(cfg.env.wrapper, **instantiate_kwargs)

        # action repeat
        if (
            cfg.env.action_repeat > 1
            and "atari" not in env_spec
            and (not (_IS_DIAMBRA_ARENA_AVAILABLE and _IS_DIAMBRA_AVAILABLE) or not isinstance(env, DiambraWrapper))
        ):
            env = ActionRepeat(env, cfg.env.action_repeat)

        if "mask_velocities" in cfg.env and cfg.env.mask_velocities:
            env = MaskVelocityWrapper(env)

        if not (
            isinstance(cfg.algo.mlp_keys.encoder, list)
            and isinstance(cfg.algo.cnn_keys.encoder, list)
            and len(cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder) > 0
        ):
            raise ValueError(
                "`algo.cnn_keys.encoder` and `algo.mlp_keys.encoder` must be lists of strings, got: "
                f"cnn encoder keys `{cfg.algo.cnn_keys.encoder}` of type `{type(cfg.algo.cnn_keys.encoder)}` "
                f"and mlp encoder keys `{cfg.algo.mlp_keys.encoder}` of type `{type(cfg.algo.mlp_keys.encoder)}`. "
                "Both must be non-empty lists."
            )

        # Create observation dict
        encoder_cnn_keys_length = len(cfg.algo.cnn_keys.encoder)
        encoder_mlp_keys_length = len(cfg.algo.mlp_keys.encoder)
        if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) < 2:
            # Vector only observation
            if encoder_cnn_keys_length > 0:
                if encoder_cnn_keys_length > 1:
                    warnings.warn(
                        "Multiple cnn keys have been specified and only one pixel observation "
                        f"is allowed in {cfg.env.id}, "
                        f"only the first one is kept: {cfg.algo.cnn_keys.encoder[0]}"
                    )
                if encoder_mlp_keys_length > 0:
                    gym.wrappers.pixel_observation.STATE_KEY = cfg.algo.mlp_keys.encoder[0]
                env = gym.wrappers.PixelObservationWrapper(
                    env, pixels_only=encoder_mlp_keys_length == 0, pixel_keys=(cfg.algo.cnn_keys.encoder[0],)
                )
            else:
                if encoder_mlp_keys_length > 1:
                    warnings.warn(
                        "Multiple mlp keys have been specified and only one pixel observation "
                        f"is allowed in {cfg.env.id}, "
                        f"only the first one is kept: {cfg.algo.mlp_keys.encoder[0]}"
                    )
                mlp_key = cfg.algo.mlp_keys.encoder[0]
                env = gym.wrappers.TransformObservation(env, lambda obs: {mlp_key: obs})
                env.observation_space = gym.spaces.Dict({mlp_key: env.observation_space})
        elif isinstance(env.observation_space, gym.spaces.Box) and 2 <= len(env.observation_space.shape) <= 3:
            # Pixel only observation
            if encoder_cnn_keys_length > 1:
                warnings.warn(
                    "Multiple cnn keys have been specified and only one pixel observation "
                    f"is allowed in {cfg.env.id}, "
                    f"only the first one is kept: {cfg.algo.cnn_keys.encoder[0]}"
                )
            elif encoder_cnn_keys_length == 0:
                raise ValueError(
                    "You have selected a pixel observation but no cnn key has been specified. "
                    "Please set at least one cnn key in the config file: `algo.cnn_keys.encoder=[your_cnn_key]`"
                )
            cnn_key = cfg.algo.cnn_keys.encoder[0]
            env = gym.wrappers.TransformObservation(env, lambda obs: {cnn_key: obs})
            env.observation_space = gym.spaces.Dict({cnn_key: env.observation_space})

        if (
            len(
                set(k for k in env.observation_space.keys()).intersection(
                    set(cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder)
                )
            )
            == 0
        ):
            raise ValueError(
                f"The user specified keys `{cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder}` "
                "are not a subset of the "
                f"environment `{env.observation_space.keys()}` observation keys. Please check your config file."
            )

        env_cnn_keys = set(
            [k for k in env.observation_space.spaces.keys() if len(env.observation_space[k].shape) in {2, 3}]
        )
        cnn_keys = env_cnn_keys.intersection(set(cfg.algo.cnn_keys.encoder))

        def transform_obs(obs: Dict[str, Any]):
            for k in cnn_keys:
                current_obs = obs[k]
                shape = current_obs.shape
                is_3d = len(shape) == 3
                is_grayscale = not is_3d or shape[0] == 1 or shape[-1] == 1
                channel_first = not is_3d or shape[0] in (1, 3)

                # to 3D image
                if not is_3d:
                    current_obs = np.expand_dims(current_obs, axis=0)

                # channel last (opencv needs it)
                if channel_first:
                    current_obs = np.transpose(current_obs, (1, 2, 0))

                # resize
                if current_obs.shape[:-1] != (cfg.env.screen_size, cfg.env.screen_size):
                    current_obs = cv2.resize(
                        current_obs, (cfg.env.screen_size, cfg.env.screen_size), interpolation=cv2.INTER_AREA
                    )

                # to grayscale
                if cfg.env.grayscale and not is_grayscale:
                    current_obs = cv2.cvtColor(current_obs, cv2.COLOR_RGB2GRAY)

                # back to 3D
                if len(current_obs.shape) == 2:
                    current_obs = np.expand_dims(current_obs, axis=-1)
                    if not cfg.env.grayscale:
                        current_obs = np.repeat(current_obs, 3, axis=-1)

                # channel first (PyTorch default)
                obs[k] = current_obs.transpose(2, 0, 1)

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

        if cfg.env.reward_as_observation:
            env = RewardAsObservationWrapper(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if cfg.env.max_episode_steps and cfg.env.max_episode_steps > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.env.max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if cfg.env.capture_video and rank == 0 and vector_env_idx == 0 and run_name is not None:
            if cfg.env.grayscale:
                env = GrayscaleRenderWrapper(env)
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
