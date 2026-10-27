import os
import shutil
import tempfile
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Type

import cv2
import gymnasium as gym
import hydra
import numpy as np

from sheeprl.envs.wrappers import (
    ActionRepeat,
    ActionsAsObservationWrapper,
    FrameStack,
    GrayscaleRenderWrapper,
    MaskVelocityWrapper,
    NormalizeAction,
    RewardAsObservationWrapper,
)
from sheeprl.utils import fs
from sheeprl.utils.imports import _IS_ATARI_AVAILABLE, _IS_DIAMBRA_ARENA_AVAILABLE, _IS_DIAMBRA_AVAILABLE
from sheeprl.utils.utils import dotdict

if _IS_ATARI_AVAILABLE:
    import ale_py

    # Since gymnasium 1.0 the Atari environments are registered by `ale-py`
    gym.register_envs(ale_py)
if _IS_DIAMBRA_ARENA_AVAILABLE and _IS_DIAMBRA_AVAILABLE:
    from sheeprl.envs.diambra import DiambraWrapper


def get_vector_env_cls(sync_env: bool) -> Callable[[Sequence[Callable[[], gym.Env]]], gym.vector.VectorEnv]:
    """Return the vectorized environment class (`SyncVectorEnv` or `AsyncVectorEnv`) configured
    with the same-step autoreset mode: when an episode ends, the sub-environment is reset in the same step,
    the returned observation is the first one of the new episode and the last observation and info
    of the finished episode are stored in `info["final_obs"]` and `info["final_info"]` respectively.

    Args:
        sync_env (bool): whether to use the `SyncVectorEnv` (True) or the `AsyncVectorEnv` (False).

    Returns:
        The callable that creates the vectorized environment given the list of environment thunks.
    """
    vector_env_cls = gym.vector.SyncVectorEnv if sync_env else gym.vector.AsyncVectorEnv
    return partial(vector_env_cls, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def get_episode_stats(info: Dict[str, Any]) -> List[Tuple[int, float, int]]:
    """Return the statistics of the episodes ended in the last step of a vectorized environment, as recorded
    by the `gymnasium.wrappers.RecordEpisodeStatistics` wrapper in `info["final_info"]`.

    Args:
        info (Dict[str, Any]): the info returned by the `step` method of a vectorized environment
            created with `get_vector_env_cls`.

    Returns:
        The list of (environment index, episode return, episode length) of the ended episodes.
    """
    final_info = info.get("final_info", {})
    if "episode" not in final_info:
        return []
    episode = final_info["episode"]
    return [(int(i), float(episode["r"][i]), int(episode["l"][i])) for i in np.flatnonzero(final_info["_episode"])]


def _is_wrapped_by(env: gym.Env, wrapper_cls: Type[gym.Wrapper]) -> bool:
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_cls):
            return True
        env = env.env
    return False


class UploadedRecordVideo(gym.wrappers.RecordVideo):
    """`RecordVideo` for a run logged on a remote filesystem (`log_root`): the videos are written in a local temporary
    directory, then copied to `remote_folder` and removed, at the end of every recording."""

    def __init__(self, env: gym.Env, remote_folder: str, **kwargs: Any) -> None:
        super().__init__(env, tempfile.mkdtemp(prefix="sheeprl_videos_"), disable_logger=True, **kwargs)
        self.remote_folder = remote_folder

    def stop_recording(self) -> None:
        super().stop_recording()
        for name in os.listdir(self.video_folder):
            local_path = os.path.join(self.video_folder, name)
            fs.upload(local_path, fs.join(self.remote_folder, name))
            os.remove(local_path)

    def close(self) -> None:
        super().close()
        shutil.rmtree(self.video_folder, ignore_errors=True)


def make_env(
    cfg: dotdict,
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
        cfg (dotdict): the configs of the environment to initialize.
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
        env = hydra.utils.instantiate(cfg.env.wrapper, **instantiate_kwargs, _convert_="all")

        # The bounded continuous actions exposed as [-1, 1], the range of the actor (the Dreamers). Read with a
        # default: the configurations saved before the option don't have it, and their checkpoints play as trained
        if cfg.algo.get("normalize_actions", False) and isinstance(env.action_space, gym.spaces.Box):
            env = NormalizeAction(env)

        # action repeat: the Atari environments repeat the action through the `frame_skip`
        # argument of the `AtariPreprocessing` wrapper
        if (
            cfg.env.action_repeat > 1
            and not _is_wrapped_by(env, gym.wrappers.AtariPreprocessing)
            and (not (_IS_DIAMBRA_ARENA_AVAILABLE and _IS_DIAMBRA_AVAILABLE) or not isinstance(env, DiambraWrapper))
        ):
            env = ActionRepeat(env, cfg.env.action_repeat)

        if "mask_velocities" in cfg.env and cfg.env.mask_velocities:
            env = MaskVelocityWrapper(env)

        if not (
            isinstance(legacy_keys(cfg).vector_keys, list)
            and isinstance(legacy_keys(cfg).image_keys, list)
            and len(legacy_keys(cfg).image_keys + legacy_keys(cfg).vector_keys) > 0
        ):
            raise ValueError(
                "`algo.inputs.image.keys` and `algo.inputs.vector.keys` must be lists of strings, got: "
                f"cnn encoder keys `{legacy_keys(cfg).image_keys}` of type `{type(legacy_keys(cfg).image_keys)}` "
                f"and mlp encoder keys `{legacy_keys(cfg).vector_keys}` "
                f"of type `{type(legacy_keys(cfg).vector_keys)}`. "
                "Both must be non-empty lists."
            )

        # Create observation dict
        encoder_cnn_keys_length = len(legacy_keys(cfg).image_keys)
        encoder_mlp_keys_length = len(legacy_keys(cfg).vector_keys)
        if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) < 2:
            # Vector only observation
            if encoder_cnn_keys_length > 0:
                if encoder_cnn_keys_length > 1:
                    warnings.warn(
                        "Multiple cnn keys have been specified and only one pixel observation "
                        f"is allowed in {cfg.env.id}, "
                        f"only the first one is kept: {legacy_keys(cfg).image_keys[0]}"
                    )
                cnn_key = legacy_keys(cfg).image_keys[0]
                if encoder_mlp_keys_length > 0:
                    env = gym.wrappers.AddRenderObservation(
                        env, render_only=False, render_key=cnn_key, obs_key=legacy_keys(cfg).vector_keys[0]
                    )
                else:
                    env = gym.wrappers.AddRenderObservation(env, render_only=True)
                    env = gym.wrappers.TransformObservation(
                        env, lambda obs: {cnn_key: obs}, gym.spaces.Dict({cnn_key: env.observation_space})
                    )
            else:
                if encoder_mlp_keys_length > 1:
                    warnings.warn(
                        "Multiple mlp keys have been specified and only one pixel observation "
                        f"is allowed in {cfg.env.id}, "
                        f"only the first one is kept: {legacy_keys(cfg).vector_keys[0]}"
                    )
                mlp_key = legacy_keys(cfg).vector_keys[0]
                env = gym.wrappers.TransformObservation(
                    env, lambda obs: {mlp_key: obs}, gym.spaces.Dict({mlp_key: env.observation_space})
                )
        elif isinstance(env.observation_space, gym.spaces.Box) and 2 <= len(env.observation_space.shape) <= 3:
            # Pixel only observation
            if encoder_cnn_keys_length > 1:
                warnings.warn(
                    "Multiple cnn keys have been specified and only one pixel observation "
                    f"is allowed in {cfg.env.id}, "
                    f"only the first one is kept: {legacy_keys(cfg).image_keys[0]}"
                )
            elif encoder_cnn_keys_length == 0:
                raise ValueError(
                    "You have selected a pixel observation but no cnn key has been specified. "
                    "Please set at least one image key in the config file: `algo.inputs.image.keys=[your_image_key]`"
                )
            cnn_key = legacy_keys(cfg).image_keys[0]
            env = gym.wrappers.TransformObservation(
                env, lambda obs: {cnn_key: obs}, gym.spaces.Dict({cnn_key: env.observation_space})
            )

        if (
            len(
                set(k for k in env.observation_space.keys()).intersection(
                    set(legacy_keys(cfg).vector_keys + legacy_keys(cfg).image_keys)
                )
            )
            == 0
        ):
            raise ValueError(
                f"The user specified keys `{legacy_keys(cfg).vector_keys + legacy_keys(cfg).image_keys}` "
                "are not a subset of the "
                f"environment `{env.observation_space.keys()}` observation keys. Please check your config file."
            )

        env_cnn_keys = set(
            [k for k in env.observation_space.spaces.keys() if len(env.observation_space[k].shape) in {2, 3}]
        )
        cnn_keys = env_cnn_keys.intersection(set(legacy_keys(cfg).image_keys))

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

        transformed_cnn_space = gym.spaces.Box(
            0, 255, (1 if cfg.env.grayscale else 3, cfg.env.screen_size, cfg.env.screen_size), np.uint8
        )
        env = gym.wrappers.TransformObservation(
            env,
            transform_obs,
            gym.spaces.Dict(
                {k: transformed_cnn_space if k in cnn_keys else v for k, v in env.observation_space.spaces.items()}
            ),
        )

        frame_stack, frame_stack_dilation = legacy_frame_stack(cfg)
        if cnn_keys is not None and len(cnn_keys) > 0 and frame_stack > 1:
            if frame_stack_dilation <= 0:
                raise ValueError(
                    f"The frame stack dilation argument must be greater than zero, got: {frame_stack_dilation}"
                )
            env = FrameStack(env, frame_stack, cnn_keys, frame_stack_dilation)

        if cfg.env.actions_as_observation.num_stack > 0 and "diambra" not in cfg.env.wrapper._target_:
            env = ActionsAsObservationWrapper(env, **cfg.env.actions_as_observation)

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
            video_folder = fs.join(run_name, prefix + "_videos" if prefix else "videos")
            if fs.is_url(video_folder):
                env = UploadedRecordVideo(env, video_folder)
            else:
                env = gym.wrappers.RecordVideo(env, video_folder, disable_logger=True)
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


class LegacyKeys(NamedTuple):
    image_keys: List[str]
    vector_keys: List[str]
    decoder_image_keys: List[str]
    decoder_vector_keys: List[str]


def legacy_keys(cfg: dotdict) -> LegacyKeys:
    """Transitional, removed once every algorithm builds its inputs from `algo.inputs` (`sheeprl.models.io`): the keys
    of the inputs `image` and `vector` and of the outputs `image` and `vector`, for the code written for the former
    `algo.cnn_keys` and `algo.mlp_keys`."""

    def keys(section: Any, name: str) -> List[str]:
        return list(((section or {}).get(name) or {}).get("keys") or [])

    inputs, outputs = cfg.algo.get("inputs"), cfg.algo.get("outputs")
    return LegacyKeys(keys(inputs, "image"), keys(inputs, "vector"), keys(outputs, "image"), keys(outputs, "vector"))


def _legacy_transform(cfg: dotdict, input_name: str, cls_name: str) -> Optional[Dict[str, Any]]:
    from sheeprl import transforms

    cls = getattr(transforms, cls_name)
    for transform in ((cfg.algo.get("inputs") or {}).get(input_name) or {}).get("transforms") or []:
        if issubclass(hydra.utils.get_class(transform["_target_"]), cls):
            return transform
    return None


def legacy_frame_stack(cfg: dotdict) -> Tuple[int, int]:
    """Transitional, removed with the `FrameStack` wrapper: the frames and the dilation of the `FrameStack` transform of
    the input `image`, `(1, 1)` without one."""
    transform = _legacy_transform(cfg, "image", "FrameStack")
    return (1, 1) if transform is None else (transform["n"], transform.get("dilation", 1))


def legacy_standardize(cfg: dotdict) -> Optional[Tuple[float, float]]:
    """Transitional, removed with `ObsNormalizer`: the `clip` and `epsilon` of the `Standardize` transform of the input
    `vector`, `None` without one."""
    transform = _legacy_transform(cfg, "vector", "Standardize")
    return None if transform is None else (transform.get("clip", 10.0), transform.get("epsilon", 1e-8))


def actions_dim_of(action_space: gym.Space) -> Tuple[Tuple[int, ...], bool]:
    """The dimensions of the actions (one per discrete action) and whether they are continuous, as Python integers:
    `torch.compile` reads the NumPy ones (of `Discrete.n`) as tensors."""
    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        int(dim)
        for dim in (
            action_space.shape
            if is_continuous
            else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
        )
    )
    return actions_dim, is_continuous
