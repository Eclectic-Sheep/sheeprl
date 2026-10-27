import os
import shutil
import tempfile
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Type

import gymnasium as gym
import hydra
import numpy as np

from sheeprl.envs.wrappers import (
    ActionRepeat,
    ActionsAsObservationWrapper,
    GrayscaleRenderWrapper,
    MaskVelocityWrapper,
    NormalizeAction,
    RewardAsObservationWrapper,
    TransformsWrapper,
)
from sheeprl.transforms.observations import frame_pipelines
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
    """The callable that creates one environment of `cfg.env`, whose observations are a dictionary of the keys the agent
    stores (`algo.inputs`, `algo.extra_keys`), each through the `frame` stage of its transforms: a bare observation is
    the key `state`, the render is the key `env.render_key` when set, and the keys nobody declares are dropped.

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

        # A bare observation is the key `state`; the render is added under `env.render_key` when set
        if not isinstance(env.observation_space, gym.spaces.Dict):
            env = gym.wrappers.TransformObservation(
                env, lambda obs: {"state": obs}, gym.spaces.Dict({"state": env.observation_space})
            )
        if cfg.env.get("render_key"):
            env = gym.wrappers.AddRenderObservation(env, render_only=False, render_key=cfg.env.render_key)

        if cfg.env.actions_as_observation.num_stack > 0 and "diambra" not in cfg.env.wrapper._target_:
            env = ActionsAsObservationWrapper(env, **cfg.env.actions_as_observation)

        if cfg.env.reward_as_observation:
            env = RewardAsObservationWrapper(env)

        # The `frame` stage of the keys the agent stores (`algo.inputs`, `algo.extra_keys`), the other keys dropped
        env = TransformsWrapper(env, frame_pipelines(cfg, env.observation_space))

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
