"""Copies of the code the transforms replace, as it was on `041adff3`: the oracles the transforms must reproduce bit for
bit (the originals are removed by the refactor)."""

from collections import deque
from typing import Any, Dict, NamedTuple, Sequence, Tuple

import cv2
import gymnasium as gym
import numpy as np


def transform_obs(obs: Dict[str, Any], cnn_keys: Sequence[str], screen_size: int, grayscale: bool):
    """The `transform_obs` closure of `make_env` (`sheeprl/utils/env.py`)."""
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
        if current_obs.shape[:-1] != (screen_size, screen_size):
            current_obs = cv2.resize(current_obs, (screen_size, screen_size), interpolation=cv2.INTER_AREA)

        # to grayscale
        if grayscale and not is_grayscale:
            current_obs = cv2.cvtColor(current_obs, cv2.COLOR_RGB2GRAY)

        # back to 3D
        if len(current_obs.shape) == 2:
            current_obs = np.expand_dims(current_obs, axis=-1)
            if not grayscale:
                current_obs = np.repeat(current_obs, 3, axis=-1)

        # channel first (PyTorch default)
        obs[k] = current_obs.transpose(2, 0, 1)

    return obs


class FrameStackWrapper(gym.Wrapper):
    """The `FrameStack` wrapper (`sheeprl/envs/wrappers.py`)."""

    def __init__(self, env, num_stack: int, cnn_keys: Sequence[str], dilation: int = 1) -> None:
        super().__init__(env)
        self._num_stack = num_stack
        self._cnn_keys = [k for k, v in env.observation_space.spaces.items() if cnn_keys and len(v.shape) == 3]
        self._dilation = dilation
        self._frames = {k: deque(maxlen=num_stack * dilation) for k in self._cnn_keys}

    def _get_obs(self, key):
        frames_subset = list(self._frames[key])[self._dilation - 1 :: self._dilation]
        assert len(frames_subset) == self._num_stack
        return np.stack(list(frames_subset), axis=0)

    def step(self, action):
        obs, reward, done, truncated, infos = self.env.step(action)
        for k in self._cnn_keys:
            self._frames[k].append(obs[k])
            if (
                "env_domain" in infos
                and infos["env_domain"] == "DIAMBRA"
                and len(set(["round_done", "stage_done", "game_done"]).intersection(infos.keys())) == 3
                and (infos["round_done"] or infos["stage_done"] or infos["game_done"])
                and not (done or truncated)
            ):
                for _ in range(self._num_stack * self._dilation - 1):
                    self._frames[k].append(obs[k])
            obs[k] = self._get_obs(k)
        return obs, reward, done, truncated, infos

    def reset(self, *, seed=None, options=None, **kwargs):
        obs, infos = self.env.reset(seed=seed, **kwargs)
        [self._frames[k].clear() for k in self._cnn_keys]
        for k in self._cnn_keys:
            [self._frames[k].append(obs[k]) for _ in range(self._num_stack * self._dilation)]
            obs[k] = self._get_obs(k)
        return obs, infos


class SampleMoments(NamedTuple):
    mean: np.ndarray
    var: np.ndarray
    count: float

    def merge(self, other):
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / count
        m2 = self.var * self.count + other.var * other.count + delta**2 * self.count * other.count / count
        return SampleMoments(mean, m2 / count, count)


class RunningMeanStd:
    """`RunningMeanStd` (`sheeprl/envs/normalization.py`), without the sync."""

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.moments = SampleMoments(np.zeros(shape), np.ones(shape), epsilon)

    @property
    def mean(self):
        return self.moments.mean

    @property
    def var(self):
        return self.moments.var

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.moments = self.moments.merge(SampleMoments(x.mean(axis=0), x.var(axis=0), x.shape[0]))


class ObsNormalizer:
    """`ObsNormalizer` (`sheeprl/envs/normalization.py`), without the sync."""

    def __init__(self, observation_space, keys: Sequence[str], clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.stats = {k: RunningMeanStd(observation_space[k].shape) for k in keys}
        self.clip = clip
        self.epsilon = epsilon

    def update(self, obs):
        for k, stats in self.stats.items():
            stats.update(obs[k])

    def normalize(self, obs):
        obs = dict(obs)
        for k, stats in self.stats.items():
            normalized = (obs[k] - stats.mean) / np.sqrt(stats.var + self.epsilon)
            obs[k] = np.clip(normalized, -self.clip, self.clip).astype(np.float32)
        return obs


class RewardNormalizer:
    """`RewardNormalizer` (`sheeprl/envs/normalization.py`), without the sync."""

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.stats = RunningMeanStd()
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        self.stats.update(self.returns)
        normalized = np.clip(rewards / np.sqrt(self.stats.var + self.epsilon), -self.clip, self.clip)
        self.returns[dones] = 0.0
        return normalized.astype(np.result_type(rewards.dtype, np.float32))
