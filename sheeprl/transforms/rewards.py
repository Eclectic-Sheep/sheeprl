"""The transforms of the rewards (`algo.rewards.transforms`), run at `envs` in `GymEnvironment.step` before the rewards
are stored: `ScaleByReturnStd` and the elementwise `Tanh`, `Sign` and `Clip` (these also at `batch`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from sheeprl.transforms.base import Pipeline, Transform, instantiate_transforms
from sheeprl.transforms.stats import RunningMeanStd

if TYPE_CHECKING:
    from lightning import Fabric

    from sheeprl.utils.utils import dotdict

# The space of the rewards of one environment
REWARD_SPACE = gym.spaces.Box(-np.inf, np.inf, (), np.float64)


def _box(space: gym.Space, name: str) -> gym.spaces.Box:
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"{name} takes a `Box` space, got {space}")
    return space


class ScaleByReturnStd(Transform):
    """The rewards divided by the running standard deviation of the discounted return of every environment
    (`R = gamma * R + r`, from 0 again where an episode ends), then clipped to [-`clip`, `clip`], with their dtype if it
    is a floating-point one: they are not centered, their sign doesn't change. The statistics of the returns of the
    environments of all the processes are synced and saved; the returns themselves are episode state, not saved."""

    where = "envs"
    learns_from_data = True

    def __init__(self, gamma: float, clip: float = 10.0, epsilon: float = 1e-8, where: Optional[str] = None) -> None:
        super().__init__(where)
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.returns: Optional[np.ndarray] = None

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "ScaleByReturnStd")
        self.stats = RunningMeanStd(space.shape)
        return gym.spaces.Box(-self.clip, self.clip, space.shape, np.result_type(space.dtype, np.float32).type)

    def envs(self, x: np.ndarray, update: bool = True, dones: Optional[np.ndarray] = None) -> np.ndarray:
        if update:
            returns = self.returns
            if returns is None:
                returns = np.zeros(x.shape, dtype=np.float64)
            returns = returns * self.gamma + x
            self.stats.update(returns)
            # From 0 again where an episode ended (`update` has read the returns already)
            if dones is not None:
                returns[dones] = 0.0
            self.returns = returns
        normalized = np.clip(x / np.sqrt(self.stats.var.numpy() + self.epsilon), -self.clip, self.clip)
        return normalized.astype(np.result_type(x.dtype, np.float32))

    def sync(self, fabric: Fabric) -> None:
        self.stats.sync(fabric)

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, clip={self.clip}, epsilon={self.epsilon}"


class _Elementwise(Transform):
    """An elementwise function of the data, at `envs` or at `batch`, that keeps its dtype."""

    where = "envs"
    supports = ("envs", "batch")
    low, high = -np.inf, np.inf

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, type(self).__name__)
        low = np.maximum(self._numpy(space.low.astype(np.float64)), self.low)
        high = np.minimum(self._numpy(space.high.astype(np.float64)), self.high)
        return gym.spaces.Box(low.astype(space.dtype), high.astype(space.dtype), space.shape, space.dtype.type)

    def _numpy(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _torch(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def envs(self, x: np.ndarray, update: bool = True, dones: Optional[np.ndarray] = None) -> np.ndarray:
        return self._numpy(x)

    def batch(self, x: Tensor) -> Tensor:
        return self._torch(x)


class Tanh(_Elementwise):
    """`tanh(x)`: e.g. the rewards squashed to [-1, 1] (the former `env.clip_rewards`)."""

    low, high = -1.0, 1.0

    def _numpy(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _torch(self, x: Tensor) -> Tensor:
        return torch.tanh(x)


class Sign(_Elementwise):
    """`sign(x)`: -1, 0 or 1 (e.g. the rewards of the Atari games, as in DQN)."""

    low, high = -1.0, 1.0

    def _numpy(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)

    def _torch(self, x: Tensor) -> Tensor:
        return torch.sign(x)


class Clip(_Elementwise):
    """`x` clipped to [`low`, `high`]."""

    def __init__(self, low: float, high: float, where: Optional[str] = None, phases: Optional[Any] = None) -> None:
        super().__init__(where, phases)
        if low > high:
            raise ValueError(f"Clip needs `low <= high`, got {low} and {high}")
        self.low, self.high = low, high

    def _numpy(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.low, self.high)

    def _torch(self, x: Tensor) -> Tensor:
        return torch.clamp(x, self.low, self.high)

    def extra_repr(self) -> str:
        return f"low={self.low}, high={self.high}, where={self.where}"


def reward_pipeline(cfg: dotdict, off_policy: bool = False) -> Pipeline:
    """The transforms of the rewards (`algo.rewards.transforms`): `envs` transforms only, since the rewards have no
    space to size a network from and what the algorithms do to them on the device is part of their losses. A transform
    that learns from the data (e.g. `ScaleByReturnStd`) is an error for an off-policy algorithm, whose replay buffer
    would keep rewards transformed with statistics that change during the training."""
    transforms = instantiate_transforms(((cfg.algo.get("rewards") or {}).get("transforms")) or [])
    not_envs = [type(t).__name__ for t in transforms if t.where != "envs"]
    if not_envs:
        raise ValueError(f"The transforms of the rewards run at `envs`, got {not_envs} at another placement")
    learning = [type(t).__name__ for t in transforms if t.learns_from_data]
    if off_policy and learning:
        raise ValueError(
            f"The transforms of the rewards {learning} learn statistics from the data: they are supported only by "
            "on-policy algorithms, as the replay buffer of an off-policy algorithm would keep rewards transformed with "
            "statistics that change during the training"
        )
    return Pipeline(transforms, REWARD_SPACE, "envs", "envs")
