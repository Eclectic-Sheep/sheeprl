"""The transforms of the shapes, the dtypes and the values: `Flatten`, `ExpandDims`, `Cast`, `OneHot` (`frame`),
`Standardize` (`envs`), `Scale` and `Symlog` (`batch`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import gymnasium as gym
import numpy as np
from torch import Tensor

from sheeprl.transforms.base import Transform
from sheeprl.transforms.stats import RunningMeanStd
from sheeprl.utils.utils import symexp, symlog

if TYPE_CHECKING:
    from lightning import Fabric


def _box(space: gym.Space, name: str) -> gym.spaces.Box:
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"{name} takes a `Box` space, got {space}")
    return space


class Flatten(Transform):
    """Any shape to a vector."""

    where = "frame"
    supports = ("frame", "batch")

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "Flatten")
        self.sample_ndim = len(space.shape)
        return gym.spaces.Box(space.low.reshape(-1), space.high.reshape(-1), (int(np.prod(space.shape)),), space.dtype)

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return x.reshape(-1)

    def batch(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[: x.dim() - self.sample_ndim], -1)


class ExpandDims(Transform):
    """A new axis of size 1 at `axis` (of one observation), e.g. a 2D grid as one channel."""

    where = "frame"

    def __init__(self, axis: int = 0, where: Optional[str] = None) -> None:
        super().__init__(where)
        self.axis = axis

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "ExpandDims")
        low, high = np.expand_dims(space.low, self.axis), np.expand_dims(space.high, self.axis)
        return gym.spaces.Box(low, high, low.shape, space.dtype)

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return np.expand_dims(x, self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"


class Cast(Transform):
    """The data cast to `dtype` (a NumPy dtype name, e.g. `float32`)."""

    where = "frame"

    def __init__(self, dtype: str, where: Optional[str] = None) -> None:
        super().__init__(where)
        self.dtype = np.dtype(dtype)

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "Cast")
        return gym.spaces.Box(space.low.astype(self.dtype), space.high.astype(self.dtype), space.shape, self.dtype)

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return np.asarray(x).astype(self.dtype)

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}"


class OneHot(Transform):
    """A `Discrete(n)` observation as a one-hot float32 vector of `n` values."""

    where = "frame"

    def space(self, space: gym.Space) -> gym.Space:
        if not isinstance(space, gym.spaces.Discrete):
            raise TypeError(f"OneHot takes a `Discrete` space, got {space}")
        self.n, self.start = int(space.n), int(space.start)
        return gym.spaces.Box(0, 1, (self.n,), np.float32)

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        one_hot = np.zeros(self.n, dtype=np.float32)
        one_hot[int(x) - self.start] = 1
        return one_hot


class Standardize(Transform):
    """The observations standardized with their running mean and variance over the environments of all the
    processes, then clipped to [-`clip`, `clip`], as float32. The statistics are updated with the observations of every
    step (not with the last observations of the ended episodes, nor in the test), synced across the processes at the
    end of every rollout and saved in the checkpoints."""

    where = "envs"
    learns_from_data = True

    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8, where: Optional[str] = None) -> None:
        super().__init__(where)
        self.clip = clip
        self.epsilon = epsilon

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "Standardize")
        self.stats = RunningMeanStd(space.shape)
        return gym.spaces.Box(-self.clip, self.clip, space.shape, np.float32)

    def envs(self, x: np.ndarray, update: bool = True, dones: Optional[np.ndarray] = None) -> np.ndarray:
        if update:
            self.stats.update(x)
        mean, var = self.stats.mean.numpy(), self.stats.var.numpy()
        return np.clip((x - mean) / np.sqrt(var + self.epsilon), -self.clip, self.clip).astype(np.float32)

    def sync(self, fabric: Fabric) -> None:
        self.stats.sync(fabric)

    def extra_repr(self) -> str:
        return f"clip={self.clip}, epsilon={self.epsilon}"


class Scale(Transform):
    """`x / divisor - shift` as float32, in this order (e.g. the pixels to [-0.5, 0.5])."""

    where = "batch"
    supports = ("frame", "envs", "batch")

    def __init__(
        self, divisor: float = 255, shift: float = 0.5, where: Optional[str] = None, phases: Optional[Any] = None
    ) -> None:
        super().__init__(where, phases)
        self.divisor = divisor
        self.shift = shift

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "Scale")
        low = space.low.astype(np.float32) / self.divisor - self.shift
        high = space.high.astype(np.float32) / self.divisor - self.shift
        return gym.spaces.Box(low, high, space.shape, np.float32)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32) / np.float32(self.divisor)
        return x - np.float32(self.shift) if self.shift else x

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return self._scale(x)

    def envs(self, x: np.ndarray, update: bool = True, dones: Optional[np.ndarray] = None) -> np.ndarray:
        return self._scale(x)

    def batch(self, x: Tensor) -> Tensor:
        x = x.float() / self.divisor
        return x - self.shift if self.shift else x

    def inverse(self, x: Tensor) -> Tensor:
        return (x + self.shift) * self.divisor

    def extra_repr(self) -> str:
        return f"divisor={self.divisor}, shift={self.shift}, where={self.where}"


class Symlog(Transform):
    """`sign(x) * log(1 + |x|)` as float32: values of any magnitude in a range a network learns from."""

    where = "batch"

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "Symlog")
        low = np.sign(space.low) * np.log1p(np.abs(space.low))
        high = np.sign(space.high) * np.log1p(np.abs(space.high))
        return gym.spaces.Box(low.astype(np.float32), high.astype(np.float32), space.shape, np.float32)

    def batch(self, x: Tensor) -> Tensor:
        return symlog(x.float())

    def inverse(self, x: Tensor) -> Tensor:
        return symexp(x)
