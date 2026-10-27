"""The transforms of the images: resize and color (`Image`) and frame stacking (`FrameStack`) at `frame`, the pixels
with fewer bits and a uniform noise (`Dequantize`) at `batch`."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Sequence

import cv2
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from sheeprl.transforms.base import Transform


def _box(space: gym.Space, name: str) -> gym.spaces.Box:
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"{name} takes a `Box` space, got {space}")
    return space


class Image(Transform):
    """An image (2D, or 3D channel-first or channel-last) resized to `size` x `size` with OpenCV (`INTER_AREA`),
    converted to grayscale with `grayscale`, and returned channel-first. A one-channel image is repeated on three
    channels without `grayscale` when it is resized. The channels come first when `channel_first`, or, if it is not
    given, when the image has 1 or 3 of them on its first axis. The data keeps its dtype."""

    where = "frame"

    def __init__(
        self, size: int, grayscale: bool = False, channel_first: Optional[bool] = None, where: Optional[str] = None
    ) -> None:
        super().__init__(where)
        self.size = size
        self.grayscale = grayscale
        self.channel_first = channel_first

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "Image")
        if len(space.shape) not in (2, 3):
            raise ValueError(f"Image takes 2D or 3D images, got the shape {space.shape}")
        try:
            out = self.frame(np.zeros(space.shape, dtype=space.dtype))
        except cv2.error as e:
            raise ValueError(f"Image cannot transform images of shape {space.shape} and dtype {space.dtype}: {e}")
        # Resizing (by area) and the grayscale conversion keep the values within the bounds of the input
        return gym.spaces.Box(float(np.min(space.low)), float(np.max(space.high)), out.shape, out.dtype)

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        shape = x.shape
        is_3d = len(shape) == 3
        is_grayscale = not is_3d or shape[0] == 1 or shape[-1] == 1
        channel_first = not is_3d or (shape[0] in (1, 3) if self.channel_first is None else self.channel_first)
        # A 3D image, channel-last (as OpenCV needs it)
        if not is_3d:
            x = np.expand_dims(x, axis=0)
        if channel_first:
            x = np.transpose(x, (1, 2, 0))
        if x.shape[:-1] != (self.size, self.size):
            x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.grayscale and not is_grayscale:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        # OpenCV drops a single channel
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=-1)
            if not self.grayscale:
                x = np.repeat(x, 3, axis=-1)
        # Channel-first, as PyTorch
        return x.transpose(2, 0, 1)

    def extra_repr(self) -> str:
        return f"size={self.size}, grayscale={self.grayscale}, channel_first={self.channel_first}"


class FrameStack(Transform):
    """The last `n` frames of the episode concatenated along the channel axis (`(n * C, H, W)`), oldest first: every
    `dilation`-th of the last `n * dilation` frames. The first frame of an episode fills all of them. When one of the
    `refill_on` keys of the `info` of a step is true and the episode goes on (e.g. a new round of a DIAMBRA game), the
    current frame fills them again."""

    where = "frame"

    def __init__(self, n: int, dilation: int = 1, refill_on: Sequence[str] = (), where: Optional[str] = None) -> None:
        super().__init__(where)
        if n <= 0:
            raise ValueError(f"FrameStack stacks at least one frame, got n={n}")
        if dilation <= 0:
            raise ValueError(f"The dilation of FrameStack must be greater than zero, got {dilation}")
        self.n = n
        self.dilation = dilation
        self.refill_on = tuple(refill_on)
        self.needs_info = len(self.refill_on) > 0
        self._frames: deque = deque(maxlen=n * dilation)

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "FrameStack")
        if len(space.shape) < 1:
            raise ValueError("FrameStack stacks arrays of at least one dimension along their first axis")
        return gym.spaces.Box(
            np.concatenate([space.low] * self.n, axis=0),
            np.concatenate([space.high] * self.n, axis=0),
            (self.n * space.shape[0], *space.shape[1:]),
            space.dtype.type,
        )

    def reset(self) -> None:
        self._frames.clear()

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        refill = (
            info is not None
            and any(info.get(k, False) for k in self.refill_on)
            and not (info.get("terminated", False) or info.get("truncated", False))
        )
        if len(self._frames) == 0 or refill:
            self._frames.extend([x] * (self.n * self.dilation))
        else:
            self._frames.append(x)
        return np.concatenate(list(self._frames)[self.dilation - 1 :: self.dilation], axis=0)

    def extra_repr(self) -> str:
        return f"n={self.n}, dilation={self.dilation}, refill_on={self.refill_on}"


class Dequantize(Transform):
    """8-bit pixels reduced to `bits` bits, scaled to [0, 1] with a uniform noise of one level of the reduced bits,
    then shifted to [-0.5, 0.5] (https://arxiv.org/abs/1807.03039), as float: e.g. the targets of the reconstructions
    of the images of SAC-AE. The noise is drawn at every call."""

    where = "batch"

    def __init__(self, bits: int = 5, where: Optional[str] = None, phases: Optional[Sequence[str]] = None) -> None:
        super().__init__(where, phases)
        if not 1 <= bits <= 8:
            raise ValueError(f"Dequantize keeps from 1 to 8 bits of the pixels, got {bits}")
        self.bits = bits

    def space(self, space: gym.Space) -> gym.Space:
        space = _box(space, "Dequantize")
        return gym.spaces.Box(-0.5, 0.5, space.shape, np.float32)

    def batch(self, x: Tensor) -> Tensor:
        bins = 2**self.bits
        if self.bits < 8:
            x = torch.floor(x / 2 ** (8 - self.bits))
        x = x / bins
        x = x + torch.rand_like(x) / bins
        return x - 0.5

    def extra_repr(self) -> str:
        return f"bits={self.bits}, where={self.where}"
