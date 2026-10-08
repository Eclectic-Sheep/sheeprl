"""Sample a `ReplayBuffer` in the tests as its `sample` did up to sheeprl 0.8.3: with a generator and an online queue of
the buffer, shared by all its samples."""

import weakref
from typing import Dict, Optional

import numpy as np
from torch import Tensor

from sheeprl.data.buffers import ReplayBuffer, get_tensor
from sheeprl.data.samplers import OnlineQueue, SequenceSampler, TransitionSampler

_STATE: "weakref.WeakKeyDictionary[ReplayBuffer, tuple]" = weakref.WeakKeyDictionary()


def draw(
    rb: ReplayBuffer,
    batch_size: int,
    sample_next_obs: bool = False,
    n_samples: int = 1,
    sequence_length: Optional[int] = None,
    online: bool = False,
    clone: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Single steps (`TransitionSampler`), or sequences of `sequence_length` steps (`SequenceSampler`), drawn with the
    generator of the buffer (from `seed` at its first sample) and its online queue."""
    rng, queue = _STATE.setdefault(rb, (np.random.default_rng(seed), OnlineQueue()))
    if sequence_length is None:
        sampler = TransitionSampler(sample_next_obs, online, rng=rng, queue=queue)
    else:
        sampler = SequenceSampler(sequence_length, sample_next_obs, online, rng=rng, queue=queue)
    return sampler.sample(rb, batch_size, n_samples, clone=clone)


def draw_tensors(
    rb: ReplayBuffer, batch_size: int, device="cpu", dtype=None, from_numpy=False, **kwargs
) -> Dict[str, Tensor]:
    """`draw`, as tensors on `device`."""
    clone = kwargs.get("clone", False)
    return {
        k: get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy)
        for k, v in draw(rb, batch_size, **kwargs).items()
    }
