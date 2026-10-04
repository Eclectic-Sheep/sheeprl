"""The replay buffer of the off-policy algorithms: a storage, where the player writes the steps, and a sampler, which
draws the batches of the training from it (`sheeprl.data.samplers`)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Sequence

import torch
from torch import Tensor

from sheeprl.data.buffers import get_tensor


class ReplayStore:
    """A storage (`ReplayBuffer`, `EnvIndependentReplayBuffer` or `EpisodeBuffer`) read by a sampler, whose batches
    are moved to `device`.

    The store is saved in the checkpoints with its storage and its sampler, whose generator continues where it was: a
    resumed run draws the batches that the run would have drawn without stopping (the online queue restarts empty,
    with the steps added after the loading).

    Args:
        storage: where the steps are written (`add`) and read from.
        sampler: what is read: it draws the steps of the samples, which the storage gathers.
        device: the device of the sampled batches.
        from_numpy: whether the samples are converted with `torch.from_numpy` (`buffer.from_numpy`).
    """

    def __init__(self, storage: Any, sampler: Any, device: str | torch.device = "cpu", from_numpy: bool = False):
        self.storage = storage
        self.sampler = sampler
        self.device = torch.device(device)
        self.from_numpy = from_numpy

    def add(self, *args, **kwargs) -> None:
        """Write steps in the storage (its `add`)."""
        self.storage.add(*args, **kwargs)

    def sample(
        self,
        batch_size: int,
        n_samples: int = 1,
        online: Optional[bool] = None,
        numpy_keys: Sequence[str] = (),
        sampler: Any = None,
    ) -> Dict[str, Tensor]:
        """`n_samples` batches of `batch_size` elements, on the device and in the dtypes of the storage, but the keys
        `numpy_keys`, left as NumPy arrays on the CPU. `online` overrides the one of the sampler, and `sampler` the
        sampler (e.g. one that shares its generator, to sample in another way)."""
        sampler = self.sampler if sampler is None else sampler
        samples = sampler.sample(self.storage, batch_size, n_samples, online=online)
        return {
            k: v if k in numpy_keys else get_tensor(v, device=self.device, from_numpy=self.from_numpy)
            for k, v in samples.items()
        }

    def batches(self, n_steps: int, batch_size: int, max_sampled: Optional[int] = None) -> Iterator[Dict[str, Tensor]]:
        """The batches of `n_steps` gradient steps, in single precision, sampled `max_sampled` at a time (all at once
        when `None`): the samples of many gradient steps (e.g. of a pretraining) may not fit in the memory."""
        chunk = n_steps if max_sampled is None else max_sampled
        for first in range(0, n_steps, chunk):
            n_samples = min(chunk, n_steps - first)
            sample = self.sample(batch_size, n_samples)
            for i in range(n_samples):
                yield {k: v[i].float() for k, v in sample.items()}

    def load(self, saved: Any) -> "ReplayStore":
        """This store with the storage of `saved`, a store from a checkpoint, and its sampler continuing the generators
        of the sampler of `saved` (with its own configuration: e.g. a finetuning that loads the buffer of its
        exploration samples it as configured). A checkpoint of sheeprl up to 0.8.2 holds the storage itself, whose
        generators the sampler continues. The online queue restarts empty, with the steps added from now on."""
        storage, source = (saved.storage, saved.sampler) if isinstance(saved, ReplayStore) else (saved, saved)
        if not isinstance(storage, type(self.storage)):
            raise RuntimeError(
                f"The checkpoint holds a replay buffer of type {type(storage).__name__}, "
                f"but this run uses a {type(self.storage).__name__}"
            )
        self.sampler.continue_from(source)
        self.sampler.reset_online(storage)
        self.storage = storage
        return self
