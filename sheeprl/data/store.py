"""The replay buffer of the off-policy algorithms: a storage, where the player writes the steps, and a sampler, which
draws the batches of the training from it (`sheeprl.data.samplers`)."""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from sheeprl.data.buffers import get_tensor


class _Prefetch:
    """The batches sampled for the next call of `ReplayStore.batches`, by a thread: ready once `thread` ends."""

    def __init__(self, batch_size: int, n_samples: int) -> None:
        self.batch_size, self.n_samples = batch_size, n_samples
        self.samples: Optional[Dict[str, Tensor]] = None
        self.event: Optional[torch.cuda.Event] = None
        self.error: Optional[BaseException] = None
        self.thread: Optional[threading.Thread] = None


class ReplayStore:
    """A storage (`ReplayBuffer`, `EnvIndependentReplayBuffer` or `EpisodeBuffer`) read by a sampler, whose batches
    are moved to `device`.

    The store is saved in the checkpoints with its storage and its sampler, whose generator continues where it was: a
    resumed run draws the batches that the run would have drawn without stopping (the online queue restarts empty,
    with the steps added after the loading).

    With `prefetch`, while the training uses the batches of a sample (`batches`), a thread samples the next ones: the
    next ones of the iteration, or the first ones of the next iteration, which are gathered before its steps are
    written (`add` waits for them). They are moved to the device from pinned memory, on a CUDA stream of their own. The
    first batches of an iteration are then sampled from the steps written until the training of the previous one, and
    the sampling (on the CPU) and the copy to the device overlap the training (on the device).

    Args:
        storage: where the steps are written (`add`) and read from.
        sampler: what is read: it draws the steps of the samples, which the storage gathers.
        device: the device of the sampled batches.
        from_numpy: whether the samples are converted with `torch.from_numpy` (`buffer.from_numpy`).
        prefetch: whether the batches of the next iteration are sampled in the background (`buffer.prefetch`).
    """

    def __init__(
        self,
        storage: Any,
        sampler: Any,
        device: str | torch.device = "cpu",
        from_numpy: bool = False,
        prefetch: bool = False,
    ):
        self.storage = storage
        self.sampler = sampler
        self.device = torch.device(device)
        self.from_numpy = from_numpy
        self.prefetch = prefetch
        self._init_transient()

    def _init_transient(self) -> None:
        # Held while the storage or the generator of the sampler are used: by the prefetching thread until its batches
        # are gathered
        self._lock = threading.Lock()
        self._prefetched: Optional[_Prefetch] = None
        self._stream: Optional[torch.cuda.Stream] = None

    def __getstate__(self) -> Dict[str, Any]:
        # A checkpoint waits for the batches being prefetched: the generator of the sampler is saved after them
        self.wait()
        state = self.__dict__.copy()
        for k in ("_lock", "_prefetched", "_stream"):
            state.pop(k)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        state.setdefault("prefetch", False)
        self.__dict__.update(state)
        self._init_transient()

    def add(self, *args, **kwargs) -> None:
        """Write steps in the storage (its `add`), once the batches being prefetched are gathered."""
        with self._lock:
            self.storage.add(*args, **kwargs)

    def wait(self) -> None:
        """Wait for the batches being prefetched: then the storage can be read and changed directly."""
        if self._prefetched is not None and self._prefetched.thread is not None:
            self._prefetched.thread.join()

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
        with self._lock:
            samples = sampler.sample(self.storage, batch_size, n_samples, online=online)
        return {
            k: (
                (v.cpu().numpy() if torch.is_tensor(v) else v)
                if k in numpy_keys
                else get_tensor(v, device=self.device, from_numpy=self.from_numpy)
            )
            for k, v in samples.items()
        }

    @property
    def on_device(self) -> bool:
        """Whether the storage is in the memory of a device (`ReplayBuffer` with `device`): its batches are gathered
        there, and there is nothing to prefetch."""
        buffers = getattr(self.storage, "buffer", None)
        buffers = buffers if isinstance(buffers, tuple) else (self.storage,)
        return any(getattr(b, "device", None) is not None for b in buffers)

    def batches(self, n_steps: int, batch_size: int, max_sampled: Optional[int] = None) -> Iterator[Dict[str, Tensor]]:
        """The batches of `n_steps` gradient steps, in single precision, sampled `max_sampled` at a time (all at once
        when `None`): the samples of many gradient steps (e.g. of a pretraining) may not fit in the memory.

        With `prefetch`, once the batches of a sample are taken, the next sample is prefetched while the training uses
        them: the next one of this call, drawn as without prefetching, or the first one of the next call, expected
        to be as this one, drawn from the steps written until now (the training doesn't write any)."""
        chunk = n_steps if max_sampled is None else max_sampled
        sizes = [min(chunk, n_steps - first) for first in range(0, n_steps, chunk)]
        prefetch = self.prefetch and not self.on_device
        for i, n_samples in enumerate(sizes):
            sample = self._take_prefetched(batch_size, n_samples) if prefetch else None
            if sample is None:
                sample = self.sample(batch_size, n_samples)
            if prefetch:
                self._start_prefetch(batch_size, sizes[i + 1] if i + 1 < len(sizes) else sizes[0])
            for j in range(n_samples):
                yield {k: v[j].float() for k, v in sample.items()}

    def _start_prefetch(self, batch_size: int, n_samples: int) -> None:
        prefetched = _Prefetch(batch_size, n_samples)
        # The lock is taken here, before anything else can use the storage or the generator, and released by the
        # thread once its batches are gathered: the sampling happens at this point of the run, whatever the timing
        self._lock.acquire()
        prefetched.thread = threading.Thread(target=self._prefetch, args=(prefetched,), daemon=True)
        self._prefetched = prefetched
        prefetched.thread.start()

    def _prefetch(self, prefetched: _Prefetch) -> None:
        try:
            try:
                samples = self.sampler.sample(self.storage, prefetched.batch_size, prefetched.n_samples)
            finally:
                self._lock.release()
            if self.device.type == "cuda":
                if self._stream is None:
                    self._stream = torch.cuda.Stream(self.device)
                with torch.cuda.stream(self._stream):
                    prefetched.samples = {
                        k: torch.from_numpy(np.ascontiguousarray(v)).pin_memory().to(self.device, non_blocking=True)
                        for k, v in samples.items()
                    }
                    prefetched.event = torch.cuda.Event()
                    prefetched.event.record(self._stream)
            else:
                prefetched.samples = {
                    k: get_tensor(v, device=self.device, from_numpy=self.from_numpy) for k, v in samples.items()
                }
        except BaseException as e:  # raised by the training, when it takes the batches
            prefetched.error = e

    def _take_prefetched(self, batch_size: int, n_samples: int) -> Optional[Dict[str, Tensor]]:
        """The prefetched batches, if they are `n_samples` batches of `batch_size` elements (else they are dropped)."""
        prefetched, self._prefetched = self._prefetched, None
        if prefetched is None:
            return None
        prefetched.thread.join()
        if prefetched.error is not None:
            raise prefetched.error
        if (prefetched.batch_size, prefetched.n_samples) != (batch_size, n_samples):
            return None
        if prefetched.event is not None:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(prefetched.event)
            # The memory of the batches, allocated on the stream of the prefetching, is used on the current one
            for v in prefetched.samples.values():
                v.record_stream(stream)
        return prefetched.samples

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
        self.wait()
        self.sampler.continue_from(source)
        self.sampler.reset_online(storage)
        # Where this run keeps it (the checkpoints are loaded in the memory of the CPU)
        self.storage = storage.to(getattr(self.storage, "device", None))
        return self
