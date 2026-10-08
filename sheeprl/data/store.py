"""Where every algorithm writes the steps it plays and reads what it trains on: a `ReplayStore`, a `ReplayBuffer` where
the writer writes the steps and a sampler of `sheeprl.data.samplers`, which draws the batches of the training from it.

The stores are built here for the three ways the algorithms train on their data:

- `rollout_store`: the rollout of an on-policy algorithm (PPO, A2C, PPO recurrent), whose `EpochSampler` draws the
  minibatches of an update;
- `transition_store`: the replay buffer of an off-policy algorithm trained on single steps (SAC, DroQ, SAC-AE), read by
  a `TransitionSampler`;
- `sequence_store`: the replay buffer of an algorithm trained on sequences of one environment (the Dreamers), read by a
  `SequenceSampler` or an `EpisodeSampler`, with `env_buffer_size` steps per environment, or by the
  `CuriousSequenceSampler` of Curious Replay (`buffer.curious`) for DreamerV3 and DreamerV3.5.

The training loop saves the replay buffers of the off-policy algorithms in the checkpoints when `buffer.checkpoint` is
set (`load_replay_buffer` restores them).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
from lightning import Fabric
from torch import Tensor

from sheeprl.data.buffers import ReplayBuffer, get_tensor
from sheeprl.data.samplers import (
    CuriousSequenceSampler,
    EpisodeSampler,
    EpochSampler,
    ReplaySampler,
    SequenceSampler,
    TransitionSampler,
)
from sheeprl.utils import fs

if TYPE_CHECKING:
    from sheeprl.core.collector import Act
    from sheeprl.core.environment import EnvStep


class _Prefetch:
    """The batches sampled for the next call of `ReplayStore.batches`, by a thread: ready once `thread` ends."""

    def __init__(self, batch_size: int, n_samples: int) -> None:
        self.batch_size, self.n_samples = batch_size, n_samples
        self.samples: Optional[Dict[str, Tensor]] = None
        self.event: Optional[torch.cuda.Event] = None
        self.error: Optional[BaseException] = None
        self.thread: Optional[threading.Thread] = None


class ReplayStore:
    """A `ReplayBuffer` read by a sampler, whose batches are moved to `device`.

    The off-policy algorithms sample their batches (`batches`, `sample`) from the steps of the whole training. The
    on-policy algorithms read their rollout whole (`read`), compute from it what they train on (e.g. the advantages),
    and draw the minibatches of its epochs from that with an `EpochSampler` (`minibatches`); the writer keeps the last
    step written (`last_step`) and the actions it was played with (`last_act`): what follows the last step of the
    rollout (e.g. the observations whose value bootstraps the returns).

    The store of an off-policy algorithm is saved in the checkpoints with its storage and its sampler, whose generator
    continues where it was: a resumed run draws the batches that the run would have drawn without stopping (the online
    queue restarts empty, with the steps added after the loading).

    With `prefetch`, while the training uses the batches of a sample (`batches`), a thread samples the next ones: the
    next ones of the iteration, or the first ones of the next iteration, which are gathered before its steps are
    written (`add` waits for them). They are moved to the device from pinned memory, on a CUDA stream of their own. The
    first batches of an iteration are then sampled from the steps written until the training of the previous one, and
    the sampling (on the CPU) and the copy to the device overlap the training (on the device).

    Args:
        storage: the `ReplayBuffer` where the steps are written (`add`) and read from.
        sampler: what is read: a `ReplaySampler` draws the steps of the samples, which the storage gathers; an
            `EpochSampler` draws the minibatches of a rollout.
        device: the device of the sampled batches.
        from_numpy: whether the samples are converted with `torch.from_numpy` (`buffer.from_numpy`).
        prefetch: whether the batches of the next iteration are sampled in the background (`buffer.prefetch`).
    """

    def __init__(
        self,
        storage: ReplayBuffer,
        sampler: ReplaySampler | EpochSampler,
        device: str | torch.device = "cpu",
        from_numpy: bool = False,
        prefetch: bool = False,
    ):
        self.storage = storage
        self.sampler = sampler
        self.device = torch.device(device)
        self.from_numpy = from_numpy
        self.prefetch = prefetch
        # The last step written and the actions it was played with, set by the writer: what follows the last step of a
        # rollout (e.g. the observations whose value bootstraps the returns)
        self.last_step: Optional[EnvStep] = None
        self.last_act: Optional[Act] = None
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
        state["last_step"] = state["last_act"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
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

    def update_priorities(self, steps: np.ndarray, losses: Tensor | np.ndarray) -> None:
        """Give the losses of the world model on the steps `steps` of a batch, after the training on it, to the sampler
        of Curious Replay (`CuriousSequenceSampler.update`), which writes their priorities. The losses are read on the
        CPU once."""
        if torch.is_tensor(losses):
            losses = losses.detach().float().cpu().numpy()
        with self._lock:
            self.sampler.update(self.storage, steps, losses)

    def read(self) -> Dict[str, Tensor]:
        """The steps of the storage, `[Buffer_Size, Num_Envs, ...]` (e.g. the whole rollout of an on-policy algorithm),
        on the device, in their dtypes."""
        return self.storage.to_tensor(dtype=None, device=self.device, from_numpy=self.from_numpy)

    def minibatches(self, data: Dict[str, Tensor], epochs: int, dim: int = 0) -> Iterator[Dict[str, Tensor]]:
        """The minibatches of `epochs` epochs over the elements of `data` along its dimension `dim` (e.g. the steps of a
        flattened rollout, or its sequences), drawn by the sampler (an `EpochSampler`)."""
        n = next(iter(data.values())).shape[dim]
        for idxes, _ in self.sampler.epochs(n, epochs):
            index = torch.as_tensor(idxes, device=next(iter(data.values())).device)
            yield {k: v.index_select(dim, index) for k, v in data.items()}

    @property
    def on_device(self) -> bool:
        """Whether the storage is in the memory of a device (`ReplayBuffer` with `device`): its batches are gathered
        there, and there is nothing to prefetch."""
        return getattr(self.storage, "device", None) is not None

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

    def load(self, saved: "ReplayStore") -> "ReplayStore":
        """This store with the storage of `saved`, a store from a checkpoint, and its sampler continuing the generator
        of the sampler of `saved` (and its priorities, with Curious Replay), with its own configuration: e.g. a
        finetuning that loads the buffer of its exploration samples it as configured. The online queue restarts empty,
        with the steps added from now on."""
        self.wait()
        self.sampler.continue_from(saved.sampler)
        self.sampler.reset_online(saved.storage)
        # Where this run keeps it (the checkpoints are loaded in the memory of the CPU)
        self.storage = saved.storage.to(self.storage.device)
        return self


def rollout_store(fabric: Fabric, cfg: Dict[str, Any], log_dir: str, size: int) -> ReplayStore:
    """The store of an on-policy algorithm: a rollout of `size` steps of every environment of the process, whose
    minibatches have `algo.per_rank_batch_size` steps (from the rollouts of all the processes with
    `buffer.share_data`)."""
    curious_replay(cfg, supported=False)
    storage = ReplayBuffer(
        size,
        cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=fs.memmap_dir(cfg, log_dir, fabric.global_rank),
        obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
    )
    sampler = EpochSampler(
        cfg.algo.per_rank_batch_size,
        fabric.world_size,
        fabric.global_rank,
        seed=cfg.seed,
        distributed=cfg.buffer.get("share_data", False),
    )
    return ReplayStore(storage, sampler, fabric.device, cfg.buffer.from_numpy)


def transition_store(
    fabric: Fabric, cfg: Dict[str, Any], log_dir: str, obs_keys: Tuple[str, ...] = ("observations",)
) -> ReplayStore:
    """The store of an off-policy algorithm trained on single steps (SAC, DroQ, SAC-AE): a replay buffer of
    `buffer.size` steps split among the environments of all the processes (1 in a dry run), sampled one step at a time
    (`TransitionSampler`, with the next observations with `buffer.sample_next_obs` and the online queue with
    `buffer.online`). Every process trains on its own data, as the Dreamers do, and `update` averages the gradients over
    the processes."""
    curious_replay(cfg, supported=False)
    storage = ReplayBuffer(
        cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 1,
        cfg.env.num_envs,
        obs_keys=obs_keys,
        memmap=cfg.buffer.memmap and not cfg.buffer.on_device,
        memmap_dir=fs.memmap_dir(cfg, log_dir, fabric.global_rank),
        device=fabric.device if cfg.buffer.on_device else None,
    )
    sampler = TransitionSampler(cfg.buffer.sample_next_obs, cfg.buffer.online, seed=cfg.seed + fabric.global_rank)
    return ReplayStore(storage, sampler, fabric.device, cfg.buffer.from_numpy, cfg.buffer.prefetch)


def curious_replay(cfg: Dict[str, Any], supported: bool = True) -> Optional[Dict[str, Any]]:
    """The configuration of Curious Replay (`buffer.curious`) when it is enabled, else `None` (also with a configuration
    saved before it existed). It raises when it is enabled for an algorithm that doesn't train with it (not
    `supported`): its sequences would be drawn with priorities never updated."""
    curious = cfg.buffer.get("curious") or {}
    if not curious.get("enabled", False):
        return None
    if not supported:
        raise ValueError(
            f"Curious Replay (`buffer.curious.enabled=True`) is implemented for DreamerV3 and DreamerV3.5, "
            f"not for {cfg.algo.name}"
        )
    return curious


def env_buffer_size(fabric: Fabric, cfg: Dict[str, Any], dry_run_size: int) -> int:
    """The capacity of the replay buffer of every environment: `buffer.size` split among the environments of all the
    processes, or `dry_run_size` in a dry run. It must hold a sequence of `algo.per_rank_sequence_length` steps (a dry
    run makes it large enough)."""
    sequence_length = cfg.algo.per_rank_sequence_length
    if cfg.dry_run:
        return max(dry_run_size, sequence_length)
    size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size)
    if size < sequence_length:
        raise ValueError(
            f"The replay buffer of every environment holds `buffer.size // (env.num_envs * world_size)` = {size} "
            f"steps, fewer than a sequence (`algo.per_rank_sequence_length={sequence_length}`): increase `buffer.size`"
        )
    return size


def sequence_store(
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    buffer_size: int,
    sequence_length: int,
    buffer_type: str = "sequential",
    curious: bool = False,
) -> ReplayStore:
    """The store of an algorithm trained on sequences (the Dreamers): a buffer of `buffer_size` steps per environment
    (`env_buffer_size`), every environment written at its own row (its first steps after the end of an episode),
    sampled in sequences of `sequence_length` steps of a single environment: anywhere (`buffer_type="sequential"`,
    `SequenceSampler`) or inside the episodes that ended (`"episode"`, `EpisodeSampler`, with their ends prioritized
    with `buffer.prioritize_ends`), with the online queue with `buffer.online`. Every process samples its buffer with a
    generator of its own.

    An algorithm that trains with Curious Replay (`curious`: it gives the losses of the world model on every batch to
    `ReplayStore.update_priorities`) draws the sequences anywhere by their priorities (`CuriousSequenceSampler`) when
    `buffer.curious.enabled` is set."""
    buffer_type = buffer_type.lower()
    if buffer_type not in ("sequential", "episode"):
        raise ValueError(f"Unrecognized buffer type: must be one of `sequential` or `episode`, received: {buffer_type}")
    curious_cfg = curious_replay(cfg, supported=curious and buffer_type == "sequential")
    storage = ReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
        memmap=cfg.buffer.memmap and not cfg.buffer.on_device,
        memmap_dir=fs.memmap_dir(cfg, log_dir, fabric.global_rank),
        device=fabric.device if cfg.buffer.on_device else None,
    )
    if buffer_type == "episode":
        sampler = EpisodeSampler(
            sequence_length,
            prioritize_ends=cfg.buffer.prioritize_ends,
            online=cfg.buffer.online,
            seed=cfg.seed + fabric.global_rank,
        )
    elif curious_cfg is not None:
        sampler = CuriousSequenceSampler(
            sequence_length,
            c=float(curious_cfg["c"]),
            beta=float(curious_cfg["beta"]),
            alpha=float(curious_cfg["alpha"]),
            epsilon=float(curious_cfg["epsilon"]),
            initial_priority=float(curious_cfg["initial_priority"]),
            online=cfg.buffer.online,
            seed=cfg.seed + fabric.global_rank,
        )
    else:
        sampler = SequenceSampler(sequence_length, online=cfg.buffer.online, seed=cfg.seed + fabric.global_rank)
    return ReplayStore(storage, sampler, fabric.device, cfg.buffer.from_numpy, cfg.buffer.prefetch)


def load_replay_buffer(fabric: Fabric, saved: Any, store: ReplayStore) -> ReplayStore:
    """The replay buffer of this process, from the one saved in a checkpoint (`saved`) in place of `store`
    (`ReplayStore.load`).

    A checkpoint holds the list of the stores of all the processes, or a single store, which every process then starts
    from.
    """
    if isinstance(saved, list):
        if len(saved) != fabric.world_size:
            raise RuntimeError(
                f"The checkpoint holds {len(saved)} replay buffer(s), "
                f"but {fabric.world_size} processes are instantiated"
            )
        saved = saved[fabric.global_rank]
    return store.load(saved)
