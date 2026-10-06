"""How the training reads a replay buffer: a sampler draws the steps of the samples and the buffer, its storage,
gathers them (`ReplayBuffer.gather`).

A sampler owns its random number generator and its online queue (`buffer.online`): the same storage can be read by
samplers of different kinds, and the state of a sampler is saved with it in the checkpoints (`ReplayStore`).

The samplers of a replay buffer are `ReplaySampler`s:

- `TransitionSampler`: single steps of a `ReplayBuffer`, of shape `[n_samples, batch_size, ...]`;
- `SequenceSampler`: sequences of consecutive steps of a `ReplayBuffer`, every one from a single environment, of shape
  `[n_samples, sequence_length, batch_size, ...]`;
- `EpisodeSampler`: sequences inside the episodes of a `ReplayBuffer`;
The rollout of an on-policy algorithm is read whole instead, and its `EpochSampler` draws the minibatches of the
epochs of its update: it shuffles the indices of the rollout, so it is not a `ReplaySampler`.

When the environments of a `ReplayBuffer` aren't at the same row (steps were added to some of them only, e.g. the first
steps of the ones that ended an episode), the steps and the sequences are drawn from one environment at a time: an
environment, then a step or a sequence among its rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Protocol, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from sheeprl.data.buffers import ReplayBuffer

# No queued sequences: the environments and the first steps of none
_NO_SEQUENCES = (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))


def _oldest_first(envs: np.ndarray, starts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """The first `n` of the sequences of the environments `envs` that start at the steps `starts`: the ones that start
    first, in the order of their environments."""
    order = np.lexsort((envs, starts))[:n]
    return envs[order], starts[order]


class OnlineQueue:
    """The online queue of a `ReplayBuffer` (as in the one of DreamerV3, https://github.com/danijar/dreamerv3): the
    steps of every environment are cut into consecutive sequences of `sequence_length` steps (from its second step, or
    from its first one with sequences of one step), which join the queue when their steps are in the buffer (and the
    step after them, with `sample_next_obs`). The steps are the ones added from the `origin`-th one: the queue isn't
    checkpointed, it restarts empty with the steps added after the loading (`reset`). The sequences overwritten before
    being sampled are dropped."""

    def __init__(self, origin: int | np.ndarray = 0) -> None:
        # The steps added to every environment (or to all of them) before the ones queued
        self.origin = origin
        # The first step of the next sequence of every environment, set at the first sample
        self.next: Optional[np.ndarray] = None

    def reset(self, origin: int | np.ndarray) -> None:
        self.origin, self.next = origin, None

    def pending(
        self, storage: ReplayBuffer, sequence_length: int, sample_next_obs: bool, limit: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The sequences of the queue, up to `limit` of every environment, the oldest first: their environments and
        their first steps (the steps added to the environment before them)."""
        if self.next is None:
            origin = np.asarray(self.origin, dtype=np.int64) + int(sequence_length > 1)
            self.next = np.broadcast_to(origin, (storage.n_envs,)).copy()
        added = storage.env_added
        # The sequences that start before the oldest step in the buffer are dropped
        behind = np.maximum(added - storage.buffer_size - self.next, 0)
        self.next += -(-behind // sequence_length) * sequence_length
        ready = np.maximum((added - int(sample_next_obs) - self.next) // sequence_length, 0)
        counts = np.minimum(ready, limit)
        envs = np.repeat(np.arange(storage.n_envs, dtype=np.intp), counts)
        starts = np.concatenate([n + sequence_length * np.arange(c) for n, c in zip(self.next, counts)])
        return _oldest_first(envs, starts, len(starts))

    def take(self, envs: np.ndarray, starts: np.ndarray, sequence_length: int) -> None:
        """Remove from the queue the sequences of the environments `envs` that start at the steps `starts`: the first
        ones of their environments (`pending`)."""
        np.maximum.at(self.next, envs, starts + sequence_length)


def _generator(seed: int | np.random.SeedSequence | None, rng: np.random.Generator | None) -> np.random.Generator:
    """The random number generator of a sampler: `rng`, or a new one from `seed`."""
    return rng if rng is not None else np.random.default_rng(seed)


class ReplaySampler(Protocol):
    """Draws the steps of the samples of a replay buffer, with its own random number generator (`rng`). The samplers of
    `sheeprl.data` implement it, e.g. `class TransitionSampler(ReplaySampler)`."""

    rng: np.random.Generator

    def sample(
        self,
        storage: ReplayBuffer,
        batch_size: int,
        n_samples: int = 1,
        online: Optional[bool] = None,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        """`n_samples` batches of `batch_size` elements of `storage`. With `online` (by default the one of the
        sampler), the batches start with the sequences of the online queue, the oldest first, and only the rest of
        them is drawn uniformly: every step added is sampled once soon after."""
        raise NotImplementedError

    def reset_online(self, storage: ReplayBuffer) -> None:
        """Restart the online queue from the steps that `storage` holds now: the ones added after are queued."""
        raise NotImplementedError

    def continue_from(self, source: ReplaySampler) -> None:
        """Draw with the generator of `source`, the sampler of a checkpoint: a run resumed from the checkpoint draws
        what it would have drawn without stopping."""
        self.rng = source.rng


def _check_samples(batch_size: int, n_samples: int) -> None:
    if batch_size <= 0 or n_samples <= 0:
        raise ValueError(f"'batch_size' ({batch_size}) and 'n_samples' ({n_samples}) must be both greater than 0")


class TransitionSampler(ReplaySampler):
    """Single steps of a `ReplayBuffer`, drawn uniformly, of shape `[n_samples, batch_size, ...]`.

    With `sample_next_obs`, the next observations (of the observation keys of the buffer) are sampled too, as
    `next_<key>`, and the last step added is never sampled: its next observation isn't in the buffer
    (https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274)."""

    def __init__(
        self,
        sample_next_obs: bool = False,
        online: bool = False,
        seed: int | np.random.SeedSequence | None = None,
        rng: np.random.Generator | None = None,
        queue: OnlineQueue | None = None,
    ):
        self.rng = _generator(seed, rng)
        self.sample_next_obs = sample_next_obs
        self.online = online
        self.queue = queue if queue is not None else OnlineQueue()

    def reset_online(self, storage: ReplayBuffer) -> None:
        self.queue.reset(storage.env_added)

    def draw(self, storage: ReplayBuffer, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """The rows and the environments of `n` steps drawn uniformly."""
        if not storage.lockstep:
            return _draw_per_env(storage, self.rng, n, 1 + int(self.sample_next_obs))
        if not storage.full and storage._pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. Please add at least one sample calling 'self.add()'"
            )
        if storage.full:
            # Every row can be sampled, except the last inserted one if the next observation is needed:
            # the valid rows are the ones in [pos, pos + n_valid) (modulo the buffer size)
            n_valid = storage.buffer_size - int(self.sample_next_obs)
            rows = (storage._pos + self.rng.integers(0, n_valid, size=(n,), dtype=np.intp)) % storage.buffer_size
        else:
            max_pos_to_sample = storage._pos - 1 if self.sample_next_obs else storage._pos
            if max_pos_to_sample == 0:
                raise RuntimeError(
                    "You want to sample the next observations, but one sample has been added to the buffer. "
                    "Make sure that at least two samples are added."
                )
            rows = self.rng.integers(0, max_pos_to_sample, size=(n,), dtype=np.intp)
        return rows, self.rng.integers(0, storage.n_envs, size=(n,), dtype=np.intp)

    def sample(
        self,
        storage: ReplayBuffer,
        batch_size: int,
        n_samples: int = 1,
        online: Optional[bool] = None,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        _check_samples(batch_size, n_samples)
        online = self.online if online is None else online
        n = batch_size * n_samples
        envs, starts = self.queue.pending(storage, 1, self.sample_next_obs, n) if online else _NO_SEQUENCES
        envs, starts = envs[:n], starts[:n]
        rows, env_rows = self.draw(storage, n - len(starts))
        # The queued steps leave the queue once the uniform ones are drawn: a sample that fails leaves it as it is
        if online:
            self.queue.take(envs, starts, 1)
        samples = storage.gather(
            np.concatenate((starts % storage.buffer_size, rows)),
            np.concatenate((envs, env_rows)),
            sample_next_obs=self.sample_next_obs,
            clone=clone,
        )
        return {k: v.reshape(n_samples, batch_size, *v.shape[1:]) for k, v in samples.items()}


class SequenceSampler(ReplaySampler):
    """Sequences of `sequence_length` consecutive steps of a `ReplayBuffer`, every one from a single environment,
    drawn uniformly without considering the ends of the episodes, of shape `[n_samples, sequence_length, batch_size,
    ...]`. A sequence never crosses the position of the next insertion, where the newest steps are followed by the
    oldest ones. With `sample_next_obs`, the steps after the ones of the sequences are sampled too, as `next_<key>`."""

    def __init__(
        self,
        sequence_length: int,
        sample_next_obs: bool = False,
        online: bool = False,
        seed: int | np.random.SeedSequence | None = None,
        rng: np.random.Generator | None = None,
        queue: OnlineQueue | None = None,
    ):
        self.rng = _generator(seed, rng)
        self.sequence_length = sequence_length
        self.sample_next_obs = sample_next_obs
        self.online = online
        self.queue = queue if queue is not None else OnlineQueue()

    def reset_online(self, storage: ReplayBuffer) -> None:
        self.queue.reset(storage.env_added)

    def draw(self, storage: ReplayBuffer, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """The first rows and the environments of `n` sequences drawn uniformly. With `sample_next_obs`, the step after
        the last one of a sequence (its next observation) must be in the buffer too."""
        sequence_length = self.sequence_length
        span = sequence_length + int(self.sample_next_obs)
        if not storage.lockstep:
            return _draw_per_env(storage, self.rng, n, span)
        with_next = " and its next observation" if self.sample_next_obs else ""
        if not storage.full and storage._pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. Please add at least one sample calling 'self.add()'"
            )
        if not storage.full and storage._pos - span + 1 < 1:
            raise ValueError(
                f"Cannot sample a sequence of length {sequence_length}{with_next}. Data added so far: {storage._pos}"
            )
        if storage.full and span > storage.buffer_size:
            raise ValueError(
                f"The sequence length ({sequence_length}){with_next} is greater than the buffer size "
                f"({storage.buffer_size})"
            )
        if storage.full:
            # A sequence (and its next observation) must not cross the position of the next insertion, where the
            # newest data are followed by the oldest ones: the valid first rows are the ones in
            # [pos, pos + buffer_size - span] (modulo the buffer size)
            n_valid = storage.buffer_size - span + 1
            starts = (storage._pos + self.rng.integers(0, n_valid, size=(n,), dtype=np.intp)) % storage.buffer_size
        else:
            # The sequences must not go beyond the steps added
            starts = self.rng.integers(0, storage._pos - span + 1, size=(n,), dtype=np.intp)
        if storage.n_envs == 1:
            envs = np.zeros((n,), dtype=np.intp)
        else:
            envs = self.rng.integers(0, storage.n_envs, size=(n,), dtype=np.intp)
        return starts, envs

    def sample(
        self,
        storage: ReplayBuffer,
        batch_size: int,
        n_samples: int = 1,
        online: Optional[bool] = None,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        _check_samples(batch_size, n_samples)
        online = self.online if online is None else online
        n, sequence_length = batch_size * n_samples, self.sequence_length
        envs, starts = (
            self.queue.pending(storage, sequence_length, self.sample_next_obs, n) if online else _NO_SEQUENCES
        )
        envs, starts = envs[:n], starts[:n]
        first_rows, env_rows = self.draw(storage, n - len(starts))
        # The queued sequences leave the queue once the uniform ones are drawn: a sample that fails leaves it as it is
        if online:
            self.queue.take(envs, starts, sequence_length)
        samples = storage.gather(
            np.concatenate((starts % storage.buffer_size, first_rows)),
            np.concatenate((envs, env_rows)),
            sequence_length=sequence_length,
            sample_next_obs=self.sample_next_obs,
            clone=clone,
        )
        return {k: _sequences(v, n_samples, batch_size) for k, v in samples.items()}


def _draw_per_env(storage: ReplayBuffer, rng: np.random.Generator, n: int, span: int) -> Tuple[np.ndarray, np.ndarray]:
    """The first rows and the environments of `n` spans of `span` steps, drawn uniformly from the environments of a
    buffer that aren't in lockstep (steps were added to some of them only): an environment, then a span among the
    ones that fit in its rows, without crossing the row of its next step."""
    full, pos = storage.env_full, storage.positions
    n_valid = np.where(full, storage.buffer_size, pos) - span + 1
    valid_envs = np.flatnonzero(n_valid > 0)
    if len(valid_envs) == 0:
        raise ValueError(f"No environment of the buffer holds {span} steps")
    envs = valid_envs[rng.integers(0, len(valid_envs), size=(n,))].astype(np.intp)
    offsets = rng.integers(0, n_valid[envs], size=(n,))
    rows = np.where(full[envs], (pos[envs] + offsets) % storage.buffer_size, offsets).astype(np.intp)
    return rows, envs


def _sequences(v: np.ndarray, n_samples: int, batch_size: int) -> np.ndarray:
    """The sequences `v`, of shape `[n_samples * batch_size, sequence_length, ...]`, reshaped to
    `[n_samples, sequence_length, batch_size, ...]`."""
    return np.swapaxes(v.reshape(n_samples, batch_size, *v.shape[1:]), 1, 2)


class EpochSampler:
    """The minibatches of the epochs of an on-policy update: the `n` elements of a rollout (e.g. its steps, or its
    sequences), shuffled at every epoch and split in minibatches of `batch_size` elements, the last one smaller.

    The elements are shuffled with the global generator of PyTorch, as `torch.utils.data.RandomSampler` does. With
    `num_replicas` > 1 they are the elements of the rollouts of all the processes, and every process takes its share of
    every epoch, shuffled with `seed`, as `torch.utils.data.DistributedSampler` does (`buffer.share_data`).

    With `pad_to`, every minibatch is padded to `pad_to` elements with copies of its first one (e.g. to give the same
    shapes to a compiled loss, which masks them out).
    """

    def __init__(
        self, batch_size: int, num_replicas: int = 1, rank: int = 0, seed: int = 0, distributed: bool = False
    ) -> None:
        self.batch_size = batch_size
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.distributed = distributed

    def epochs(
        self,
        n: int,
        epochs: int,
        first_epoch: int = 0,
        pad_to: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Iterator[Tuple[List[int], int]]:
        """The minibatches of `epochs` epochs over `n` elements, from the epoch `first_epoch` (which shuffles the
        elements of the processes): their elements, padded to `pad_to`, and the number of the ones not padded.
        `batch_size` overrides the one of the sampler (e.g. a share of a number of elements that changes)."""
        from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler

        indexes = list(range(n))
        if self.distributed:
            sampler = DistributedSampler(
                indexes, num_replicas=self.num_replicas, rank=self.rank, shuffle=True, seed=self.seed
            )
        else:
            sampler = RandomSampler(indexes)
        batches = BatchSampler(sampler, batch_size=batch_size or self.batch_size, drop_last=False)
        for epoch in range(first_epoch, first_epoch + epochs):
            if self.distributed:
                sampler.set_epoch(epoch)
            for idxes in batches:
                size = len(idxes)
                if pad_to is not None:
                    idxes = idxes + idxes[:1] * (pad_to - size)
                yield idxes, size


class EpisodeSampler(ReplaySampler):
    """Sequences of `sequence_length` steps inside the episodes of a `ReplayBuffer`, of shape
    `[n_samples, sequence_length, batch_size, ...]`, as DreamerV2 samples its episodes: the episode of every sequence
    is drawn uniformly among the ones long enough, then its first step. With `prioritize_ends`, the first steps are
    drawn among all the steps of the episode and the ones too close to its end are moved back: the sequences that end
    at the last step are drawn more often.

    The episodes are the steps of an environment up to the end of an episode (`terminated` or `truncated`): the ones
    still being played are left out, and the oldest one of an environment can be cut, its first steps overwritten by
    the new ones. The sampler indexes them as they are added (its index is rebuilt when it is loaded).

    With `online`, the samples start with the sequences of the episodes ended since the previous sample, the oldest
    first, cut into consecutive sequences that end at their last step (at the one before, with `sample_next_obs`):
    their first steps, fewer than a sequence, are left out. The queue isn't checkpointed: it restarts empty, with the
    episodes that end after the loading.
    """

    def __init__(
        self,
        sequence_length: int,
        sample_next_obs: bool = False,
        prioritize_ends: bool = False,
        online: bool = False,
        seed: int | np.random.SeedSequence | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.rng = _generator(seed, rng)
        self.sequence_length = sequence_length
        self.sample_next_obs = sample_next_obs
        self.prioritize_ends = prioritize_ends
        self.online = online
        # The index of the episodes: the steps of every environment scanned, and the ones that end an episode (in the
        # numbering of the steps added to the environment, whose remainder by the buffer size is their row)
        self._scanned: Optional[np.ndarray] = None
        self._ends: List[np.ndarray] = []
        # The online queue: the steps of every environment after which the episodes are queued, and the sequences
        # taken from every queued episode (environment, last step)
        self._origin: Optional[np.ndarray] = None
        self._taken: Dict[Tuple[int, int], int] = {}

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        # The index is rebuilt from the buffer, the queue restarts empty (`reset_online`)
        state["_scanned"], state["_ends"], state["_taken"] = None, [], {}
        return state

    def reset_online(self, storage: ReplayBuffer) -> None:
        self._origin = storage.env_added
        self._taken = {}

    def _index(self, storage: ReplayBuffer) -> None:
        """Index the episodes that ended in the steps added since the previous call (and in the last step scanned of
        every environment, which may have been changed, e.g. when its environment restarted)."""
        added, size = storage.env_added, storage.buffer_size
        if self._scanned is None:
            self._scanned = np.zeros(storage.n_envs, dtype=np.int64)
            self._ends = [np.empty(0, dtype=np.int64) for _ in range(storage.n_envs)]
        for env in range(storage.n_envs):
            oldest = max(added[env] - size, 0)
            first = max(self._scanned[env] - 1, oldest)
            ends = self._ends[env]
            ends = ends[(ends >= oldest) & (ends < first)]
            if added[env] > first:
                steps = np.arange(first, added[env])
                done = _column(storage, "terminated", steps % size, env) | _column(
                    storage, "truncated", steps % size, env
                )
                ends = np.concatenate((ends, steps[done]))
            self._ends[env] = ends
            self._scanned[env] = added[env]

    def _episodes(self, storage: ReplayBuffer) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The episodes in the buffer, the oldest ending first: their environments, their first steps and their
        lengths."""
        added, size = storage.env_added, storage.buffer_size
        envs, starts, lengths = [], [], []
        for env, ends in enumerate(self._ends):
            if len(ends) == 0:
                continue
            first = np.concatenate(([max(added[env] - size, 0)], ends[:-1] + 1))
            envs.append(np.full(len(ends), env, dtype=np.intp))
            starts.append(first)
            lengths.append(ends - first + 1)
        if not envs:
            return (np.empty(0, dtype=np.intp),) * 3
        envs, starts, lengths = np.concatenate(envs), np.concatenate(starts), np.concatenate(lengths)
        order = np.lexsort((envs, starts + lengths))
        return envs[order], starts[order], lengths[order]

    def _pending(self, envs: np.ndarray, starts: np.ndarray, lengths: np.ndarray, n: int) -> Tuple:
        """The first `n` sequences of the online queue: their environments and their first steps, and the sequences
        they take from their episodes."""
        sequence_length = self.sequence_length
        queued_envs, first_steps, taken = [], [], {}
        for env, start, length in zip(envs, starts, lengths):
            last = int(start + length - 1)
            if self._origin is not None and last < self._origin[env]:
                continue
            steps = int(length) - int(self.sample_next_obs)
            n_sequences = steps // sequence_length
            done = self._taken.get((int(env), last), 0)
            k = max(min(n_sequences - done, n - len(first_steps)), 0)
            offsets = steps % sequence_length + sequence_length * (done + np.arange(k))
            first_steps.extend(start + offsets)
            queued_envs.extend([env] * k)
            if k > 0:
                taken[int(env), last] = done + k
            if len(first_steps) >= n:
                break
        return np.array(queued_envs, dtype=np.intp), np.array(first_steps, dtype=np.int64), taken

    def sample(
        self,
        storage: ReplayBuffer,
        batch_size: int,
        n_samples: int = 1,
        online: Optional[bool] = None,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        _check_samples(batch_size, n_samples)
        online = self.online if online is None else online
        sequence_length, sample_next_obs = self.sequence_length, self.sample_next_obs
        n = batch_size * n_samples
        self._index(storage)
        envs, starts, lengths = self._episodes(storage)
        queued_envs, queued_steps, taken = self._pending(envs, starts, lengths, n) if online else (*_NO_SEQUENCES, {})
        valid = np.flatnonzero(lengths > sequence_length if sample_next_obs else lengths >= sequence_length)
        if len(valid) == 0:
            raise RuntimeError(
                "No valid episodes has been added to the buffer. Please add at least one episode of length greater "
                f"than or equal to {sequence_length} calling `self.add()`"
            )
        # The episode of every sequence, drawn independently: each of the `n_samples` batches is a uniform sample
        episodes = valid[self.rng.integers(0, len(valid), (n - len(queued_steps),))]
        ep_lens = lengths[episodes] - int(sample_next_obs)
        # The last first step that leaves room for a sequence in the episode
        upper = ep_lens - sequence_length + 1
        # With the ends prioritized, every step of the episode can be drawn as the first one
        if self.prioritize_ends:
            upper = upper + sequence_length
        first = np.minimum(self.rng.integers(0, upper), ep_lens - sequence_length)
        # The queued sequences leave the queue once the uniform ones are drawn: a sample that fails leaves it as it is
        if online:
            self._taken.update(taken)
            # The episodes overwritten leave it
            oldest = np.maximum(storage.env_added - storage.buffer_size, 0)
            self._taken = {key: v for key, v in self._taken.items() if key[1] >= oldest[key[0]]}
        steps = np.concatenate((queued_steps, starts[episodes] + first))
        samples = storage.gather(
            (steps % storage.buffer_size).astype(np.intp),
            np.concatenate((queued_envs, envs[episodes])).astype(np.intp),
            sequence_length=sequence_length,
            sample_next_obs=sample_next_obs,
            clone=clone,
        )
        return {k: _sequences(v, n_samples, batch_size) for k, v in samples.items()}


def _column(storage: ReplayBuffer, key: str, rows: np.ndarray, env: int) -> np.ndarray:
    """The values of `key` at the rows `rows` of the environment `env` of a buffer, as booleans: in the memory of the
    CPU, of a device, or memory-mapped."""
    values = storage[key]
    if torch.is_tensor(values):
        return values[torch.as_tensor(rows, device=values.device), env].cpu().numpy().reshape(-1).astype(bool)
    return np.asarray(values[rows, env]).reshape(-1).astype(bool)
