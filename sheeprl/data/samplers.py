"""How the training reads a replay buffer: a sampler draws the steps of the samples and the buffer, its storage,
gathers them (`ReplayBuffer.gather`, `EnvIndependentReplayBuffer.gather`, `EpisodeBuffer.gather`).

A sampler owns its random number generator and its online queue (`buffer.online`): the same storage can be read by
samplers of different kinds, and the state of a sampler is saved with it in the checkpoints (`ReplayStore`).

- `TransitionSampler`: single steps of a `ReplayBuffer`, of shape `[n_samples, batch_size, ...]`;
- `SequenceSampler`: sequences of consecutive steps of a `ReplayBuffer`, of shape
  `[n_samples, sequence_length, batch_size, ...]`;
- `EnvIndependentSampler`: the steps or the sequences of an `EnvIndependentReplayBuffer`, every one from a single
  environment;
- `EpisodeSampler`: sequences inside the episodes of an `EpisodeBuffer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer, ReplayBuffer

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

    def __init__(self, origin: int = 0) -> None:
        self.origin = origin
        # The first step of the next sequence of every environment, set at the first sample
        self.next: Optional[np.ndarray] = None

    def reset(self, origin: int) -> None:
        self.origin, self.next = origin, None

    def pending(
        self, storage: ReplayBuffer, sequence_length: int, sample_next_obs: bool, limit: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The sequences of the queue, up to `limit` of every environment, the oldest first: their environments and
        their first steps (the steps added to the environment before them)."""
        if self.next is None:
            self.next = np.full(storage.n_envs, self.origin + int(sequence_length > 1), dtype=np.int64)
        # The sequences that start before the oldest step in the buffer are dropped
        behind = np.maximum(storage._added - storage.buffer_size - self.next, 0)
        self.next += -(-behind // sequence_length) * sequence_length
        ready = np.maximum((storage._added - int(sample_next_obs) - self.next) // sequence_length, 0)
        counts = np.minimum(ready, limit)
        envs = np.repeat(np.arange(storage.n_envs, dtype=np.intp), counts)
        starts = np.concatenate([n + sequence_length * np.arange(c) for n, c in zip(self.next, counts)])
        return _oldest_first(envs, starts, len(starts))

    def take(self, envs: np.ndarray, starts: np.ndarray, sequence_length: int) -> None:
        """Remove from the queue the sequences of the environments `envs` that start at the steps `starts`: the first
        ones of their environments (`pending`)."""
        np.maximum.at(self.next, envs, starts + sequence_length)


class EpisodeQueue:
    """The online queue of an `EpisodeBuffer`: the steps of every stored episode are cut into consecutive sequences of
    `sequence_length` steps that end at its last step (at the one before, with `sample_next_obs`), which join the queue
    with the episode: its first steps, fewer than a sequence, are left out. The episodes are the ones stored from the
    `episode`-th one: the queue isn't checkpointed, it restarts empty with the episodes stored after the loading
    (`reset`). The episodes removed before being sampled are dropped."""

    def __init__(self, episode: int = 0) -> None:
        # The next episode with sequences in the queue, and the next of its sequences
        self.episode, self.sequence = episode, 0

    def reset(self, episode: int) -> None:
        self.episode, self.sequence = episode, 0

    def pending(
        self, storage: EpisodeBuffer, sequence_length: int, sample_next_obs: bool, n: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """The oldest `n` sequences of the queue, or all of them if they are fewer: their first steps in the storage,
        and the position of the queue after them."""
        lengths = np.diff(storage._cum_lengths, prepend=0)
        oldest = storage._stored - len(storage._starts)
        episode, sequence = self.episode, self.sequence
        if episode < oldest:
            episode, sequence = oldest, 0
        first_steps: List[int] = []
        while len(first_steps) < n and episode < storage._stored:
            steps = lengths[episode - oldest] - int(sample_next_obs)
            n_sequences = steps // sequence_length
            taken = max(min(n_sequences - sequence, n - len(first_steps)), 0)
            offsets = steps % sequence_length + sequence_length * (sequence + np.arange(taken))
            first_steps.extend(storage._starts[episode - oldest] + offsets)
            sequence += taken
            if sequence >= n_sequences:
                episode, sequence = episode + 1, 0
        return np.array(first_steps, dtype=np.intp), (episode, sequence)


class Sampler:
    """Draws the steps of the samples of a storage, with its own random number generator (from `seed`, or `rng`)."""

    def __init__(self, seed: int | np.random.SeedSequence | None = None, rng: np.random.Generator | None = None):
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(seed)

    def sample(
        self, storage, batch_size: int, n_samples: int = 1, online: Optional[bool] = None, clone: bool = False
    ) -> Dict[str, np.ndarray]:
        """`n_samples` batches of `batch_size` elements of `storage`. With `online` (by default the one of the
        sampler), the batches start with the sequences of the online queue, the oldest first, and only the rest of
        them is drawn uniformly: every step added is sampled once soon after."""
        raise NotImplementedError

    def reset_online(self, storage) -> None:
        """Restart the online queue from the steps that `storage` holds now: the ones added after are queued."""
        raise NotImplementedError


def _check_samples(batch_size: int, n_samples: int) -> None:
    if batch_size <= 0 or n_samples <= 0:
        raise ValueError(f"'batch_size' ({batch_size}) and 'n_samples' ({n_samples}) must be both greater than 0")


class TransitionSampler(Sampler):
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
        super().__init__(seed, rng)
        self.sample_next_obs = sample_next_obs
        self.online = online
        self.queue = queue if queue is not None else OnlineQueue()

    def reset_online(self, storage: ReplayBuffer) -> None:
        self.queue.reset(storage._added)

    def draw(self, storage: ReplayBuffer, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """The rows and the environments of `n` steps drawn uniformly."""
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


class SequenceSampler(Sampler):
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
        super().__init__(seed, rng)
        self.sequence_length = sequence_length
        self.sample_next_obs = sample_next_obs
        self.online = online
        self.queue = queue if queue is not None else OnlineQueue()

    def reset_online(self, storage: ReplayBuffer) -> None:
        self.queue.reset(storage._added)

    def draw(self, storage: ReplayBuffer, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """The first rows and the environments of `n` sequences drawn uniformly. With `sample_next_obs`, the step after
        the last one of a sequence (its next observation) must be in the buffer too."""
        sequence_length = self.sequence_length
        span = sequence_length + int(self.sample_next_obs)
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


def _sequences(v: np.ndarray, n_samples: int, batch_size: int) -> np.ndarray:
    """The sequences `v`, of shape `[n_samples * batch_size, sequence_length, ...]`, reshaped to
    `[n_samples, sequence_length, batch_size, ...]`."""
    return np.swapaxes(v.reshape(n_samples, batch_size, *v.shape[1:]), 1, 2)


class EnvIndependentSampler(Sampler):
    """The steps (`sequence_length=None`, as `TransitionSampler`) or the sequences (as `SequenceSampler`) of an
    `EnvIndependentReplayBuffer`: the environment of every element is drawn independently, then its steps in the
    buffer of that environment. Its generator draws the environments, and every environment has a sampler of its own
    (`env_samplers`), whose generators are spawned from the same seed."""

    def __init__(
        self,
        n_envs: int,
        sequence_length: Optional[int] = None,
        sample_next_obs: bool = False,
        online: bool = False,
        seed: int | np.random.SeedSequence | None = None,
        rng: np.random.Generator | None = None,
        env_samplers: Optional[List[TransitionSampler | SequenceSampler]] = None,
    ):
        if env_samplers is None:
            seed_sequences = np.random.SeedSequence(seed).spawn(n_envs + 1)
            env_samplers = [
                (
                    TransitionSampler(sample_next_obs, seed=s)
                    if sequence_length is None
                    else SequenceSampler(sequence_length, sample_next_obs, seed=s)
                )
                for s in seed_sequences[:-1]
            ]
            rng = np.random.default_rng(seed_sequences[-1])
        super().__init__(rng=rng)
        self.sequence_length = sequence_length
        self.sample_next_obs = sample_next_obs
        self.online = online
        self.env_samplers = env_samplers

    def reset_online(self, storage: EnvIndependentReplayBuffer) -> None:
        for buf, sampler in zip(storage.buffer, self.env_samplers):
            sampler.reset_online(buf)

    def _pending(self, storage: EnvIndependentReplayBuffer, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """The oldest `n` sequences of the online queues of the environments, or all of them if they are fewer: their
        environments and their first steps."""
        sequence_length = self.sequence_length or 1
        pending = [
            sampler.queue.pending(buf, sequence_length, self.sample_next_obs, n)[1]
            for buf, sampler in zip(storage.buffer, self.env_samplers)
        ]
        envs = np.repeat(np.arange(storage.n_envs, dtype=np.intp), [len(starts) for starts in pending])
        return _oldest_first(envs, np.concatenate(pending), n)

    def sample(
        self,
        storage: EnvIndependentReplayBuffer,
        batch_size: int,
        n_samples: int = 1,
        online: Optional[bool] = None,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        _check_samples(batch_size, n_samples)
        online = self.online if online is None else online
        sequence_length = self.sequence_length or 1
        n = batch_size * n_samples
        queued_envs, queued_starts = self._pending(storage, n) if online else _NO_SEQUENCES
        # The environment of every element of every batch: the queued ones first, then the ones drawn independently
        # (the batches of one call don't take the same number of elements from each environment)
        env_idxes = np.concatenate((queued_envs, self.rng.integers(0, storage.n_envs, (n - len(queued_envs),))))
        samples: Dict[str, np.ndarray] = {}
        for env, (buf, sampler) in enumerate(zip(storage.buffer, self.env_samplers)):
            positions = np.flatnonzero(env_idxes == env)
            if len(positions) == 0:
                continue
            # All the elements of this environment at once: its queued sequences come first among them
            starts = queued_starts[queued_envs == env] % buf.buffer_size
            rows, buf_env_idxes = (
                sampler.draw(buf, len(positions) - len(starts)) if len(positions) > len(starts) else _NO_SEQUENCES
            )
            env_samples = buf.gather(
                np.concatenate((starts, rows)),
                np.concatenate((np.zeros(len(starts), dtype=np.intp), buf_env_idxes)),
                sequence_length=self.sequence_length,
                sample_next_obs=self.sample_next_obs,
                clone=clone,
            )
            for k, v in env_samples.items():
                if k not in samples:
                    samples[k] = np.empty((n, *v.shape[1:]), dtype=v.dtype)
                # Write them where they were drawn
                samples[k][positions] = v
        # The queued sequences leave the queues once the uniform ones are drawn: a sample that fails leaves them as they
        # are
        for env in np.unique(queued_envs):
            env_starts = queued_starts[queued_envs == env]
            self.env_samplers[env].queue.take(np.zeros(len(env_starts), dtype=np.intp), env_starts, sequence_length)
        if self.sequence_length is None:
            return {k: v.reshape(n_samples, batch_size, *v.shape[1:]) for k, v in samples.items()}
        return {k: _sequences(v, n_samples, batch_size) for k, v in samples.items()}


class EpisodeSampler(Sampler):
    """Sequences of `sequence_length` steps inside the episodes of an `EpisodeBuffer`, of shape
    `[n_samples, sequence_length, batch_size, ...]`: the episode of every sequence is drawn uniformly among the ones
    long enough, then its first step. With `prioritize_ends`, as in DreamerV2, the first steps are drawn among all the
    steps of the episode and the ones too close to its end are moved back: the sequences that end at the last step are
    drawn more often.

    Without `rng` and `seed`, the draws use the global generator of NumPy (`np.random`), as the `EpisodeBuffer` always
    did; with them, a generator of its own."""

    def __init__(
        self,
        sequence_length: int,
        sample_next_obs: bool = False,
        prioritize_ends: bool = False,
        online: bool = False,
        seed: int | np.random.SeedSequence | None = None,
        rng: np.random.Generator | None = None,
        queue: EpisodeQueue | None = None,
    ):
        self.rng = rng if rng is not None or seed is None else np.random.default_rng(seed)
        self.sequence_length = sequence_length
        self.sample_next_obs = sample_next_obs
        self.prioritize_ends = prioritize_ends
        self.online = online
        self.queue = queue if queue is not None else EpisodeQueue()

    def reset_online(self, storage: EpisodeBuffer) -> None:
        self.queue.reset(storage._stored)

    def _integers(self, low, high, size=None) -> np.ndarray:
        if self.rng is None:
            return np.random.randint(low, high, size)
        return self.rng.integers(low, high, size)

    def sample(
        self,
        storage: EpisodeBuffer,
        batch_size: int,
        n_samples: int = 1,
        online: Optional[bool] = None,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError(f"Batch size must be greater than 0, got: {batch_size}")
        if n_samples <= 0:
            raise ValueError(f"The number of samples must be greater than 0, got: {n_samples}")
        online = self.online if online is None else online
        sequence_length, sample_next_obs = self.sequence_length, self.sample_next_obs
        n = batch_size * n_samples
        queued, queue_position = (
            self.queue.pending(storage, sequence_length, sample_next_obs, n)
            if online
            else (np.empty(0, dtype=np.intp), None)
        )
        lengths = np.diff(storage._cum_lengths, prepend=0)
        if sample_next_obs:
            valid_episodes = np.flatnonzero(lengths > sequence_length)
        else:
            valid_episodes = np.flatnonzero(lengths >= sequence_length)
        if len(valid_episodes) == 0:
            raise RuntimeError(
                "No valid episodes has been added to the buffer. Please add at least one episode of length greater "
                f"than or equal to {sequence_length} calling `self.add()`"
            )
        # The episode of every sequence, drawn independently: each of the `n_samples` batches is a uniform sample
        episodes = valid_episodes[self._integers(0, len(valid_episodes), (n - len(queued),))]
        ep_lens = lengths[episodes] - 1 if sample_next_obs else lengths[episodes]
        # The last first step that leaves room for a sequence in the episode
        upper = ep_lens - sequence_length + 1
        # With the ends prioritized, every step of the episode can be drawn as the first one
        if self.prioritize_ends:
            upper += sequence_length
        start_idxes = np.minimum(self._integers(0, upper), ep_lens - sequence_length, dtype=np.intp)
        # The queued sequences leave the queue once the uniform ones are drawn: a sample that fails leaves it as it is
        if online:
            self.queue.episode, self.queue.sequence = queue_position
        # The steps of the sequences in the storage, where an episode can continue from its end to its start: the
        # queued sequences first
        first_steps = np.concatenate((queued, np.array(storage._starts, dtype=np.intp)[episodes] + start_idxes))
        samples = storage.gather(first_steps, sequence_length, sample_next_obs=sample_next_obs, clone=clone)
        return {k: _sequences(v, n_samples, batch_size) for k, v in samples.items()}
