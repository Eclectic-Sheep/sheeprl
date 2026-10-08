import pickle

import numpy as np
import pytest

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import EpisodeSampler
from sheeprl.data.store import ReplayStore


def episodes_of(env, lengths, first_id=0):
    """The steps of episodes of `lengths` steps of one environment (the last one still being played), with their
    episode and their step in it."""
    rows = []
    for i, length in enumerate(lengths):
        for t in range(length):
            rows.append((first_id + i, t, float(t == length - 1 and i < len(lengths) - 1)))
    episode, step, done = (np.array(c, dtype=np.float32).reshape(-1, 1, 1) for c in zip(*rows))
    return {"episode": episode, "step": step, "terminated": done, "truncated": np.zeros_like(done)}


def filled(lengths_per_env, size=64):
    rb = ReplayBuffer(size, len(lengths_per_env))
    for env, lengths in enumerate(lengths_per_env):
        rb.add(episodes_of(env, lengths, first_id=100 * env), env_idxes=[env])
    return rb


def test_the_sequences_are_inside_the_episodes_that_ended():
    # The environments at their own rows; the last episode of every environment is still being played
    rb = filled([[5, 9, 4, 7], [12, 3, 6]])
    sample = EpisodeSampler(3, seed=0).sample(rb, 16, n_samples=8)
    episode, step = sample["episode"][..., 0], sample["step"][..., 0]  # [N_samples, Sequence_Length, Batch_Size]
    assert (episode == episode[:, :1]).all() and (np.diff(step, axis=1) == 1).all()
    # The episodes being played (3, 102) are left out
    assert set(np.unique(episode)) <= {0, 1, 2, 100, 101}


def test_the_episodes_shorter_than_a_sequence_are_skipped():
    rb = filled([[2, 5, 2, 9]])
    sample = EpisodeSampler(4, seed=0).sample(rb, 32)
    assert set(np.unique(sample["episode"])) == {1}
    with pytest.raises(RuntimeError, match="No valid episodes"):
        EpisodeSampler(6, seed=0).sample(rb, 1)


def test_the_ends_of_the_episodes_are_prioritized():
    rb = filled([[20, 1]])
    ends = []
    for prioritize in (False, True):
        sample = EpisodeSampler(4, prioritize_ends=prioritize, seed=0).sample(rb, 512)
        ends.append((sample["step"][:, -1] == 19).mean())
    assert ends[1] > 2 * ends[0]


def test_the_online_queue_holds_the_sequences_of_the_episodes_that_ended():
    rb = filled([[7, 9, 3]])
    sampler = EpisodeSampler(3, online=True, seed=0)
    # The sequences of the two episodes that ended, ending at their last step: 7 // 3 + 9 // 3
    first = sampler.sample(rb, 5)
    assert first["episode"][0, :, :, 0].T.tolist()[:5] == [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert first["step"][0, -1, :2, 0].tolist() == [3, 6] and first["step"][0, -1, 2:5, 0].tolist() == [2, 5, 8]
    # Taken once: none is queued until another episode ends
    assert sampler._pending(*sampler._episodes(rb), 10)[1].size == 0


def test_the_oldest_episode_can_be_cut_by_the_new_steps():
    # 30 steps in a buffer of 16: the first episode (12 steps) is gone, the second one (10 steps) lost its first 2 steps
    rb = filled([[12, 10, 8]], size=16)
    sample = EpisodeSampler(3, seed=0).sample(rb, 64)
    assert set(np.unique(sample["episode"])) == {1}
    assert sample["step"].min() == 2


def test_the_index_of_the_episodes_is_rebuilt_from_a_checkpoint():
    store = ReplayStore(filled([[5, 9, 4]]), EpisodeSampler(3, seed=0))
    store.sample(4)
    loaded = pickle.loads(pickle.dumps(store))
    assert loaded.sampler._scanned is None
    assert set(np.unique(loaded.sample(32)["episode"].numpy())) == {0, 1}
