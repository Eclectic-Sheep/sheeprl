"""The online queue of the replay buffers (`sample(online=True)`, `buffer.online`): the samples start with the steps
added since the previous ones, the oldest first, and only the rest of them is sampled uniformly."""

import pickle

import numpy as np
import pytest

from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer, ReplayBuffer, SequentialReplayBuffer


def add_steps(rb, first: int, n: int, envs=None) -> None:
    """Add the steps `first, ..., first + n - 1` of the environments `envs` (all by default): their value is
    `10 * step + environment`."""
    envs = list(range(rb.n_envs)) if envs is None else envs
    value = 10 * np.arange(first, first + n).reshape(n, 1, 1) + np.array(envs).reshape(1, -1, 1)
    data = {"value": value.astype(np.float32)}
    if envs == list(range(rb.n_envs)):
        rb.add(data)
    else:
        rb.add(data, envs)


def test_the_transitions_of_the_online_queue_come_first():
    rb = ReplayBuffer(10, 2, obs_keys=("value",), seed=0)
    add_steps(rb, 0, 3)
    # The transitions of the steps 0 and 1 of both environments, in the order they were added
    value = rb.sample(2, n_samples=2, online=True)["value"][..., 0]
    assert value.tolist() == [[0, 1], [10, 11]]
    # Then the ones of step 2, and the rest of the batch is sampled uniformly
    value = rb.sample(4, online=True)["value"][0, :, 0]
    assert value[:2].tolist() == [20, 21]
    add_steps(rb, 3, 1)
    assert rb.sample(4, online=True)["value"][0, :2, 0].tolist() == [30, 31]
    # Without the online queue, a batch is sampled uniformly: the queue is left as it is
    add_steps(rb, 4, 1)
    rb.sample(4)
    assert rb.sample(4, online=True)["value"][0, :2, 0].tolist() == [40, 41]


def test_a_transition_joins_the_online_queue_with_its_next_observation():
    rb = ReplayBuffer(10, 1, obs_keys=("value",), seed=0)
    add_steps(rb, 0, 3)
    sample = rb.sample(4, sample_next_obs=True, online=True)
    assert sample["value"][0, :2, 0].tolist() == [0, 10] and sample["next_value"][0, :2, 0].tolist() == [10, 20]
    assert 20 not in sample["value"]
    add_steps(rb, 3, 1)
    sample = rb.sample(4, sample_next_obs=True, online=True)
    assert sample["value"][0, 0, 0] == 20 and sample["next_value"][0, 0, 0] == 30


def test_the_online_queue_drops_the_overwritten_steps():
    rb = ReplayBuffer(4, 1, obs_keys=("value",), seed=0)
    add_steps(rb, 0, 6)
    # The buffer holds the steps from 2 on
    assert rb.sample(3, online=True)["value"][0, :, 0].tolist() == [20, 30, 40]
    assert rb.sample(3, online=True)["value"][0, 0, 0] == 50


def test_the_sequences_of_the_online_queue_follow_each_other():
    rb = SequentialReplayBuffer(20, 2, seed=0)
    add_steps(rb, 0, 10)
    # As in DreamerV3, the sequences follow each other from the second step: 1-3, 4-6 and 7-9 of every environment
    value = rb.sample(2, n_samples=2, sequence_length=3, online=True)["value"][..., 0]
    assert value.shape == (2, 3, 2)
    assert value[0].T.tolist() == [[10, 20, 30], [11, 21, 31]]
    assert value[1].T.tolist() == [[40, 50, 60], [41, 51, 61]]
    value = rb.sample(3, sequence_length=3, online=True)["value"][0, :, :, 0]
    assert value[:, :2].T.tolist() == [[70, 80, 90], [71, 81, 91]]
    # Sequences of one step start from the first one
    rb = SequentialReplayBuffer(20, 1, seed=0)
    add_steps(rb, 0, 2)
    assert rb.sample(2, sequence_length=1, online=True)["value"][0, 0, :, 0].tolist() == [0, 10]


@pytest.mark.parametrize("buffer_cls", [ReplayBuffer, SequentialReplayBuffer])
def test_the_online_queues_of_the_environments_are_taken_oldest_first(buffer_cls):
    # The environment 1 has more steps than the environment 0: the sequences that start first come first, the ones of
    # the environment 0 before the ones of the environment 1 at the same step
    rb = EnvIndependentReplayBuffer(20, 2, buffer_cls=buffer_cls, seed=0)
    if buffer_cls is ReplayBuffer:
        add_steps(rb, 0, 2)
        add_steps(rb, 2, 1, envs=[1])
        assert rb.sample(5, online=True)["value"][0, :, 0].tolist() == [0, 1, 10, 11, 21]
    else:
        add_steps(rb, 0, 7)
        add_steps(rb, 7, 3, envs=[1])
        value = rb.sample(5, sequence_length=3, online=True)["value"][0, :, :, 0]
        expected = [[10, 20, 30], [11, 21, 31], [40, 50, 60], [41, 51, 61], [71, 81, 91]]
        assert value.T.tolist() == expected
    # The queues are empty: the batches are sampled uniformly
    kwargs = {} if buffer_cls is ReplayBuffer else {"sequence_length": 3}
    assert rb.sample(5, online=True, **kwargs)["value"].shape == rb.sample(5, **kwargs)["value"].shape


def test_a_failed_sample_leaves_the_online_queue_as_it_was():
    # The environment 1 has no steps: a sample that draws it fails after the queued transitions of the environment 0
    # have been chosen, which must stay in the queue
    rb = EnvIndependentReplayBuffer(10, 2, buffer_cls=ReplayBuffer, seed=0)
    add_steps(rb, 0, 3, envs=[0])
    with pytest.raises(ValueError, match="No sample has been added"):
        rb.sample(8, online=True)
    add_steps(rb, 0, 1, envs=[1])
    assert rb.sample(4, online=True)["value"][0, :, 0].tolist() == [0, 1, 10, 20]


def episode(values, rb: EpisodeBuffer) -> None:
    """Add an episode of the steps of values `values`, which ends with its last step."""
    n = len(values)
    terminated = np.zeros((n, 1, 1), dtype=np.float32)
    terminated[-1] = 1
    data = {"value": np.array(values, dtype=np.float32).reshape(n, 1, 1), "terminated": terminated}
    rb.add({**data, "truncated": np.zeros_like(terminated)})


@pytest.mark.parametrize("sample_next_obs", [False, True])
def test_the_sequences_of_the_online_queue_end_with_the_episodes(sample_next_obs):
    rb = EpisodeBuffer(100, 3, obs_keys=("value",))
    episode(range(8), rb)
    # The first steps of the episode, fewer than a sequence, are left out (the last one too, with the next observations)
    value = rb.sample(2, sequence_length=3, sample_next_obs=sample_next_obs, online=True)["value"][0, :, :, 0]
    expected = [[1, 2, 3], [4, 5, 6]] if sample_next_obs else [[2, 3, 4], [5, 6, 7]]
    assert value.T.tolist() == expected
    episode(range(10, 14), rb)
    value = rb.sample(2, sequence_length=3, sample_next_obs=sample_next_obs, online=True)["value"][0, :, :, 0]
    assert value[:, 0].tolist() == ([10, 11, 12] if sample_next_obs else [11, 12, 13])


def test_the_online_queue_drops_the_removed_episodes():
    rb = EpisodeBuffer(10, 3, obs_keys=("value",))
    episode(range(4), rb)
    episode(range(10, 14), rb)
    episode(range(20, 28), rb)
    # The first two episodes made room for the third one
    value = rb.sample(2, sequence_length=3, online=True)["value"][0, :, :, 0]
    assert value.T.tolist() == [[22, 23, 24], [25, 26, 27]]


def test_the_online_queue_restarts_with_the_steps_added_after_a_checkpoint():
    rb = SequentialReplayBuffer(20, 1, seed=0)
    add_steps(rb, 0, 7)
    rb = pickle.loads(pickle.dumps(rb))
    # The sequences of the steps before the checkpoint are left out: the ones from the second step after it are queued
    add_steps(rb, 7, 4)
    assert rb.sample(2, sequence_length=3, online=True)["value"][0, :, 0, 0].tolist() == [80, 90, 100]
    rb = EpisodeBuffer(100, 3, obs_keys=("value",))
    episode(range(8), rb)
    rb = pickle.loads(pickle.dumps(rb))
    episode(range(10, 14), rb)
    assert rb.sample(1, sequence_length=3, online=True)["value"][0, :, 0, 0].tolist() == [11, 12, 13]


def test_the_buffers_saved_by_older_versions_count_their_steps():
    # Up to sheeprl 0.8.0 the buffers didn't count the steps added: the steps of a full buffer are counted from the
    # position of the next one (only its remainder by the buffer size matters)
    rb = ReplayBuffer(5, 1, obs_keys=("value",))
    add_steps(rb, 0, 7)
    state = {k: v for k, v in rb.__dict__.items() if k not in ("_added", "_online_origin", "_online_next")}
    old = ReplayBuffer.__new__(ReplayBuffer)
    old.__setstate__(state)
    assert old._added == 7
    add_steps(old, 7, 1)
    assert old.sample(2, online=True)["value"][0, 0, 0] == 70
    rb = EpisodeBuffer(100, 3, obs_keys=("value",))
    episode(range(8), rb)
    state = {k: v for k, v in rb.__dict__.items() if k not in ("_stored", "_online_episode", "_online_sequence")}
    old = EpisodeBuffer.__new__(EpisodeBuffer)
    old.__setstate__(state)
    episode(range(10, 14), old)
    assert old.sample(1, sequence_length=3, online=True)["value"][0, :, 0, 0].tolist() == [11, 12, 13]
