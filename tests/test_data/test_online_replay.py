"""The online queue of the replay buffers (`sample(online=True)`, `buffer.online`): the samples start with the steps
added since the previous ones, the oldest first, and only the rest of them is sampled uniformly."""

import pickle

import numpy as np
import pytest

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import SequenceSampler
from sheeprl.data.store import ReplayStore
from tests.test_data.sampling import draw


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
    rb = ReplayBuffer(10, 2, obs_keys=("value",))
    add_steps(rb, 0, 3)
    # The transitions of the steps 0 and 1 of both environments, in the order they were added
    value = draw(rb, 2, n_samples=2, online=True, seed=0)["value"][..., 0]
    assert value.tolist() == [[0, 1], [10, 11]]
    # Then the ones of step 2, and the rest of the batch is sampled uniformly
    value = draw(rb, 4, online=True)["value"][0, :, 0]
    assert value[:2].tolist() == [20, 21]
    add_steps(rb, 3, 1)
    assert draw(rb, 4, online=True)["value"][0, :2, 0].tolist() == [30, 31]
    # Without the online queue, a batch is sampled uniformly: the queue is left as it is
    add_steps(rb, 4, 1)
    draw(rb, 4)
    assert draw(rb, 4, online=True)["value"][0, :2, 0].tolist() == [40, 41]


def test_a_transition_joins_the_online_queue_with_its_next_observation():
    rb = ReplayBuffer(10, 1, obs_keys=("value",))
    add_steps(rb, 0, 3)
    sample = draw(rb, 4, sample_next_obs=True, online=True, seed=0)
    assert sample["value"][0, :2, 0].tolist() == [0, 10] and sample["next_value"][0, :2, 0].tolist() == [10, 20]
    assert 20 not in sample["value"]
    add_steps(rb, 3, 1)
    sample = draw(rb, 4, sample_next_obs=True, online=True)
    assert sample["value"][0, 0, 0] == 20 and sample["next_value"][0, 0, 0] == 30


def test_the_online_queue_drops_the_overwritten_steps():
    rb = ReplayBuffer(4, 1, obs_keys=("value",))
    add_steps(rb, 0, 6)
    # The buffer holds the steps from 2 on
    assert draw(rb, 3, online=True, seed=0)["value"][0, :, 0].tolist() == [20, 30, 40]
    assert draw(rb, 3, online=True)["value"][0, 0, 0] == 50


def test_the_sequences_of_the_online_queue_follow_each_other():
    rb = ReplayBuffer(20, 2)
    add_steps(rb, 0, 10)
    # As in DreamerV3, the sequences follow each other from the second step: 1-3, 4-6 and 7-9 of every environment
    value = draw(rb, 2, n_samples=2, sequence_length=3, online=True, seed=0)["value"][..., 0]
    assert value.shape == (2, 3, 2)
    assert value[0].T.tolist() == [[10, 20, 30], [11, 21, 31]]
    assert value[1].T.tolist() == [[40, 50, 60], [41, 51, 61]]
    value = draw(rb, 3, sequence_length=3, online=True)["value"][0, :, :, 0]
    assert value[:, :2].T.tolist() == [[70, 80, 90], [71, 81, 91]]
    # Sequences of one step start from the first one
    rb = ReplayBuffer(20, 1)
    add_steps(rb, 0, 2)
    assert draw(rb, 2, sequence_length=1, online=True, seed=0)["value"][0, 0, :, 0].tolist() == [0, 10]


@pytest.mark.parametrize("sequence_length", [None, 3])
def test_the_online_queues_of_the_environments_are_taken_oldest_first(sequence_length):
    # The environment 1 has more steps than the environment 0: the sequences that start first come first, the ones of
    # the environment 0 before the ones of the environment 1 at the same step
    rb = ReplayBuffer(20, 2)
    if sequence_length is None:
        add_steps(rb, 0, 2)
        add_steps(rb, 2, 1, envs=[1])
        assert draw(rb, 5, online=True, seed=0)["value"][0, :, 0].tolist() == [0, 1, 10, 11, 21]
    else:
        add_steps(rb, 0, 7)
        add_steps(rb, 7, 3, envs=[1])
        value = draw(rb, 5, sequence_length=3, online=True, seed=0)["value"][0, :, :, 0]
        expected = [[10, 20, 30], [11, 21, 31], [40, 50, 60], [41, 51, 61], [71, 81, 91]]
        assert value.T.tolist() == expected
    # The queues are empty: the batches are sampled uniformly
    kwargs = {"sequence_length": sequence_length}
    assert draw(rb, 5, online=True, **kwargs)["value"].shape == draw(rb, 5, **kwargs)["value"].shape


def test_a_failed_sample_leaves_the_online_queue_as_it_was(monkeypatch):
    # The queued transitions leave the queue once the uniform ones are drawn: a sample whose draw fails keeps them
    from sheeprl.data.samplers import TransitionSampler

    rb = ReplayBuffer(10, 2)
    add_steps(rb, 0, 3, envs=[0])

    def failing_draw(self, storage, n):
        raise ValueError("The draw failed")

    with monkeypatch.context() as patch:
        patch.setattr(TransitionSampler, "draw", failing_draw)
        with pytest.raises(ValueError, match="The draw failed"):
            draw(rb, 8, online=True, seed=0)
    add_steps(rb, 0, 1, envs=[1])
    assert draw(rb, 4, online=True)["value"][0, :, 0].tolist() == [0, 1, 10, 20]


def test_an_environment_without_steps_is_not_sampled():
    # The environments are drawn among the ones with steps: the environment 1, which has none, doesn't fail the sample
    rb = ReplayBuffer(10, 2)
    add_steps(rb, 0, 3, envs=[0])
    assert set(draw(rb, 16, seed=0)["value"][0, :, 0].tolist()) <= {0, 10, 20}


def test_the_online_queue_restarts_with_the_steps_added_after_a_checkpoint():
    store = ReplayStore(ReplayBuffer(20, 1), SequenceSampler(3, online=True, seed=0))
    add_steps(store.storage, 0, 7)
    saved = pickle.loads(pickle.dumps(store))
    store = ReplayStore(ReplayBuffer(20, 1), SequenceSampler(3, online=True, seed=1)).load(saved)
    # The sequences of the steps before the checkpoint are left out: the ones from the second step after it are queued
    add_steps(store.storage, 7, 4)
    assert store.sample(2, numpy_keys=("value",))["value"][0, :, 0, 0].tolist() == [80, 90, 100]
