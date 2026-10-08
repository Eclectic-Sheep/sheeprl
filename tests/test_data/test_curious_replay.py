"""Curious Replay (`CuriousSequenceSampler`, `buffer.curious`): the sequences are drawn in proportion to the priority of
their last step, `c * beta ** visits + (|loss| + epsilon) ** alpha`, which the training writes from the losses of the
world model on every batch."""

import pickle

import numpy as np
import pytest
import torch

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import SAMPLED_STEPS, CuriousSequenceSampler, SequenceSampler, SumTree
from sheeprl.data.store import ReplayStore, curious_replay
from sheeprl.utils.utils import dotdict

C, BETA, ALPHA, EPSILON, INITIAL = 1e4, 0.7, 0.7, 0.01, 1e5


def priority(visits: int, loss: float) -> float:
    return C * BETA**visits + (abs(loss) + EPSILON) ** ALPHA


def add_steps(rb: ReplayBuffer, counters: np.ndarray, n: int, envs=None) -> None:
    """Add `n` steps to the environments `envs` (all by default), with their numbers in the column `step`: the
    environment and the steps added to it before."""
    envs = np.arange(rb.n_envs) if envs is None else np.asarray(envs)
    for _ in range(n):
        data = {"step": np.stack((envs, counters[envs]), -1)[np.newaxis]}
        rb.add(data, None if len(envs) == rb.n_envs else envs)
        counters[envs] += 1


def priorities(sampler: CuriousSequenceSampler, rb: ReplayBuffer) -> np.ndarray:
    """The priorities of the rows of the buffer, of shape `[buffer_size, n_envs]`."""
    return sampler.tree[np.arange(rb.buffer_size * rb.n_envs)].reshape(rb.buffer_size, rb.n_envs)


def test_the_sum_tree_finds_the_elements_by_their_cumulative_priorities():
    tree = SumTree(5)
    tree.update([0, 2, 3], [1.0, 2.0, 3.0])
    assert tree.total == 6
    assert tree.find([0, 0.5, 1, 2.9, 3, 5.9]).tolist() == [0, 0, 2, 2, 3, 3]
    # A value at the total, as the rounding can give, finds the last element with a priority, never an empty one
    assert tree.find([6.0]).tolist() == [3]


def test_the_sum_tree_keeps_the_last_priority_of_a_repeated_element():
    tree = SumTree(4)
    tree.update([1, 1, 2], [5.0, 2.0, 1.0])
    assert tree[[0, 1, 2, 3]].tolist() == [0, 2, 1, 0] and tree.total == 3


def test_every_node_of_the_sum_tree_is_the_sum_of_its_children():
    rng = np.random.default_rng(0)
    tree = SumTree(37)
    for _ in range(20):
        tree.update(rng.integers(0, 37, 10), rng.random(10))
    inner = np.arange(1, tree.first_leaf)
    assert np.array_equal(tree.nodes[inner], tree.nodes[2 * inner] + tree.nodes[2 * inner + 1])
    assert np.all(tree.nodes[tree.first_leaf + 37 :] == 0)


def test_the_new_steps_have_the_initial_priority_when_they_end_a_sequence():
    # Sequences of 3 steps in a buffer of 6 rows: a step ends a sequence when the 2 steps before it are in the buffer
    rb, counters = ReplayBuffer(6, n_envs=2), np.zeros(2, np.int64)
    sampler = CuriousSequenceSampler(3, seed=0)
    add_steps(rb, counters, 4)
    sampler.sample(rb, batch_size=2)
    assert priorities(sampler, rb).T.tolist() == [[0, 0, INITIAL, INITIAL, 0, 0]] * 2
    # The buffer goes around: it holds the steps 2 to 7 in the rows 2, 3, 4, 5, 0, 1, and the steps 2 and 3 no longer
    # end a sequence (the steps before them were overwritten)
    add_steps(rb, counters, 4)
    sampler.sample(rb, batch_size=2)
    assert priorities(sampler, rb).T.tolist() == [[INITIAL, INITIAL, 0, 0, INITIAL, INITIAL]] * 2


@pytest.mark.parametrize("device", [None, "cpu"])
def test_the_sequences_are_consecutive_steps_in_the_buffer(device):
    # A sequence never crosses the row of the next step of its environment, nor reads the rows never written, also when
    # the environments are at different rows; the steps given with the samples are the ones of their sequences
    rb, counters = ReplayBuffer(7, n_envs=2, device=device), np.zeros(2, np.int64)
    sampler = CuriousSequenceSampler(3, seed=0)
    add_steps(rb, counters, 5)
    add_steps(rb, counters, 4, envs=[1])
    for _ in range(20):
        sample = sampler.sample(rb, batch_size=8)
        steps = np.asarray(sample["step"][0])
        assert np.array_equal(steps, sample[SAMPLED_STEPS][0])
        assert np.all(np.diff(steps[..., 1], axis=0) == 1) and np.all(steps[..., 0] == steps[:1, :, 0])
        assert np.all(steps[0, :, 1] >= np.maximum(counters[steps[0, :, 0]] - rb.buffer_size, 0))


def test_the_sequences_are_drawn_in_proportion_to_the_priorities_of_their_last_steps():
    rb, counters = ReplayBuffer(10, n_envs=1), np.zeros(1, np.int64)
    sampler = CuriousSequenceSampler(2, seed=0)
    add_steps(rb, counters, 4)
    sampler.sample(rb, batch_size=1)
    # The steps 1, 2 and 3 end a sequence
    sampler.tree.update([1, 2, 3], [1.0, 2.0, 7.0])
    last = sampler.sample(rb, batch_size=10_000)[SAMPLED_STEPS][0, -1, :, 1]
    np.testing.assert_allclose(np.bincount(last, minlength=4) / len(last), [0, 0.1, 0.2, 0.7], atol=0.02)


def test_a_trained_step_gets_the_priority_of_its_visits_and_of_its_loss():
    rb, counters = ReplayBuffer(10, n_envs=2), np.zeros(2, np.int64)
    sampler = CuriousSequenceSampler(2, seed=0)
    add_steps(rb, counters, 5)
    # A batch of one sequence, the steps 2 and 3 of the environment 1: `[T, B, 2]`
    steps = np.array([[[1, 2]], [[1, 3]]])
    sampler.update(rb, steps, np.array([[3.0], [-5.0]]))
    assert sampler.visits[:, 1].tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert priorities(sampler, rb)[2:4, 1] == pytest.approx([priority(1, 3.0), priority(1, -5.0)])
    sampler.update(rb, steps, np.array([[1.0], [2.0]]))
    assert priorities(sampler, rb)[2:4, 1] == pytest.approx([priority(2, 1.0), priority(2, 2.0)])
    # The other steps keep the initial priority
    assert priorities(sampler, rb)[[1, 4], 1].tolist() == [INITIAL, INITIAL]
    assert priorities(sampler, rb)[:5, 0].tolist() == [0, INITIAL, INITIAL, INITIAL, INITIAL]


def test_a_step_in_two_sequences_of_a_batch_is_visited_once_with_the_loss_of_the_last_one():
    rb, counters = ReplayBuffer(10, n_envs=1), np.zeros(1, np.int64)
    sampler = CuriousSequenceSampler(2, seed=0)
    add_steps(rb, counters, 5)
    # The sequences of the steps 1-2 and 2-3: the step 2 is in both, with the losses 2 and 4
    steps = np.array([[[0, 1], [0, 2]], [[0, 2], [0, 3]]])
    sampler.update(rb, steps, np.array([[1.0, 4.0], [2.0, 8.0]]))
    assert sampler.visits[:5, 0].tolist() == [0, 1, 1, 1, 0]
    assert priorities(sampler, rb)[1:4, 0] == pytest.approx([priority(1, 1.0), priority(1, 4.0), priority(1, 8.0)])


def test_the_steps_overwritten_since_the_sample_are_skipped():
    rb, counters = ReplayBuffer(4, n_envs=1), np.zeros(1, np.int64)
    sampler = CuriousSequenceSampler(2, seed=0)
    add_steps(rb, counters, 4)
    sampler.sample(rb, batch_size=1)
    steps = np.array([[[0, 1]], [[0, 2]]])
    # The buffer holds the steps 4, 5, 2, 3 (rows 0 to 3): the step 1 was overwritten by the step 5, and the step 2 no
    # longer ends a sequence
    add_steps(rb, counters, 2)
    sampler.update(rb, steps, np.array([[1.0], [1.0]]))
    assert sampler.visits[:, 0].tolist() == [0, 0, 1, 0]
    assert priorities(sampler, rb)[:, 0].tolist() == [INITIAL, INITIAL, 0, INITIAL]


def test_the_online_queue_comes_before_the_priorities():
    rb, counters = ReplayBuffer(20, n_envs=1), np.zeros(1, np.int64)
    sampler = CuriousSequenceSampler(2, online=True, seed=0)
    add_steps(rb, counters, 7)
    # The queue cuts the steps into sequences from the second step: 1-2, 3-4 and 5-6, the oldest first
    first = sampler.sample(rb, batch_size=4)[SAMPLED_STEPS][0, 0, :, 1]
    assert first[:3].tolist() == [1, 3, 5]


def test_the_priorities_are_saved_with_the_store():
    rb, counters = ReplayBuffer(10, n_envs=1), np.zeros(1, np.int64)
    store = ReplayStore(rb, CuriousSequenceSampler(2, seed=0))
    add_steps(rb, counters, 5)
    sample = store.sample(2, numpy_keys=(SAMPLED_STEPS,))
    # The losses on the device of the training are read on the CPU
    store.update_priorities(sample[SAMPLED_STEPS][0], torch.ones(2, 2))
    assert store.sampler.visits.sum() > 0
    saved = pickle.loads(pickle.dumps(store))
    resumed = ReplayStore(ReplayBuffer(10, n_envs=1), CuriousSequenceSampler(2, seed=1)).load(saved)
    assert np.array_equal(resumed.sampler.tree.nodes, store.sampler.tree.nodes)
    assert np.array_equal(resumed.sampler.visits, store.sampler.visits)
    # The resumed run draws what the run would have drawn
    assert torch.equal(resumed.sample(4)["step"], store.sample(4)["step"])


def test_a_buffer_sampled_uniformly_starts_with_the_initial_priorities():
    rb, counters = ReplayBuffer(10, n_envs=1), np.zeros(1, np.int64)
    store = ReplayStore(rb, SequenceSampler(2, seed=0))
    add_steps(rb, counters, 5)
    store.sample(2)
    resumed = ReplayStore(ReplayBuffer(10, n_envs=1), CuriousSequenceSampler(2, seed=1))
    resumed.load(pickle.loads(pickle.dumps(store))).sample(1)
    assert priorities(resumed.sampler, resumed.storage)[:5, 0].tolist() == [0, INITIAL, INITIAL, INITIAL, INITIAL]


def test_an_empty_buffer_has_no_sequence_to_draw():
    rb, counters = ReplayBuffer(10, n_envs=2), np.zeros(2, np.int64)
    add_steps(rb, counters, 2)
    with pytest.raises(ValueError, match="no environment of the buffer holds as many steps"):
        CuriousSequenceSampler(3, seed=0).sample(rb, batch_size=1)


def test_curious_replay_is_enabled_only_for_the_algorithms_that_train_with_it():
    cfg = dotdict({"algo": {"name": "sac"}, "buffer": {"curious": {"enabled": True}}})
    with pytest.raises(ValueError, match="Curious Replay"):
        curious_replay(cfg, supported=False)
    assert curious_replay(cfg)["enabled"]
    # Disabled, or a configuration saved before Curious Replay existed
    assert curious_replay(dotdict({"buffer": {"curious": {"enabled": False}}}), supported=False) is None
    assert curious_replay(dotdict({"buffer": {}}), supported=False) is None
