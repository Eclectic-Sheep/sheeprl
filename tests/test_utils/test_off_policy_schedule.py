"""When an off-policy run plays random actions and trains, also when it resumes (`off_policy_schedule`)."""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

from sheeprl.utils.utils import Ratio, dotdict, off_policy_schedule

# 2 environments, 1 process
POLICY_STEPS_PER_ITER = 2


def schedule(
    iterations: Iterable[int],
    state: Optional[Dict[str, Any]] = None,
    learning_starts: int = 10,
    buffer_checkpoint: bool = True,
    pretrain_steps: int = 0,
    sequence_length: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[List[int], List[int], Ratio]:
    """The iterations that play random actions and the gradient steps of every iteration, computed as the training
    loops do, of a run that starts from the first of `iterations` (resuming from `state` if given)."""
    cfg = dotdict(
        {
            "dry_run": dry_run,
            "algo": {
                "learning_starts": learning_starts,
                "replay_ratio": 1.0,
                "per_rank_pretrain_steps": pretrain_steps,
                # Read only by the algorithms that sample sequences (Dreamer)
                "per_rank_sequence_length": sequence_length,
            },
            "buffer": {"checkpoint": buffer_checkpoint},
            "checkpoint": {"resume_from": "checkpoint.ckpt" if state is not None else None},
        }
    )
    iterations = list(iterations)
    learning_starts, train_starts, pretrain, total_iters, ratio = off_policy_schedule(
        cfg, state, iterations[0], iterations[-1] if not dry_run else 1, POLICY_STEPS_PER_ITER, world_size=1
    )
    # A dry run lasts until its first training
    iterations = range(iterations[0], total_iters + 1)
    random = [i for i in iterations if i <= learning_starts]
    # The replay ratio counts the policy steps from the start of the first iteration that trains, which also pretrains
    steps = [
        (
            ratio((i - (train_starts - 1)) * POLICY_STEPS_PER_ITER) + (pretrain if i == train_starts else 0)
            if i >= train_starts
            else 0
        )
        for i in iterations
    ]
    return random, steps, ratio


def test_a_new_run_plays_random_actions_then_trains():
    random, steps, _ = schedule(range(1, 9))
    # `learning_starts=10`: 5 iterations of random actions, training from the last of them
    assert random == [1, 2, 3, 4, 5]
    assert steps == [0, 0, 0, 0, 2, 2, 2, 2]


def test_a_new_run_without_random_actions_trains_from_the_first_iteration():
    random, steps, _ = schedule(range(1, 4), learning_starts=0)
    assert random == [] and steps == [2, 2, 2]


@pytest.mark.parametrize("checkpoint_iter", [3, 8])
def test_a_resumed_run_with_its_buffer_continues_as_the_run_it_resumes(checkpoint_iter):
    # Resumed during the random actions (after iteration 3) or after them (after iteration 8)
    full_random, full_steps, _ = schedule(range(1, 16))
    first_random, first_steps, ratio = schedule(range(1, checkpoint_iter + 1))
    random, steps, _ = schedule(range(checkpoint_iter + 1, 16), state={"ratio": ratio.state_dict()})
    assert first_random + random == full_random
    assert first_steps + steps == full_steps


def test_a_resumed_run_without_its_buffer_fills_a_new_one_with_its_policy():
    _, _, ratio = schedule(range(1, 9), buffer_checkpoint=False)
    random, steps, _ = schedule(range(9, 16), state={"ratio": ratio.state_dict()}, buffer_checkpoint=False)
    # No random actions: the policy fills the new buffer for `learning_starts` steps (5 iterations), then the training
    # starts as in a new run
    assert random == []
    assert steps == [0, 0, 0, 0, 2, 2, 2]


def test_the_ratio_of_a_checkpoint_counting_from_another_start_is_realigned():
    # A run resumed by an older version counted the steps of its ratio from where it resumed (here 4 iterations later):
    # its checkpoint, resumed, would do all the gradient steps of those iterations at once
    _, full_steps, ratio = schedule(range(1, 9))
    state = ratio.state_dict()
    state["_prev"] -= 4 * POLICY_STEPS_PER_ITER
    with pytest.warns(UserWarning, match="instead of doing 8 gradient steps at once"):
        _, steps, _ = schedule(range(9, 12), state={"ratio": state})
    assert steps == [2, 2, 2]


@pytest.mark.parametrize("replay_ratio", [0.25, 0.5, 1.0, 3.0])
def test_a_consistent_ratio_is_not_realigned(replay_ratio):
    ratio = Ratio(replay_ratio)
    for step in range(1, 20, 3):
        ratio(step)
    state = ratio.state_dict()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ratio.realign(19)
    assert ratio.state_dict() == state


def test_the_first_training_pretrains_also_after_a_resume_that_refills_the_buffer():
    random, steps, ratio = schedule(range(1, 9), pretrain_steps=100)
    assert steps == [0, 0, 0, 0, 102, 2, 2, 2]
    # Resumed with its buffer, it doesn't pretrain again
    _, resumed, _ = schedule(range(9, 12), state={"ratio": ratio.state_dict()}, pretrain_steps=100)
    assert resumed == [2, 2, 2]
    # Without it, it trains as a new run once it has filled a new one: it pretrains on it
    _, resumed, _ = schedule(
        range(9, 16), state={"ratio": ratio.state_dict()}, pretrain_steps=100, buffer_checkpoint=False
    )
    assert resumed == [0, 0, 0, 0, 102, 2, 2]


def test_the_ratio_of_a_checkpoint_with_its_old_pretraining_steps_is_loaded():
    ratio = Ratio(0.5)
    ratio.load_state_dict({"_ratio": 0.5, "_prev": 10.0, "_pretrain_steps": 100})
    assert ratio.state_dict() == {"_ratio": 0.5, "_prev": 10.0}


def test_the_training_waits_for_a_sequence_of_every_environment():
    # The training started after `learning_starts` policy steps also when the environments had played fewer steps than
    # the sequences it samples: the sampling crashed. `learning_starts=4` gives 2 iterations of random actions
    with pytest.warns(UserWarning, match="sequences of `algo.per_rank_sequence_length=5`"):
        random, steps, _ = schedule(range(1, 8), learning_starts=4, sequence_length=5)
    assert random == [1, 2]
    assert steps == [0, 0, 0, 0, 2, 2, 2]
    # When `learning_starts` comes later, nothing changes
    _, steps, _ = schedule(range(1, 7), learning_starts=10, sequence_length=5)
    assert steps == [0, 0, 0, 0, 2, 2]


def test_a_dry_run_plays_until_its_first_training():
    # A dry run played one iteration: with sequences longer than one step, the sampling crashed
    random, steps, _ = schedule(range(1, 2), dry_run=True, sequence_length=3)
    assert random == [] and steps == [0, 0, 2]
