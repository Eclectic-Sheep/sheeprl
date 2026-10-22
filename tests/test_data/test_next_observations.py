"""Adjacent-row replay keeps actual next observations at timeouts and checkpoint boundaries."""

import pickle

import numpy as np
import pytest

from sheeprl.data.buffers import ReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.memmap import copied_on_pickle


@pytest.mark.parametrize("memmap", [False, True])
@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("online", [False, True])
def test_next_observations_survive_wraparound_and_independent_timeouts(tmp_path, memmap, sequential, online):
    cls = SequentialReplayBuffer if sequential else ReplayBuffer
    rb = cls(5, 2, obs_keys=("state", "rgb"), memmap=memmap, memmap_dir=tmp_path, seed=0)
    ids = np.arange(15 * 2).reshape(15, 2, 1)
    truncated = (ids % 3 == 1).astype(np.uint8)
    state = ids.astype(np.float32)
    next_state = state + 2 + truncated * 100
    rgb = np.broadcast_to(ids[..., None, None], (15, 2, 1, 2, 2)).astype(np.uint8).copy()
    next_rgb = rgb + 2 + truncated[..., None, None] * 100
    # Includes a batch larger than the capacity, crossing the insertion position more than once.
    start = 0
    for length in (1, 2, 11, 1):
        end = start + length
        rb.add(
            {"state": state[start:end], "rgb": rgb[start:end], "truncated": truncated[start:end]},
            next_obs={"state": next_state[start:end], "rgb": next_rgb[start:end]},
            validate_args=True,
        )
        start = end
    assert all(f"next_{key}" not in rb.buffer for key in rb._obs_keys)
    # Only the retained timeouts and the latest step are stored, not all next observations.
    expected = {((t % 5), env) for t in range(10, 15) for env in range(2) if truncated[t, env, 0] or t == 14}
    assert {(row, env) for row, envs in rb._next_obs.items() for env in envs} == expected
    kwargs = {"sequence_length": 2} if sequential else {}
    samples = rb.sample(100, n_samples=2, sample_next_obs=True, online=online, **kwargs)
    ids_sampled = samples["state"][..., 0].astype(int)
    t, env = ids_sampled // 2, ids_sampled % 2
    np.testing.assert_array_equal(samples["next_state"], next_state[t, env])
    np.testing.assert_array_equal(samples["next_rgb"], next_rgb[t, env])
    assert samples["next_state"].dtype == np.float32
    assert samples["next_rgb"].dtype == np.uint8


@pytest.mark.parametrize("memmap,mode", [(False, "reference"), (True, "reference"), (True, "copy")])
def test_checkpoint_resume_keeps_the_latest_next_observations(tmp_path, memmap, mode):
    rb = ReplayBuffer(4, 2, memmap=memmap, memmap_dir=tmp_path, seed=0)
    obs = np.broadcast_to(np.arange(2, dtype=np.float32)[:, None, None], (2, 2, 1)).copy()
    rb.add({"observations": obs, "truncated": np.zeros((2, 2, 1))}, next_obs={"observations": obs + 1})
    callback = CheckpointCallback(memmap_buffer=mode)
    state = callback._ckpt_rb(rb)
    with copied_on_pickle(mode == "copy"):
        checkpoint = pickle.dumps(rb)
    callback._experiment_consistent_rb(rb, state)
    assert not rb["truncated"][:2].any()
    loaded = pickle.loads(checkpoint)
    # The new process resets the environments. The preceding step must still bootstrap from 2, not this reset 0.
    loaded.add(
        {"observations": np.zeros((1, 2, 1), dtype=np.float32), "truncated": np.zeros((1, 2, 1))},
        next_obs={"observations": np.ones((1, 2, 1), dtype=np.float32)},
    )
    samples = loaded.sample_tensors(100, sample_next_obs=True)
    boundary = samples["truncated"].bool()
    assert boundary.any()
    assert (samples["next_observations"][boundary] == 2).all()
    assert (samples["next_observations"][~boundary] == 1).all()
    assert set(loaded._next_obs) == {1, 2}


def test_boundary_observations_own_their_data_and_expire_when_overwritten():
    rb = ReplayBuffer(3, 2, seed=0)
    final = np.array([[[42], [10]]], dtype=np.float32)
    rb.add(
        {"observations": np.zeros((1, 2, 1)), "truncated": np.array([[[1], [0]]])},
        next_obs={"observations": final},
    )
    final[:] = -1
    rb.add({"observations": np.ones((1, 2, 1)), "truncated": np.zeros((1, 2, 1))})
    assert set(rb._next_obs[0]) == {0}
    sample = rb.sample(100, sample_next_obs=True)
    timeout = sample["truncated"].astype(bool)
    assert timeout.any()
    assert (sample["next_observations"][timeout] == 42).all()
    sample["next_observations"][:] = -1
    assert rb._next_obs[0][0]["observations"].item() == 42
    rb.add({"observations": np.full((3, 2, 1), 8), "truncated": np.zeros((3, 2, 1))})
    assert not rb._next_obs
    assert (rb.sample(100, sample_next_obs=True)["next_observations"] == 8).all()


def test_invalid_next_observations_do_not_change_the_buffer():
    rb = ReplayBuffer(3, seed=0)
    data = {"observations": np.ones((1, 1, 2)), "truncated": np.zeros((1, 1, 1))}
    rb.add(data, next_obs={"observations": np.full((1, 1, 2), 2)})
    with pytest.raises(ValueError, match="next observations"):
        rb.add(data, next_obs={"observations": np.zeros((1, 2, 2))}, validate_args=True)
    assert rb._pos == 1
    assert (rb._next_obs[0][0]["observations"] == 2).all()
    with pytest.raises(KeyError):
        rb.add(data, next_obs={}, validate_args=True)
    assert rb._pos == 1
    # An empty add does not discard the latest observation needed by a checkpoint.
    rb.add({k: v[:0] for k, v in data.items()})
    assert (rb._next_obs[0][0]["observations"] == 2).all()


def test_loading_a_checkpoint_from_before_next_observation_overrides():
    rb = ReplayBuffer(3, seed=0)
    rb.add({"observations": np.zeros((2, 1, 1)), "truncated": np.zeros((2, 1, 1))})
    del rb._next_obs
    loaded = pickle.loads(pickle.dumps(rb))
    assert loaded._next_obs == {}
    loaded.add(
        {"observations": np.ones((1, 1, 1)), "truncated": np.ones((1, 1, 1))},
        next_obs={"observations": np.full((1, 1, 1), 2)},
    )
    assert loaded._next_obs[2][0]["observations"].item() == 2


def test_adding_another_buffer_transfers_boundary_observations():
    source = ReplayBuffer(4, seed=0)
    source.add(
        {"observations": np.arange(6).reshape(6, 1, 1), "truncated": np.ones((6, 1, 1))},
        next_obs={"observations": np.arange(100, 106).reshape(6, 1, 1)},
    )
    target = ReplayBuffer(3, seed=0)
    target.add(source)
    samples = target.sample(100, sample_next_obs=True)
    np.testing.assert_array_equal(samples["next_observations"], samples["observations"] + 100)
    # The copy is independent of the source buffer and its metadata.
    for envs in source._next_obs.values():
        for obs in envs.values():
            obs["observations"][:] = -1
    samples = target.sample(100, sample_next_obs=True)
    np.testing.assert_array_equal(samples["next_observations"], samples["observations"] + 100)


def test_broadcast_episode_flags_with_explicit_next_observations():
    rb = ReplayBuffer(3, 2, seed=0)
    rb.add(
        {"observations": np.zeros((2, 1, 1)), "truncated": np.ones((2, 1, 1))},
        next_obs={"observations": np.full((2, 2, 1), 42)},
        validate_args=True,
    )
    assert (rb.sample(100, sample_next_obs=True)["next_observations"] == 42).all()
