import numpy as np
import pytest

from sheeprl.data.buffers_np import Trajectory, TrajectoryReplayBuffer


@pytest.fixture()
def trajectory():
    return Trajectory({"obs": np.random.random((5, 3)), "rew": np.random.random((5, 1))})


@pytest.fixture()
def trajectory2():
    return Trajectory({"obs": np.random.random((7, 3)), "rew": np.random.random((7, 1))})


def test_add_trajectory(trajectory):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    buffer.add(trajectory)
    assert len(buffer) == 1


def test_add_multiple_trajectories(trajectory, trajectory2):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    buffer.add(trajectory)
    buffer.add(trajectory2)
    assert len(buffer) == 2


def test_add_none_trajectory(trajectory):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    buffer.add(trajectory)
    buffer.add(None)
    assert len(buffer) == 1


def test_overflow_trajectories(trajectory, trajectory2):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=1)
    buffer.add(trajectory)
    buffer.add(trajectory2)
    assert len(buffer) == 1
    assert len(buffer[0]) == 7


def test_sample_trajectory(trajectory, trajectory2):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=1)
    buffer.add(trajectory)
    buffer.add(trajectory2)
    sample = buffer.sample(batch_size=1, sequence_length=4)
    assert len(sample) == 1
    assert sample["obs"].shape == (1, 4, 3)
    assert sample["rew"].shape == (1, 4, 1)


def test_sample_multiple_trajectories(trajectory, trajectory2):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    buffer.add(trajectory)
    buffer.add(trajectory2)
    buffer.add(trajectory)
    sample = buffer.sample(batch_size=2, sequence_length=4)
    assert len(sample) == 2
    assert sample["obs"].shape == (2, 4, 3)
    assert sample["rew"].shape == (2, 4, 1)


def test_with_shorter_than_sequence_length_trajectory(trajectory):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    short_traj = Trajectory({"obs": np.random.random((3, 3)), "rew": np.random.random((3, 1))})
    buffer.add(short_traj)
    buffer.add(trajectory)
    sample = buffer.sample(batch_size=2, sequence_length=4)
    assert len(sample) == 2
    assert sample["obs"].shape == (2, 4, 3)


def test_with_no_long_enough_trajectory():
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    short_traj = Trajectory({"obs": np.random.random((3, 3)), "rew": np.random.random((3, 1))})
    buffer.add(short_traj)
    with pytest.raises(RuntimeError):
        buffer.sample(batch_size=2, sequence_length=4)
