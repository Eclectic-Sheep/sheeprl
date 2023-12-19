import numpy as np
import pytest

from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer


def test_env_idependent_wrong_buffer_size():
    with pytest.raises(ValueError):
        EnvIndependentReplayBuffer(-1)


def test_env_idependent_wrong_n_envs():
    with pytest.raises(ValueError):
        EnvIndependentReplayBuffer(1, -1)


def test_env_independent_missing_memmap_dir():
    with pytest.raises(ValueError):
        EnvIndependentReplayBuffer(10, 4, memmap=True, memmap_dir=None)


def test_env_independent_wrong_memmap_mode():
    with pytest.raises(ValueError):
        EnvIndependentReplayBuffer(10, 4, memmap=True, memmap_mode="a+")


def test_env_independent_add():
    bs = 20
    n_envs = 4
    rb = EnvIndependentReplayBuffer(bs, n_envs)
    stps1 = {"dones": np.zeros((10, 4, 1))}
    rb.add(stps1)
    for i in range(n_envs):
        assert rb._buf[i]._pos == 10
    stps2 = {"dones": np.zeros((10, 2, 1))}
    rb.add(stps2, [0, 3])
    assert rb._buf[0]._pos == 0
    assert rb._buf[1]._pos == 10
    assert rb._buf[2]._pos == 10
    assert rb._buf[0]._pos == 0


def test_env_independent_add_error():
    bs = 10
    n_envs = 4
    rb = EnvIndependentReplayBuffer(bs, n_envs)
    stps = {"dones": np.zeros((10, 3, 1))}
    with pytest.raises(ValueError):
        rb.add(stps)


def test_env_independent_sample_shape():
    bs = 20
    n_envs = 4
    rb = EnvIndependentReplayBuffer(bs, n_envs)
    stps1 = {"dones": np.ones((10, 4, 1))}
    rb.add(stps1)
    stps2 = {"dones": np.ones((10, 2, 1))}
    rb.add(stps2, [0, 3])
    sample = rb.sample(10, n_samples=10)
    assert sample["dones"].shape == tuple([10, 10, 1])


def test_env_independent_sample():
    bs = 20
    n_envs = 4
    rb = EnvIndependentReplayBuffer(bs, n_envs)
    stps1 = {"dones": np.ones((10, 4, 1))}
    for i in range(n_envs):
        stps1["dones"][:, i] *= i
    rb.add(stps1)
    stps2 = {"dones": np.ones((10, 2, 1))}
    for i, env in enumerate([0, 3]):
        stps2["dones"][:, i] *= env
    rb.add(stps2, [0, 3])
    sample = rb.sample(2000, n_samples=2)
    for i in range(n_envs):
        assert (sample["dones"] == i).any()


def test_env_independent_sample_error():
    bs = 20
    n_envs = 4
    rb = EnvIndependentReplayBuffer(bs, n_envs)
    with pytest.raises(ValueError, match="No sample has been added to the buffer"):
        rb.sample(10, n_samples=10)
    stps1 = {"dones": np.zeros((10, 4, 1))}
    rb.add(stps1)
    stps2 = {"dones": np.zeros((10, 2, 1))}
    rb.add(stps2, [0, 3])

    with pytest.raises(ValueError, match="must be both greater than 0"):
        rb.sample(0, n_samples=10)
        rb.sample(10, n_samples=0)
        rb.sample(-1, n_samples=10)
        rb.sample(10, n_samples=-1)


def test_env_independent_sample_tensors():
    import torch

    bs = 20
    n_envs = 4
    rb = EnvIndependentReplayBuffer(bs, n_envs, buffer_cls=SequentialReplayBuffer)
    with pytest.raises(ValueError, match="No sample has been added to the buffer"):
        rb.sample(10, n_samples=10)
    stps1 = {"dones": np.zeros((10, 4, 1))}
    rb.add(stps1)
    stps2 = {"dones": np.zeros((10, 2, 1))}
    rb.add(stps2, [0, 3])

    s = rb.sample_tensors(10, n_samples=3, sequence_length=5)
    assert isinstance(s["dones"], torch.Tensor)
    assert s["dones"].shape == torch.Size([3, 5, 10, 1])
