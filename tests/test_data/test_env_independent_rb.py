import numpy as np
import pytest

from sheeprl.data.buffers import EnvIndependentReplayBuffer, ReplayBuffer, SequentialReplayBuffer


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


def test_env_independent_sample_seed():
    data = {"a": np.arange(80, dtype=np.float32).reshape(20, 4, 1)}
    samples = []
    for seed in (42, 42, 0):
        rb = EnvIndependentReplayBuffer(20, 4, buffer_cls=SequentialReplayBuffer, seed=seed)
        rb.add(data)
        samples.append(rb.sample(8, n_samples=2, sequence_length=3)["a"])
    np.testing.assert_array_equal(samples[0], samples[1])
    assert not np.array_equal(samples[0], samples[2])


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


def test_env_independent_batches_draw_their_environments_independently():
    # The batches of one call took the same number of elements from each environment
    n_envs, batch_size, n_samples = 4, 12, 50
    rb = EnvIndependentReplayBuffer(20, n_envs, buffer_cls=SequentialReplayBuffer, seed=0)
    rb.add({"env": np.tile(np.arange(n_envs, dtype=np.float32).reshape(1, n_envs, 1), (20, 1, 1))})
    envs = rb.sample(batch_size, n_samples=n_samples, sequence_length=3)["env"][:, 0, :, 0]  # [N_samples, Batch_size]
    counts = np.stack([np.bincount(batch.astype(int), minlength=n_envs) for batch in envs])
    assert len(np.unique(counts, axis=0)) > 1
    # Every environment is drawn with the same probability
    np.testing.assert_allclose(counts.sum(0) / counts.sum(), 1 / n_envs, atol=0.05)


@pytest.mark.parametrize("buffer_cls", [ReplayBuffer, SequentialReplayBuffer])
def test_env_independent_samples_come_whole_from_one_environment(buffer_cls):
    # Every element (every sequence) of a batch is read from a single environment, at consecutive steps
    n_envs, size = 3, 30
    rb = EnvIndependentReplayBuffer(size, n_envs, buffer_cls=buffer_cls, seed=1)
    step = np.arange(size, dtype=np.float32).reshape(size, 1, 1)
    env = np.arange(n_envs, dtype=np.float32).reshape(1, n_envs, 1)
    rb.add({"value": np.broadcast_to(env * 1000 + step, (size, n_envs, 1)).copy()})
    kwargs = {"sequence_length": 4} if buffer_cls is SequentialReplayBuffer else {}
    value = rb.sample(7, n_samples=5, **kwargs)["value"][..., 0]
    if buffer_cls is SequentialReplayBuffer:
        assert value.shape == (5, 4, 7)
        envs, steps = value // 1000, value % 1000
        assert (envs == envs[:, :1]).all()
        assert (np.diff(steps, axis=1) == 1).all()
    else:
        assert value.shape == (5, 7)
        assert set(np.unique(value // 1000)) <= set(range(n_envs))


def test_a_failed_add_leaves_every_environment_as_it_was():
    # With `validate_args`, the key 'b' of the environment 1 fails the add before the environment 0, empty, is written
    rb = EnvIndependentReplayBuffer(5, 2)
    rb.add({"a": np.zeros((1, 1, 1))}, [1])
    with pytest.raises(KeyError, match="The buffer has no key 'b'"):
        rb.add({"a": np.ones((1, 2, 1)), "b": np.ones((1, 2, 1))}, validate_args=True)
    assert rb.buffer[0].empty
    assert rb.buffer[1]._pos == 1 and rb.buffer[1]["a"][0, 0, 0] == 0
