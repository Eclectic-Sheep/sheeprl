import os
import shutil
import time

import numpy as np
import pytest

from sheeprl.data.buffers import SequentialReplayBuffer


def test_seq_replay_buffer_wrong_buffer_size():
    with pytest.raises(ValueError):
        SequentialReplayBuffer(-1)


def test_seq_replay_buffer_wrong_n_envs():
    with pytest.raises(ValueError):
        SequentialReplayBuffer(1, -1)


def test_seq_replay_buffer_add_tds():
    buf_size = 5
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(2, 1, 1)}
    td2 = {"a": np.random.rand(2, 1, 1)}
    td3 = {"a": np.random.rand(3, 1, 1)}
    rb.add(td1)
    rb.add(td2)
    rb.add(td3)
    assert rb.full
    assert rb["a"][0] == td3["a"][-2]
    assert rb["a"][1] == td3["a"][-1]
    np.testing.assert_allclose(rb["a"][2:4], td2["a"])


def test_seq_replay_buffer_add_single_td():
    buf_size = 5
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(6, 1, 1)}
    rb.add(td1)
    assert rb.full
    assert rb["a"][0] == td1["a"][-1]


def test_seq_replay_buffer_sample():
    buf_size = 10
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(11, 1, 1)}
    rb.add(td1)
    s = rb.sample(4, sequence_length=2)
    assert s["a"].shape == tuple([1, 2, 4, 1])


def test_seq_replay_buffer_sample_one_element():
    buf_size = 1
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(1, 1, 1)}
    rb.add(td1)
    sample = rb.sample(1, sequence_length=1)
    assert rb.full
    assert sample["a"] == td1["a"]
    with pytest.raises(ValueError):
        rb.sample(1, sequence_length=2)


def test_seq_replay_buffer_sample_shapes():
    buf_size = 30
    n_envs = 2
    rb = SequentialReplayBuffer(buf_size, n_envs, obs_keys=("a",))
    t = {"a": np.arange(60).reshape(-1, 2, 1) % buf_size}
    rb.add(t)
    sample = rb.sample(3, sequence_length=5, n_samples=2)
    assert sample["a"].shape == tuple([2, 5, 3, 1])
    sample = rb.sample(3, sequence_length=5, n_samples=2, sample_next_obs=True, clone=True)
    assert sample["a"].shape == tuple([2, 5, 3, 1])
    assert sample["next_a"].shape == tuple([2, 5, 3, 1])


def test_seq_replay_buffer_sample_full():
    buf_size = 10000
    n_envs = 1
    seq_len = 50
    rb = SequentialReplayBuffer(buf_size, n_envs)
    t = {"a": np.arange(10500).reshape(-1, 1, 1) % buf_size}
    rb.add(t)
    samples = rb.sample(1000, sequence_length=seq_len, n_samples=5)
    assert not np.logical_and((samples["a"][:, 0, :] < rb._pos), (samples["a"][:, -1, :] >= rb._pos)).any()


def test_seq_replay_buffer_sample_full_large_sl():
    buf_size = 10000
    n_envs = 1
    seq_len = 1000
    rb = SequentialReplayBuffer(buf_size, n_envs)
    t = {"a": np.arange(10500).reshape(-1, 1, 1) % buf_size}
    rb.add(t)
    samples = rb.sample(1000, sequence_length=seq_len, n_samples=5)
    assert not np.logical_and(
        (samples["a"][:, 0, :] >= buf_size + rb._pos - seq_len + 1), (samples["a"][:, -1, :] < rb._pos)
    ).any()
    assert not np.logical_and((samples["a"][:, 0, :] < rb._pos), (samples["a"][:, -1, :] >= rb._pos)).any()


def test_seq_replay_buffer_sample_fail_not_full():
    buf_size = 10
    n_envs = 1
    seq_len = 8
    rb = SequentialReplayBuffer(buf_size, n_envs)
    t = {"a": np.arange(5).reshape(-1, 1, 1)}
    rb.add(t)
    with pytest.raises(ValueError, match="Cannot sample a sequence of length"):
        rb.sample(5, sequence_length=seq_len, n_samples=1)


def test_seq_replay_buffer_sample_not_full():
    buf_size = 10
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    rb._buf = {"a": np.ones((10, n_envs, 1)) * 20}
    t = {"a": np.arange(7).reshape(-1, 1, 1) * 1.0}
    rb.add(t)
    sample = rb.sample(2, sequence_length=5, n_samples=2)
    assert (sample["a"] < 7).all()


def test_seq_replay_buffer_sample_no_add():
    buf_size = 10
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    with pytest.raises(ValueError, match="No sample has been added"):
        rb.sample(2, sequence_length=5, n_samples=2)


def test_seq_replay_buffer_sample_error():
    buf_size = 10
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    with pytest.raises(ValueError, match="must be both greater than "):
        rb.sample(-1, sequence_length=5, n_samples=2)
        rb.sample(2, sequence_length=-1, n_samples=2)


def test_sample_tensors():
    import torch

    buf_size = 5
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td = {"observations": np.arange(8).reshape(-1, 1, 1), "dones": np.zeros((8, 1, 1))}
    td["dones"][-1] = 1
    rb.add(td)
    s = rb.sample_tensors(10, sample_next_obs=True, n_samples=3, sequence_length=5)
    assert isinstance(s["observations"], torch.Tensor)
    assert s["observations"].shape == torch.Size([3, 5, 10, 1])


def test_sample_tensor_memmap():
    import torch

    buf_size = 10
    n_envs = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = SequentialReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir, obs_keys=("observations"))
    td = {
        "observations": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8),
        "dones": np.zeros((buf_size, n_envs, 1)),
    }
    td["dones"][-1] = 1
    rb.add(td)
    sample = rb.sample_tensors(10, False, n_samples=3, sequence_length=5)
    assert isinstance(sample["observations"], torch.Tensor)
    assert sample["observations"].shape == torch.Size([3, 5, 10, 3, 64, 64])
    del rb
    shutil.rmtree(root_dir)
