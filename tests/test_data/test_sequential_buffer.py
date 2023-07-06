import pytest
import torch
from tensordict import TensorDict

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
    td1 = TensorDict({"t": torch.rand(2, 1, 1)}, batch_size=[2, n_envs])
    td2 = TensorDict({"t": torch.rand(2, 1, 1)}, batch_size=[2, n_envs])
    td3 = TensorDict({"t": torch.rand(3, 1, 1)}, batch_size=[3, n_envs])
    rb.add(td1)
    rb.add(td2)
    rb.add(td3)
    assert rb.full
    assert rb["t"][0] == td3["t"][-2]
    assert rb["t"][1] == td3["t"][-1]
    torch.testing.assert_close(rb["t"][2:4], td2["t"])


def test_seq_replay_buffer_add_single_td():
    buf_size = 5
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(6, 1, 1)}, batch_size=[6, n_envs])
    rb.add(td1)
    assert rb.full
    assert rb["t"][0] == td1["t"][-1]


def test_seq_replay_buffer_sample():
    buf_size = 10
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(11, 1, 1)}, batch_size=[11, n_envs])
    rb.add(td1)
    s = rb.sample(4, sequence_length=2)
    assert s.shape == torch.Size([1, 2, 4])


def test_seq_replay_buffer_sample_fail_full():
    buf_size = 5
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(6, 1, 1)}, batch_size=[6, n_envs])
    rb.add(td1)
    with pytest.raises(ValueError, match=f"larger than the replay buffer size"):
        rb.sample(4, sequence_length=2, n_samples=2)


def test_seq_replay_buffer_sample_one_element():
    buf_size = 1
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(1, 1, 1)}, batch_size=[1, n_envs])
    rb.add(td1)
    sample = rb.sample(1, sequence_length=1)
    assert rb.full
    assert sample["t"] == td1["t"]
    with pytest.raises(ValueError):
        rb.sample(1, sequence_length=2)


def test_seq_replay_buffer_sample_shapes():
    buf_size = 30
    n_envs = 2
    rb = SequentialReplayBuffer(buf_size, n_envs)
    t = TensorDict({f"t": torch.arange(60).reshape(-1, 2, 1) % buf_size}, batch_size=[30, n_envs])
    rb.add(t)
    sample = rb.sample(3, sequence_length=5, n_samples=2)
    assert sample.shape == torch.Size([2, 5, 3])


def test_seq_replay_buffer_sample_full():
    buf_size = 10000
    n_envs = 1
    seq_len = 50
    rb = SequentialReplayBuffer(buf_size, n_envs)
    t = TensorDict({"t": torch.arange(10500).reshape(-1, 1, 1) % buf_size}, batch_size=[10500, n_envs])
    rb.add(t)
    samples = rb.sample(1000, sequence_length=seq_len, n_samples=5)
    assert not torch.logical_and((samples["t"][:, 0, :] < rb._pos), (samples["t"][:, -1, :] >= rb._pos)).any()


def test_seq_replay_buffer_sample_full_large_sl():
    buf_size = 10000
    n_envs = 1
    seq_len = 1000
    rb = SequentialReplayBuffer(buf_size, n_envs)
    t = TensorDict({"t": torch.arange(10500).reshape(-1, 1, 1) % buf_size}, batch_size=[10500, n_envs])
    rb.add(t)
    samples = rb.sample(1000, sequence_length=seq_len, n_samples=5)
    assert not torch.logical_and(
        (samples["t"][:, 0, :] >= buf_size + rb._pos - seq_len + 1), (samples["t"][:, -1, :] < rb._pos)
    ).any()
    assert not torch.logical_and((samples["t"][:, 0, :] < rb._pos), (samples["t"][:, -1, :] >= rb._pos)).any()


def test_seq_replay_buffer_sampleable_items():
    buf_size = 10
    n_envs = 1
    seq_len = 10
    rb = SequentialReplayBuffer(buf_size, n_envs)
    t = TensorDict({"t": torch.arange(15).reshape(-1, 1, 1)}, batch_size=[15, n_envs])
    rb.add(t)
    with pytest.raises(ValueError, match=f"sampleable items"):
        rb.sample(5, sequence_length=seq_len, n_samples=2)


def test_seq_replay_buffer_sample_fail_not_full():
    buf_size = 10
    n_envs = 1
    seq_len = 8
    rb = SequentialReplayBuffer(buf_size, n_envs)
    t = TensorDict({"t": torch.arange(5).reshape(-1, 1, 1)}, batch_size=[5, n_envs])
    rb.add(t)
    with pytest.raises(ValueError, match=f"too long sequence length"):
        rb.sample(5, sequence_length=seq_len, n_samples=1)


def test_seq_replay_buffer_sample_not_full():
    buf_size = 10
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    rb._buf = TensorDict({"t": torch.ones(10, n_envs, 1) * 20}, batch_size=[10, n_envs])
    t = TensorDict({"t": torch.arange(7).reshape(-1, 1, 1) * 1.0}, batch_size=[7, n_envs])
    rb.add(t)
    sample = rb.sample(2, sequence_length=5, n_samples=2)
    assert (sample["t"] < 7).all()


def test_seq_replay_buffer_sample_no_add():
    buf_size = 10
    n_envs = 1
    rb = SequentialReplayBuffer(buf_size, n_envs)
    with pytest.raises(ValueError, match=f"No sample has been added"):
        rb.sample(2, sequence_length=5, n_samples=2)
