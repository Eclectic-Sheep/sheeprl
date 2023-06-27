import pytest
import torch
from tensordict import TensorDict

from sheeprl.data.buffers import EpisodeBuffer


def test_episode_buffer_wrong_buffer_size():
    with pytest.raises(ValueError):
        EpisodeBuffer(-1, 10)


def test_episode_buffer_wrong_sequence_length():
    with pytest.raises(ValueError):
        EpisodeBuffer(1, -1)


def test_episode_buffer_sequence_length_greater_than_batch_size():
    with pytest.raises(ValueError):
        EpisodeBuffer(5, 10)


def test_episode_buffer_add_episodes():
    buf_size = 30
    sl = 5
    rb = EpisodeBuffer(buf_size, sl)
    td1 = TensorDict({"dones": torch.zeros(sl, 1)}, batch_size=[sl])
    td2 = TensorDict({"dones": torch.zeros(sl + 5, 1)}, batch_size=[sl + 5])
    td3 = TensorDict({"dones": torch.zeros(sl + 10, 1)}, batch_size=[sl + 10])
    td4 = TensorDict({"dones": torch.zeros(sl, 1)}, batch_size=[sl])
    td1["dones"][-1] = 1
    td2["dones"][-1] = 1
    td3["dones"][-1] = 1
    td4["dones"][-1] = 1
    rb.add(td1)
    rb.add(td2)
    rb.add(td3)
    rb.add(td4)
    assert rb.full
    assert (rb[-1]["dones"] == td4["dones"]).all()
    assert (rb[0]["dones"] == td2["dones"]).all()


def test_episode_buffer_add_single_td():
    buf_size = 5
    sl = 5
    rb = EpisodeBuffer(buf_size, sl)
    td1 = TensorDict({"dones": torch.zeros(sl, 1)}, batch_size=[sl])
    td1["dones"][-1] = 1
    rb.add(td1)
    assert rb.full
    assert (rb[0]["dones"] == td1["dones"]).all()


def test_episode_buffer_error_add():
    buf_size = 10
    sl = 5
    rb = EpisodeBuffer(buf_size, sl)
    td1 = TensorDict({"dones": torch.zeros(sl - 2, 1)}, batch_size=[sl - 2])
    with pytest.raises(RuntimeError, match=f"The episode must contain exactly one done"):
        rb.add(td1)

    td1["dones"][-3:] = 1
    with pytest.raises(RuntimeError, match=f"The episode must contain exactly one done"):
        rb.add(td1)

    td1["dones"][-2:] = 0
    with pytest.raises(RuntimeError, match=f"The last step must contain a done"):
        rb.add(td1)

    td1["dones"][-3] = 0
    td1["dones"][-1] = 1
    with pytest.raises(RuntimeError, match=f"Episode too short"):
        rb.add(td1)

    td1 = TensorDict({"dones": torch.zeros(15, 1)}, batch_size=[15])
    td1["dones"][-1] = 1
    with pytest.raises(RuntimeError, match=f"Episode too long"):
        rb.add(td1)

    td1 = TensorDict({"t": torch.zeros(15, 1)}, batch_size=[15])
    td1["t"][-1] = 1
    with pytest.raises(KeyError, match=f'key "dones" not found'):
        rb.add(td1)


def test_episode_buffer_sample_one_element():
    buf_size = 5
    sl = 5
    rb = EpisodeBuffer(buf_size, sl)
    td1 = TensorDict({"dones": torch.zeros(sl, 1), "t": torch.rand(sl, 1)}, batch_size=[sl])
    td1["dones"][-1] = 1
    rb.add(td1)
    sample = rb.sample(1, 1)
    assert rb.full
    assert (sample["dones"][0, :, 0] == td1["dones"]).all()
    assert (sample["t"][0, :, 0] == td1["t"]).all()


def test_episode_buffer_sample_shapes():
    buf_size = 30
    sl = 2
    rb = EpisodeBuffer(buf_size, sl)
    t = TensorDict({f"dones": torch.zeros(sl, 1)}, batch_size=[sl])
    t["dones"][-1] = 1
    rb.add(t)
    sample = rb.sample(3, n_samples=2)
    assert sample.shape == torch.Size([2, sl, 3])


def test_episode_buffer_sample_more_episodes():
    buf_size = 100
    sl = 15
    rb = EpisodeBuffer(buf_size, sl)
    td1 = TensorDict({"dones": torch.zeros(20, 1), "t": torch.ones(20, 1) * -1}, batch_size=[20])
    td2 = TensorDict({"dones": torch.zeros(25, 1), "t": torch.ones(25, 1) * -2}, batch_size=[25])
    td3 = TensorDict({"dones": torch.zeros(30, 1), "t": torch.ones(30, 1) * -3}, batch_size=[30])
    td1["dones"][-1] = 1
    td2["dones"][-1] = 1
    td3["dones"][-1] = 1
    rb.add(td1)
    rb.add(td2)
    rb.add(td3)
    samples = rb.sample(50, n_samples=5)
    assert samples.shape == torch.Size([5, sl, 50])
    for seq in samples.permute(0, -1, -2).reshape(-1, sl, 1):
        assert (seq["t"] == -1).all() or (seq["t"] == -2).all() or (seq["t"] == -3).all()
        assert len(torch.nonzero(seq["dones"])) == 0 or seq["dones"][-1] == 1


## TODO: other tests


def test_episode_buffer_sample_full_large_sl():
    buf_size = 10000
    n_envs = 1
    seq_len = 1000
    rb = EpisodeBuffer(buf_size, n_envs)
    t = TensorDict({"t": torch.arange(10500).reshape(-1, 1, 1) % buf_size}, batch_size=[10500, n_envs])
    rb.add(t)
    samples = rb.sample(1000, sequence_length=seq_len, n_samples=5)
    assert not torch.logical_and(
        (samples["t"][:, 0, :] >= buf_size + rb._pos - seq_len + 1), (samples["t"][:, -1, :] < rb._pos)
    ).any()
    assert not torch.logical_and((samples["t"][:, 0, :] < rb._pos), (samples["t"][:, -1, :] >= rb._pos)).any()


def test_episode_buffer_sampleable_items():
    buf_size = 10
    n_envs = 1
    seq_len = 10
    rb = EpisodeBuffer(buf_size, n_envs)
    t = TensorDict({"t": torch.arange(15).reshape(-1, 1, 1)}, batch_size=[15, n_envs])
    rb.add(t)
    with pytest.raises(ValueError, match=f"sampleable items"):
        rb.sample(5, sequence_length=seq_len, n_samples=2)


def test_episode_buffer_sample_fail_not_full():
    buf_size = 10
    n_envs = 1
    seq_len = 8
    rb = EpisodeBuffer(buf_size, n_envs)
    t = TensorDict({"t": torch.arange(5).reshape(-1, 1, 1)}, batch_size=[5, n_envs])
    rb.add(t)
    with pytest.raises(ValueError, match=f"too long sequence length"):
        rb.sample(5, sequence_length=seq_len, n_samples=1)


def test_episode_buffer_sample_not_full():
    buf_size = 10
    n_envs = 1
    rb = EpisodeBuffer(buf_size, n_envs)
    rb._buf = TensorDict({"t": torch.ones(10, n_envs, 1) * 20}, batch_size=[10, n_envs])
    t = TensorDict({"t": torch.arange(7).reshape(-1, 1, 1) * 1.0}, batch_size=[7, n_envs])
    rb.add(t)
    sample = rb.sample(2, sequence_length=5, n_samples=2)
    assert (sample["t"] < 7).all()


def test_episode_buffer_sample_no_add():
    buf_size = 10
    n_envs = 1
    rb = EpisodeBuffer(buf_size, n_envs)
    with pytest.raises(ValueError, match=f"No sample has been added"):
        rb.sample(2, sequence_length=5, n_samples=2)
