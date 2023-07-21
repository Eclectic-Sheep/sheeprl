import pytest
import torch
from tensordict import TensorDict

from sheeprl.data.buffers import ReplayBuffer


def test_replay_buffer_wrong_buffer_size():
    with pytest.raises(ValueError):
        ReplayBuffer(-1)


def test_replay_buffer_wrong_n_envs():
    with pytest.raises(ValueError):
        ReplayBuffer(1, -1)


def test_replay_buffer_add_single_td_not_full():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(2, 1, 1)}, batch_size=[2, n_envs])
    rb.add(td1)
    assert not rb.full
    assert rb._pos == 2
    torch.testing.assert_close(rb["t"][:2], td1["t"])


def test_replay_buffer_add_tds():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(2, 1, 1)}, batch_size=[2, n_envs])
    td2 = TensorDict({"t": torch.rand(2, 1, 1)}, batch_size=[2, n_envs])
    td3 = TensorDict({"t": torch.rand(3, 1, 1)}, batch_size=[3, n_envs])
    rb.add(td1)
    rb.add(td2)
    rb.add(td3)
    assert rb.full
    assert rb["t"][0] == td3["t"][-2]
    assert rb["t"][1] == td3["t"][-1]
    assert rb._pos == 2
    torch.testing.assert_close(rb["t"][2:4], td2["t"])


def test_replay_buffer_add_tds_exceeding_buf_size_multiple_times():
    buf_size = 7
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(2, 1, 1)}, batch_size=[2, n_envs])
    td2 = TensorDict({"t": torch.rand(1, 1, 1)}, batch_size=[1, n_envs])
    td3 = TensorDict({"t": torch.rand(9, 1, 1)}, batch_size=[9, n_envs])
    rb.add(td1)
    rb.add(td2)
    assert not rb.full
    rb.add(td3)
    assert rb.full
    assert rb._pos == 5
    remainder = len(td3) % buf_size
    torch.testing.assert_close(rb["t"][: rb._pos], td3["t"][rb.buffer_size - rb._pos + remainder :])


def test_replay_buffer_add_single_td_size_is_not_multiple():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(17, 1, 1)}, batch_size=[17, n_envs])
    rb.add(td1)
    assert rb.full
    assert rb._pos == 2
    remainder = len(td1) % buf_size
    torch.testing.assert_close(rb["t"][:remainder], td1["t"][-remainder:])
    torch.testing.assert_close(rb["t"][remainder:], td1["t"][-buf_size:-remainder])


def test_replay_buffer_add_single_td_size_is_multiple():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(20, 1, 1)}, batch_size=[20, n_envs])
    rb.add(td1)
    assert rb.full
    assert rb._pos == 0
    torch.testing.assert_close(rb["t"], td1["t"][-buf_size:])


def test_replay_buffer_sample():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(6, 1, 1)}, batch_size=[6, n_envs])
    rb.add(td1)
    s = rb.sample(4)
    assert s.shape == torch.Size([4, 1])


def test_replay_buffer_sample_next_obs_not_full():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"observations": torch.arange(4).view(-1, 1, 1)}, batch_size=[4, n_envs])
    rb.add(td1)
    s = rb.sample(10, sample_next_obs=True)
    assert s.shape == torch.Size([10, 1])
    assert td1["observations"][-1] not in s["observations"]


def test_replay_buffer_sample_next_obs_full():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"observations": torch.arange(8).view(-1, 1, 1)}, batch_size=[8, n_envs])
    rb.add(td1)
    s = rb.sample(10, sample_next_obs=True)
    assert s.shape == torch.Size([10, 1])
    assert td1["observations"][-1] not in s["observations"]


def test_replay_buffer_sample_full():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"t": torch.rand(6, 1, 1)}, batch_size=[6, n_envs])
    rb.add(td1)
    s = rb.sample(6)
    assert s.shape == torch.Size([6, 1])


def test_replay_buffer_sample_one_element():
    buf_size = 1
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = TensorDict({"observations": torch.rand(1, 1, 1)}, batch_size=[1, n_envs])
    rb.add(td1)
    sample = rb.sample(1)
    assert rb.full
    assert sample["observations"] == td1["observations"]
    with pytest.raises(RuntimeError):
        rb.sample(1, sample_next_obs=True)


def test_replay_buffer_sample_fail():
    buf_size = 1
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    with pytest.raises(ValueError, match="No sample has been added to the buffer"):
        rb.sample(1)
    with pytest.raises(ValueError, match="Batch size must be greater than 0"):
        rb.sample(-1)


def test_memmap_replay_buffer():
    buf_size = 1000000
    n_envs = 4
    with pytest.warns(
        UserWarning,
        match="The buffer will be memory-mapped into the `/tmp` folder, this means that there is the"
        " possibility to lose the saved files. Set the `memmap_dir` to a known directory.",
    ):
        rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=None)
    td = TensorDict(
        {"observations": torch.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=torch.uint8)}, batch_size=[10, n_envs]
    )
    rb.add(td)
    assert rb.buffer.is_memmap()
