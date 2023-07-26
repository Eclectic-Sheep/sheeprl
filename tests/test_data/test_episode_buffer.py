import os
import shutil
import time

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
    td1 = TensorDict({"dones": torch.zeros(40, 1), "t": torch.ones(40, 1) * -1}, batch_size=[40])
    td2 = TensorDict({"dones": torch.zeros(45, 1), "t": torch.ones(45, 1) * -2}, batch_size=[45])
    td3 = TensorDict({"dones": torch.zeros(50, 1), "t": torch.ones(50, 1) * -3}, batch_size=[50])
    td1["dones"][-1] = 1
    td2["dones"][-1] = 1
    td3["dones"][-1] = 1
    rb.add(td1)
    rb.add(td2)
    rb.add(td3)
    samples = rb.sample(50, n_samples=5)
    assert samples.shape == torch.Size([5, sl, 50])
    for seq in samples.permute(0, -1, -2).reshape(-1, sl, 1):
        assert torch.isin(seq["t"], -1).all() or torch.isin(seq["t"], -2).all() or torch.isin(seq["t"], -3).all()
        assert len(torch.nonzero(seq["dones"])) == 0 or seq["dones"][-1] == 1


def test_episode_buffer_error_sample():
    buf_size = 10
    sl = 5
    rb = EpisodeBuffer(buf_size, sl)
    with pytest.raises(RuntimeError, match=f"No sample has been added"):
        rb.sample(2, 2)
    with pytest.raises(ValueError, match=f"Batch size must be greater than 0"):
        rb.sample(-1, n_samples=2)
    with pytest.raises(ValueError, match=f"The number of samples must be greater than 0"):
        rb.sample(2, -1)


def test_episode_buffer_prioritize_ends():
    buf_size = 100
    sl = 15
    rb = EpisodeBuffer(buf_size, sl)
    td1 = TensorDict({"dones": torch.zeros(15, 1)}, batch_size=[15])
    td2 = TensorDict({"dones": torch.zeros(25, 1)}, batch_size=[25])
    td3 = TensorDict({"dones": torch.zeros(30, 1)}, batch_size=[30])
    td1["dones"][-1] = 1
    td2["dones"][-1] = 1
    td3["dones"][-1] = 1
    rb.add(td1)
    rb.add(td2)
    rb.add(td3)
    samples = rb.sample(50, n_samples=5, prioritize_ends=True)
    assert samples.shape == torch.Size([5, sl, 50])
    assert torch.isin(samples["dones"], 1).any() > 0


def test_memmap_episode_buffer():
    buf_size = 10
    bs = 4
    sl = 4
    with pytest.warns(
        UserWarning,
        match="The buffer will be memory-mapped into the `/tmp` folder, this means that there is the"
        " possibility to lose the saved files. Set the `memmap_dir` to a known directory.",
    ):
        rb = EpisodeBuffer(buf_size, sl, memmap=True)
    for _ in range(buf_size // bs):
        td = TensorDict(
            {"observations": torch.randint(0, 256, (bs, 3, 64, 64), dtype=torch.uint8), "dones": torch.zeros(bs)},
            batch_size=[bs],
        )
        td["dones"][-1] = 1
        rb.add(td)
        assert rb[-1].is_memmap()
    assert rb.is_memmap


def test_memmap_to_file_episode_buffer():
    buf_size = 5
    bs = 5
    sl = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = EpisodeBuffer(buf_size, sl, memmap=True, memmap_dir=memmap_dir)
    for i in range(4):
        td = TensorDict(
            {"observations": torch.randint(0, 256, (bs, 3, 64, 64), dtype=torch.uint8), "dones": torch.zeros(bs)},
            batch_size=[bs],
        )
        td["dones"][-1] = 1
        rb.add(td)
        del td
        assert rb[-1].is_memmap()
        memmap_dir = os.path.dirname(rb.buffer[-1][rb.buffer[-1].sorted_keys[0]].filename)
        assert os.path.exists(os.path.join(memmap_dir, "meta.pt"))
        assert os.path.exists(os.path.join(memmap_dir, "dones.meta.pt"))
        assert os.path.exists(os.path.join(memmap_dir, "dones.memmap"))
        assert os.path.exists(os.path.join(memmap_dir, "observations.meta.pt"))
        assert os.path.exists(os.path.join(memmap_dir, "observations.memmap"))
    assert rb.is_memmap
    for ep in rb.buffer:
        del ep
    del rb
    shutil.rmtree(root_dir)
