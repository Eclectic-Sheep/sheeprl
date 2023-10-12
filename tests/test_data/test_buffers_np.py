import os
import shutil
import time

import numpy as np
import pytest
from lightning import Fabric

from sheeprl.data.buffers_np import ReplayBuffer


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
    td1 = {"a": np.random.rand(2, 1, 1)}
    rb.add(td1)
    assert not rb.full
    assert rb._pos == 2
    np.testing.assert_allclose(rb["a"][:2], td1["a"])


def test_replay_buffer_add_tds():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(2, 1, 1)}
    td2 = {"a": np.random.rand(2, 1, 1)}
    td3 = {"a": np.random.rand(3, 1, 1)}
    rb.add(td1)
    rb.add(td2)
    rb.add(td3)
    assert rb.full
    assert rb["a"][0] == td3["a"][-2]
    assert rb["a"][1] == td3["a"][-1]
    assert rb._pos == 2
    np.testing.assert_allclose(rb["a"][2:4], td2["a"])


def test_replay_buffer_add_tds_exceeding_buf_size_multiple_times():
    buf_size = 7
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(2, 1, 1)}
    td2 = {"a": np.random.rand(1, 1, 1)}
    td3 = {"a": np.random.rand(9, 1, 1)}
    rb.add(td1)
    rb.add(td2)
    assert not rb.full
    rb.add(td3)
    assert rb.full
    assert rb._pos == 5
    remainder = len(td3["a"]) % buf_size
    np.testing.assert_allclose(rb["a"][: rb._pos], td3["a"][rb.buffer_size - rb._pos + remainder :])


def test_replay_buffer_add_single_td_size_is_not_multiple():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(17, 1, 1)}
    rb.add(td1)
    assert rb.full
    assert rb._pos == 2
    remainder = len(td1["a"]) % buf_size
    np.testing.assert_allclose(rb["a"][:remainder], td1["a"][-remainder:])
    np.testing.assert_allclose(rb["a"][remainder:], td1["a"][-buf_size:-remainder])


def test_replay_buffer_add_single_td_size_is_multiple():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(20, 1, 1)}
    rb.add(td1)
    assert rb.full
    assert rb._pos == 0
    np.testing.assert_allclose(rb["a"], td1["a"][-buf_size:])


def test_replay_buffer_sample():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(6, 1, 1)}
    rb.add(td1)
    s = rb.sample(4)
    assert s["a"].shape == tuple([1, 4, 1])
    s = rb.sample(4, n_samples=3)
    assert s["a"].shape == tuple([3, 4, 1])


def test_replay_buffer_sample_next_obs_not_full():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"observations": np.arange(4).reshape(-1, 1, 1)}
    rb.add(td1)
    s = rb.sample(10, sample_next_obs=True)
    assert s["observations"].shape == tuple([1, 10, 1])
    assert td1["observations"][-1] not in s["observations"]


def test_replay_buffer_sample_next_obs_full():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"observations": np.arange(8).reshape(-1, 1, 1)}
    rb.add(td1)
    s = rb.sample(10, sample_next_obs=True)
    assert s["observations"].shape == tuple([1, 10, 1])
    assert td1["observations"][-1] not in s["observations"]


def test_replay_buffer_sample_full():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(6, 1, 1)}
    rb.add(td1)
    s = rb.sample(6)
    assert s["a"].shape == tuple([1, 6, 1])


def test_replay_buffer_sample_one_element():
    buf_size = 1
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"observations": np.random.rand(1, 1, 1)}
    rb.add(td1)
    sample = rb.sample(1)
    assert rb.full
    assert sample["observations"] == td1["observations"]
    with pytest.raises(ValueError):
        rb.sample(1, sample_next_obs=True)


def test_replay_buffer_sample_fail():
    buf_size = 1
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    with pytest.raises(ValueError, match="No sample has been added to the buffer"):
        rb.sample(1)
    with pytest.raises(ValueError, match="must be both greater than 0"):
        rb.sample(-1)


def test_memmap_replay_buffer():
    buf_size = 10
    n_envs = 4
    with pytest.raises(
        ValueError,
        match="The buffer is set to be memory-mapped but the 'memmap_dir'",
    ):
        rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=None)
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir)
    td = {"observations": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8)}
    rb.add(td)
    assert rb.is_memmap
    del rb
    shutil.rmtree(root_dir)


def test_memmap_to_file_replay_buffer():
    buf_size = 10
    n_envs = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir)
    td = {"observations": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8)}
    rb.add(td)
    assert rb.is_memmap
    assert os.path.exists(os.path.join(memmap_dir, "observations.memmap"))
    fabric = Fabric(devices=1, accelerator="cpu")
    ckpt_file = os.path.join(root_dir, "checkpoint", "ckpt.ckpt")
    fabric.save(ckpt_file, {"rb": rb})
    ckpt = fabric.load(ckpt_file)
    assert (ckpt["rb"]["observations"][:10] == rb["observations"][:10]).all()
    del rb
    del ckpt
    shutil.rmtree(root_dir)


def test_obs_keys_replay_buffer():
    buf_size = 10
    n_envs = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir, obs_keys=("rgb", "state", "tmp"))
    td = {
        "rgb": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8),
        "state": np.random.randint(0, 256, (10, n_envs, 8), dtype=np.uint8),
        "tmp": np.random.randint(0, 256, (10, n_envs, 5), dtype=np.uint8),
    }
    rb.add(td)
    sample = rb.sample(10, True)
    sample_keys = sample.keys()
    assert "rgb" in sample_keys
    assert "state" in sample_keys
    assert "tmp" in sample_keys
    assert "next_rgb" in sample_keys
    assert "next_state" in sample_keys
    assert "next_tmp" in sample_keys
    del rb
    shutil.rmtree(root_dir)


def test_obs_keys_replay_no_sample_next_obs_buffer():
    buf_size = 10
    n_envs = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir, obs_keys=("rgb", "state", "tmp"))
    td = {
        "rgb": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8),
        "state": np.random.randint(0, 256, (10, n_envs, 8), dtype=np.uint8),
        "tmp": np.random.randint(0, 256, (10, n_envs, 5), dtype=np.uint8),
        "next_rgb": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8),
        "next_state": np.random.randint(0, 256, (10, n_envs, 8), dtype=np.uint8),
        "next_tmp": np.random.randint(0, 256, (10, n_envs, 5), dtype=np.uint8),
    }
    rb.add(td)
    sample = rb.sample(10, False)
    sample_keys = sample.keys()
    assert "rgb" in sample_keys
    assert "state" in sample_keys
    assert "tmp" in sample_keys
    assert "next_rgb" in sample_keys
    assert "next_state" in sample_keys
    assert "next_tmp" in sample_keys
    del rb
    shutil.rmtree(root_dir)