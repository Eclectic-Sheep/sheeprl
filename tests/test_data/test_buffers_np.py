import os
import shutil
import time

import numpy as np
import pytest
from lightning import Fabric

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.memmap import MemmapArray


def test_replay_buffer_wrong_buffer_size():
    with pytest.raises(ValueError):
        ReplayBuffer(-1)


def test_replay_buffer_wrong_n_envs():
    with pytest.raises(ValueError):
        ReplayBuffer(1, -1)


@pytest.mark.parametrize("memmap_mode", ["r", "x", "w", "z"])
def test_replay_buffer_wrong_memmap_mode(memmap_mode):
    with pytest.raises(ValueError, match="Accepted values for memmap_mode are"):
        ReplayBuffer(10, 10, memmap_mode=memmap_mode, memmap=True)


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


def test_replay_buffer_add_replay_buffer():
    buf_size = 5
    n_envs = 1
    rb1 = ReplayBuffer(buf_size, n_envs)
    rb1.add({"a": np.random.rand(6, 1, 1)})
    rb2 = ReplayBuffer(buf_size, n_envs)
    rb2.add(rb1)
    assert (rb1.buffer["a"] == rb2.buffer["a"]).all()


def test_replay_buffer_add_error():
    import torch

    buf_size = 5
    n_envs = 3
    rb = ReplayBuffer(buf_size, n_envs)
    with pytest.raises(ValueError, match="must be a dictionary containing Numpy arrays"):
        rb.add([i for i in range(5)], validate_args=True)
    with pytest.raises(ValueError, match=r"must be a dictionary containing Numpy arrays\. Found key"):
        rb.add({"a": torch.rand(6, 1, 1)}, validate_args=True)

    with pytest.raises(RuntimeError, match="must have at least 2 dimensions"):
        rb.add(
            {
                "a": np.random.rand(
                    6,
                )
            },
            validate_args=True,
        )

    with pytest.raises(RuntimeError, match="Every array in 'data' must be congruent in the first 2 dimensions"):
        rb.add(
            {
                "a": np.random.rand(6, n_envs, 4),
                "b": np.random.rand(6, n_envs, 4),
                "c": np.random.rand(6, 1, 4),
            },
            validate_args=True,
        )


def test_replay_buffer_sample():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs, obs_keys=("a",))
    td1 = {"a": np.random.rand(6, 1, 1)}
    rb.add(td1)
    s = rb.sample(4)
    assert s["a"].shape == tuple([1, 4, 1])
    s = rb.sample(4, n_samples=3)
    assert s["a"].shape == tuple([3, 4, 1])
    s = rb.sample(4, n_samples=2, clone=True, sample_next_obs=True)
    assert s["a"].shape == tuple([2, 4, 1])
    assert s["next_a"].shape == tuple([2, 4, 1])


def test_replay_buffer_sample_one_sample_next_obs_error():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"a": np.random.rand(1, 1, 1)}
    rb.add(td1)
    with pytest.raises(RuntimeError, match="You want to sample the next observations"):
        rb.sample(1, sample_next_obs=True)


def test_replay_buffer_getitem_error():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    with pytest.raises(RuntimeError, match="The buffer has not been initialized"):
        rb["a"]
    td = {"a": np.random.rand(1, 1, 1)}
    rb.add(td)
    with pytest.raises(TypeError, match="'key' must be a string"):
        rb[0]


def test_replay_buffer_get_sample_empty_error():
    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    with pytest.raises(RuntimeError, match="The buffer has not been initialized"):
        rb._get_samples(np.zeros((1,)), sample_next_obs=True)


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


def test_sample_tensors():
    import torch

    buf_size = 5
    n_envs = 1
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"observations": np.arange(8).reshape(-1, 1, 1)}
    rb.add(td1)
    s = rb.sample_tensors(10, sample_next_obs=True, n_samples=3)
    assert isinstance(s["observations"], torch.Tensor)
    assert s["observations"].shape == torch.Size([3, 10, 1])


def test_sample_tensor_memmap():
    import torch

    buf_size = 10
    n_envs = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir, obs_keys=("observations"))
    td = {
        "observations": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8),
    }
    rb.add(td)
    sample = rb.sample_tensors(10, False, n_samples=3)
    assert isinstance(sample["observations"], torch.Tensor)
    assert sample["observations"].shape == torch.Size([3, 10, 3, 64, 64])
    del rb
    shutil.rmtree(root_dir)


def test_to_tensor():
    import torch

    buf_size = 5
    n_envs = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir, obs_keys=("observations"))
    td = {
        "observations": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8),
    }
    rb.add(td)
    sample = rb.to_tensor()
    assert isinstance(sample["observations"], torch.Tensor)
    assert sample["observations"].shape == torch.Size([buf_size, n_envs, 3, 64, 64])
    assert (td["observations"][5:] == sample["observations"].cpu().numpy()).all()
    del rb
    shutil.rmtree(root_dir)


def test_setitem():
    buf_size = 5
    n_envs = 4
    rb = ReplayBuffer(buf_size, n_envs)
    td1 = {"observations": np.arange(8).reshape(-1, 1, 1)}
    rb.add(td1)
    a = np.random.rand(buf_size, n_envs, 10)
    rb["a"] = a
    assert rb["a"].shape == tuple([buf_size, n_envs, 10])
    assert (rb["a"] == a).all()

    m = MemmapArray(filename="test.memmap", dtype=np.float32, shape=(buf_size, n_envs, 4))
    m.array = np.random.rand(buf_size, n_envs, 4)
    rb["m"] = m
    assert isinstance(rb["m"], np.ndarray) and not isinstance(rb["m"], (MemmapArray, np.memmap))
    assert rb["m"].shape == tuple([buf_size, n_envs, 4])
    assert (rb["m"] == m.array).all()

    del m
    os.unlink("test.memmap")


def test_setitem_memmap():
    buf_size = 5
    n_envs = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = ReplayBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir, obs_keys=("observations"))
    td = {
        "observations": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8),
    }
    rb.add(td)
    a = np.random.rand(buf_size, n_envs, 10)
    rb["a"] = a
    assert isinstance(rb["a"], MemmapArray)
    assert rb["a"].shape == tuple([buf_size, n_envs, 10])
    assert (rb["a"] == a).all()

    m = MemmapArray(filename=f"{root_dir}/test.memmap", dtype=np.float32, shape=(buf_size, n_envs, 4))
    m.array = np.random.rand(buf_size, n_envs, 4)
    rb["m"] = m
    assert isinstance(rb["m"], MemmapArray)
    assert rb["m"].shape == tuple([buf_size, n_envs, 4])
    assert (rb["m"].array == m.array).all()

    del m
    del rb
    shutil.rmtree(root_dir)


def test_setitem_error():
    import torch

    buf_size = 5
    n_envs = 4
    rb = ReplayBuffer(buf_size, n_envs)
    with pytest.raises(RuntimeError, match="The buffer has not been initialized"):
        rb["no_init"] = np.zeros((buf_size, n_envs, 1))

    td1 = {"observations": np.arange(8).reshape(-1, 1, 1)}
    rb.add(td1)

    with pytest.raises(ValueError, match=r"The value to be set must be an instance of 'np\.ndarray', 'np\.memmap'"):
        rb["torch"] = torch.zeros(buf_size, n_envs, 1)

    with pytest.raises(RuntimeError, match="must have at least two dimensions of dimension"):
        rb["wrong_buffer_size"] = np.zeros((buf_size + 3, n_envs, 1))
        rb["wrong_n_envs"] = np.zeros((buf_size, n_envs - 1, 1))
        rb["wrong_dims"] = np.zeros((10,))
