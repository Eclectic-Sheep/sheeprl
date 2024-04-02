import os
import shutil
import time

import numpy as np
import pytest
import torch

from sheeprl.data.buffers import EpisodeBuffer, ReplayBuffer
from sheeprl.utils.memmap import MemmapArray


def test_episode_buffer_wrong_buffer_size():
    with pytest.raises(ValueError, match="The buffer size must be greater than zero"):
        EpisodeBuffer(-1, 10)


def test_episode_buffer_wrong_sequence_length():
    with pytest.raises(ValueError, match="The sequence length must be greater than zero"):
        EpisodeBuffer(1, -1)


def test_episode_buffer_sequence_length_greater_than_batch_size():
    with pytest.raises(ValueError, match="The sequence length must be lower than the buffer size"):
        EpisodeBuffer(5, 10)


@pytest.mark.parametrize("memmap_mode", ["r", "x", "w", "z"])
def test_replay_buffer_wrong_memmap_mode(memmap_mode):
    with pytest.raises(ValueError, match="Accepted values for memmap_mode are"):
        EpisodeBuffer(10, 10, memmap_mode=memmap_mode, memmap=True)


def test_episode_buffer_add_episodes():
    buf_size = 30
    sl = 5
    n_envs = 1
    obs_keys = ("terminated",)
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep1 = {"terminated": np.zeros((sl, n_envs, 1)), "truncated": np.zeros((sl, n_envs, 1))}
    ep2 = {"terminated": np.zeros((sl + 5, n_envs, 1)), "truncated": np.zeros((sl + 5, n_envs, 1))}
    ep3 = {"terminated": np.zeros((sl + 10, n_envs, 1)), "truncated": np.zeros((sl + 10, n_envs, 1))}
    ep4 = {"terminated": np.zeros((sl, n_envs, 1)), "truncated": np.zeros((sl, n_envs, 1))}
    ep1["terminated"][-1] = 1
    ep2["truncated"][-1] = 1
    ep3["terminated"][-1] = 1
    ep4["truncated"][-1] = 1
    rb.add(ep1)
    rb.add(ep2)
    rb.add(ep3)
    rb.add(ep4)
    assert rb.full
    assert (rb._buf[-1]["terminated"] == ep4["terminated"][:, 0]).all()
    assert (rb._buf[0]["terminated"] == ep2["terminated"][:, 0]).all()


def test_episode_buffer_add_single_dict():
    buf_size = 5
    sl = 5
    n_envs = 4
    obs_keys = ("terminated", "truncated")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep1 = {"terminated": np.zeros((sl, n_envs, 1)), "truncated": np.zeros((sl, n_envs, 1))}
    ep1["truncated"][-1] = 1
    rb.add(ep1)
    assert rb.full
    for env in range(n_envs):
        assert (rb._buf[0]["terminated"] == ep1["terminated"][:, env]).all()


def test_episode_buffer_error_add():
    buf_size = 10
    sl = 5
    n_envs = 4
    obs_keys = ("terminated",)
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)

    ep1 = torch.zeros(sl, n_envs, 1)
    with pytest.raises(ValueError, match="`data` must be a dictionary containing Numpy arrays, but `data` is of type"):
        rb.add(ep1, validate_args=True)

    ep2 = {"terminated": torch.zeros((sl, n_envs, 1)), "truncated": torch.zeros((sl, n_envs, 1))}
    with pytest.raises(ValueError, match="`data` must be a dictionary containing Numpy arrays. Found key"):
        rb.add(ep2, validate_args=True)

    ep3 = None
    with pytest.raises(ValueError, match="The `data` replay buffer must be not None"):
        rb.add(ep3, validate_args=True)

    ep4 = {"terminated": np.zeros((1,)), "truncated": np.zeros((1,))}
    with pytest.raises(RuntimeError, match=r"`data` must have at least 2: \[sequence_length, n_envs"):
        rb.add(ep4, validate_args=True)

    obs_keys = ("terminated", "obs")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep5 = {"terminated": np.zeros((sl, n_envs, 1)), "truncated": np.zeros((sl, n_envs, 1)), "obs": np.zeros((sl, 1, 6))}
    with pytest.raises(RuntimeError, match="Every array in `data` must be congruent in the first 2 dimensions"):
        rb.add(ep5, validate_args=True)

    ep6 = {"obs": np.zeros((sl, 1, 6))}
    with pytest.raises(RuntimeError, match="The episode must contain the `terminated`"):
        rb.add(ep6, validate_args=True)

    ep7 = {"terminated": np.zeros((sl, 1, 1)), "truncated": np.zeros((sl, 1, 1))}
    ep7["terminated"][-1] = 1
    with pytest.raises(ValueError, match="The indices of the environment must be integers in"):
        rb.add(ep7, validate_args=True, env_idxes=[10])


def test_add_only_for_some_envs():
    buf_size = 10
    sl = 5
    n_envs = 4
    obs_keys = ("terminated", "truncated")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep1 = {"terminated": np.zeros((sl, n_envs - 2, 1)), "truncated": np.zeros((sl, n_envs - 2, 1))}
    rb.add(ep1, env_idxes=[0, 3])
    assert len(rb._open_episodes[0]) > 0
    assert len(rb._open_episodes[1]) == 0
    assert len(rb._open_episodes[2]) == 0
    assert len(rb._open_episodes[3]) > 0


def test_save_episode():
    buf_size = 100
    sl = 5
    n_envs = 4
    obs_keys = ("terminated", "truncated")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep_chunks = []
    for _ in range(8):
        chunk_dim = (np.random.randint(1, 8, (1,)).item(), 1)
        ep_chunks.append(
            {
                "terminated": np.zeros(chunk_dim),
                "truncated": np.zeros(chunk_dim),
            }
        )
    ep_chunks[-1]["terminated"][-1] = 1
    rb._save_episode(ep_chunks)

    assert len(rb._buf) == 1
    assert (
        np.concatenate([e["terminated"] for e in rb.buffer], axis=0)
        == np.concatenate([c["terminated"] for c in ep_chunks], axis=0)
    ).all()


def test_save_episode_errors():
    buf_size = 100
    sl = 5
    n_envs = 4
    obs_keys = ("terminated", "truncated")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)

    with pytest.raises(RuntimeError, match="Invalid episode, an empty sequence is given"):
        rb._save_episode([])

    ep_chunks = []
    for _ in range(8):
        chunk_dim = (np.random.randint(1, 8, (1,)).item(), 1)
        ep_chunks.append({"terminated": np.zeros(chunk_dim), "truncated": np.zeros(chunk_dim)})
    ep_chunks[-1]["terminated"][-1] = 1
    ep_chunks[0]["truncated"][-1] = 1
    with pytest.raises(RuntimeError, match="The episode must contain exactly one done"):
        rb._save_episode(ep_chunks)

    ep_chunks = []
    for _ in range(8):
        chunk_dim = (np.random.randint(1, 8, (1,)).item(), 1)
        ep_chunks.append({"terminated": np.zeros(chunk_dim), "truncated": np.zeros(chunk_dim)})
    ep_chunks[0]["terminated"][-1] = 1
    with pytest.raises(RuntimeError, match="The episode must contain exactly one done"):
        rb._save_episode(ep_chunks)

    ep_chunks = [{"terminated": np.ones((1, 1)), "truncated": np.zeros((1, 1))}]
    with pytest.raises(RuntimeError, match="Episode too short"):
        rb._save_episode(ep_chunks)

    ep_chunks = [{"terminated": np.zeros((110, 1)), "truncated": np.zeros((110, 1))} for _ in range(8)]
    ep_chunks[-1]["truncated"][-1] = 1
    with pytest.raises(RuntimeError, match="Episode too long"):
        rb._save_episode(ep_chunks)


def test_episode_buffer_sample_one_element():
    buf_size = 5
    sl = 5
    n_envs = 1
    obs_keys = ("terminated", "truncated", "a")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep = {
        "terminated": np.zeros((sl, n_envs, 1)),
        "truncated": np.zeros((sl, n_envs, 1)),
        "a": np.random.rand(sl, n_envs, 1),
    }
    ep["terminated"][-1] = 1
    rb.add(ep)
    sample = rb.sample(1, n_samples=1, sequence_length=sl)
    assert rb.full
    assert (sample["terminated"][0, :, 0] == ep["terminated"][:, 0]).all()
    assert (sample["truncated"][0, :, 0] == ep["truncated"][:, 0]).all()
    assert (sample["a"][0, :, 0] == ep["a"][:, 0]).all()


def test_episode_buffer_sample_shapes():
    buf_size = 30
    sl = 2
    n_envs = 1
    obs_keys = ("terminated", "truncated")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep = {"terminated": np.zeros((sl, n_envs, 1)), "truncated": np.zeros((sl, n_envs, 1))}
    ep["truncated"][-1] = 1
    rb.add(ep)
    sample = rb.sample(3, n_samples=2, sequence_length=sl)
    assert sample["terminated"].shape[:-1] == tuple([2, sl, 3])
    assert sample["truncated"].shape[:-1] == tuple([2, sl, 3])
    sample = rb.sample(3, n_samples=2, sequence_length=sl, clone=True)
    assert sample["terminated"].shape[:-1] == tuple([2, sl, 3])
    assert sample["truncated"].shape[:-1] == tuple([2, sl, 3])


def test_episode_buffer_sample_more_episodes():
    buf_size = 100
    sl = 15
    n_envs = 1
    obs_keys = ("terminated", "truncated", "a")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep1 = {
        "terminated": np.zeros((40, n_envs, 1)),
        "a": np.ones((40, n_envs, 1)) * -1,
        "truncated": np.zeros((40, n_envs, 1)),
    }
    ep2 = {
        "terminated": np.zeros((45, n_envs, 1)),
        "a": np.ones((45, n_envs, 1)) * -2,
        "truncated": np.zeros((45, n_envs, 1)),
    }
    ep3 = {
        "terminated": np.zeros((50, n_envs, 1)),
        "a": np.ones((50, n_envs, 1)) * -3,
        "truncated": np.zeros((50, n_envs, 1)),
    }
    ep1["terminated"][-1] = 1
    ep2["truncated"][-1] = 1
    ep3["terminated"][-1] = 1
    rb.add(ep1)
    rb.add(ep2)
    rb.add(ep3)
    samples = rb.sample(50, n_samples=5, sequence_length=sl)
    assert samples["terminated"].shape[:-1] == tuple([5, sl, 50])
    assert samples["truncated"].shape[:-1] == tuple([5, sl, 50])
    samples = {k: np.moveaxis(samples[k], 2, 1).reshape(-1, sl, 1) for k in obs_keys}
    for i in range(len(samples["terminated"])):
        assert (
            np.isin(samples["a"][i], -1).all()
            or np.isin(samples["a"][i], -2).all()
            or np.isin(samples["a"][i], -3).all()
        )
        assert len(samples["terminated"][i].nonzero()[0]) == 0 or samples["terminated"][i][-1] == 1
        assert len(samples["truncated"][i].nonzero()[0]) == 0 or samples["truncated"][i][-1] == 1


def test_episode_buffer_error_sample():
    buf_size = 10
    sl = 5
    rb = EpisodeBuffer(buf_size, sl)
    with pytest.raises(RuntimeError, match="No valid episodes has been added to the buffer"):
        rb.sample(2, n_samples=2)
    with pytest.raises(ValueError, match="Batch size must be greater than 0"):
        rb.sample(-1, n_samples=2)
    with pytest.raises(ValueError, match="The number of samples must be greater than 0"):
        rb.sample(2, n_samples=-1)
    ep1 = {"terminated": np.zeros((15, 1, 1)), "truncated": np.zeros((15, 1, 1))}
    rb.add(ep1)
    with pytest.raises(RuntimeError, match="No valid episodes has been added to the buffer"):
        rb.sample(2, n_samples=2, sequence_length=20)
        rb.sample(2, n_samples=2, sample_next_obs=True, sequence_length=sl)


def test_episode_buffer_prioritize_ends():
    buf_size = 100
    sl = 15
    n_envs = 1
    obs_keys = ("terminated", "truncated")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys, prioritize_ends=True)
    ep1 = {"terminated": np.zeros((15, n_envs, 1)), "truncated": np.zeros((15, n_envs, 1))}
    ep2 = {"terminated": np.zeros((25, n_envs, 1)), "truncated": np.zeros((25, n_envs, 1))}
    ep3 = {"terminated": np.zeros((30, n_envs, 1)), "truncated": np.zeros((30, n_envs, 1))}
    ep1["truncated"][-1] = 1
    ep2["terminated"][-1] = 1
    ep3["truncated"][-1] = 1
    rb.add(ep1)
    rb.add(ep2)
    rb.add(ep3)
    samples = rb.sample(50, n_samples=5, sequence_length=sl)
    assert samples["terminated"].shape[:-1] == tuple([5, sl, 50])
    assert samples["truncated"].shape[:-1] == tuple([5, sl, 50])
    assert np.isin(samples["terminated"], 1).any() > 0
    assert np.isin(samples["truncated"], 1).any() > 0


def test_sample_next_obs():
    buf_size = 10
    sl = 5
    n_envs = 4
    obs_keys = ("terminated", "truncated")
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys)
    ep1 = {"terminated": np.zeros((sl, n_envs, 1)), "truncated": np.zeros((sl, n_envs, 1))}
    ep1["terminated"][-1] = 1
    rb.add(ep1)
    sample = rb.sample(10, True, n_samples=5, sequence_length=sl - 1)
    assert "next_terminated" in sample
    assert "next_truncated" in sample
    assert (sample["next_terminated"][:, -1] == 1).all()
    assert not (sample["next_truncated"][:, -1] == 1).any()


def test_memmap_episode_buffer():
    buf_size = 10
    bs = 4
    sl = 4
    n_envs = 1
    obs_keys = ("terminated", "truncated", "observations")
    with pytest.raises(
        ValueError,
        match="The buffer is set to be memory-mapped but the `memmap_dir` attribute is None",
    ):
        rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, memmap=True)

    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys, memmap=True, memmap_dir="test_episode_buffer")
    for _ in range(buf_size // bs):
        ep = {
            "observations": np.random.randint(0, 256, (bs, n_envs, 3, 64, 64), dtype=np.uint8),
            "terminated": np.zeros((bs, n_envs, 1)),
            "truncated": np.zeros((bs, n_envs, 1)),
        }
        ep["truncated"][-1] = 1
        rb.add(ep)
        assert isinstance(rb._buf[-1]["terminated"], MemmapArray)
        assert isinstance(rb._buf[-1]["truncated"], MemmapArray)
        assert isinstance(rb._buf[-1]["observations"], MemmapArray)
    assert rb.is_memmap
    del rb
    shutil.rmtree(os.path.abspath("test_episode_buffer"))


def test_memmap_to_file_episode_buffer():
    buf_size = 10
    bs = 5
    sl = 4
    n_envs = 1
    obs_keys = ("terminated", "truncated", "observations")
    memmap_dir = "test_episode_buffer"
    rb = EpisodeBuffer(buf_size, sl, n_envs=n_envs, obs_keys=obs_keys, memmap=True, memmap_dir=memmap_dir)
    for i in range(4):
        if i >= 2:
            bs = 7
        else:
            bs = 5
        ep = {
            "observations": np.random.randint(0, 256, (bs, n_envs, 3, 64, 64), dtype=np.uint8),
            "terminated": np.zeros((bs, n_envs, 1)),
            "truncated": np.zeros((bs, n_envs, 1)),
        }
        ep["terminated"][-1] = 1
        rb.add(ep)
        del ep
        assert isinstance(rb._buf[-1]["terminated"], MemmapArray)
        assert isinstance(rb._buf[-1]["truncated"], MemmapArray)
        assert isinstance(rb._buf[-1]["observations"], MemmapArray)
        memmap_dir = os.path.dirname(rb._buf[-1]["terminated"].filename)
        memmap_dir = os.path.dirname(rb._buf[-1]["truncated"].filename)
        assert os.path.exists(os.path.join(memmap_dir, "terminated.memmap"))
        assert os.path.exists(os.path.join(memmap_dir, "truncated.memmap"))
        assert os.path.exists(os.path.join(memmap_dir, "observations.memmap"))
    assert rb.is_memmap
    for ep in rb.buffer:
        del ep
    del rb
    shutil.rmtree(os.path.abspath("test_episode_buffer"))


def test_sample_tensors():
    import torch

    buf_size = 10
    n_envs = 1
    rb = EpisodeBuffer(buf_size, n_envs)
    td = {
        "observations": np.arange(8).reshape(-1, 1, 1),
        "terminated": np.zeros((8, 1, 1)),
        "truncated": np.zeros((8, 1, 1)),
    }
    td["truncated"][-1] = 1
    rb.add(td)
    s = rb.sample_tensors(10, sample_next_obs=True, n_samples=3, sequence_length=5)
    assert isinstance(s["observations"], torch.Tensor)
    assert s["observations"].shape == torch.Size([3, 5, 10, 1])
    s = rb.sample_tensors(10, sample_next_obs=True, n_samples=3, sequence_length=5, from_numpy=True, clone=True)
    assert isinstance(s["observations"], torch.Tensor)
    assert s["observations"].shape == torch.Size([3, 5, 10, 1])


def test_sample_tensor_memmap():
    import torch

    buf_size = 10
    n_envs = 4
    root_dir = os.path.join("pytest_" + str(int(time.time())))
    memmap_dir = os.path.join(root_dir, "memmap_buffer")
    rb = EpisodeBuffer(buf_size, n_envs, memmap=True, memmap_dir=memmap_dir, obs_keys=("observations"))
    td = {
        "observations": np.random.randint(0, 256, (10, n_envs, 3, 64, 64), dtype=np.uint8),
        "terminated": np.zeros((buf_size, n_envs, 1)),
        "truncated": np.zeros((buf_size, n_envs, 1)),
    }
    td["terminated"][-1] = 1
    rb.add(td)
    sample = rb.sample_tensors(10, False, n_samples=3, sequence_length=5)
    assert isinstance(sample["observations"], torch.Tensor)
    assert sample["observations"].shape == torch.Size([3, 5, 10, 3, 64, 64])
    del rb
    shutil.rmtree(root_dir)


def test_add_rb():
    buf_size = 10
    n_envs = 3
    rb = ReplayBuffer(buf_size, n_envs)
    rb.add(
        {
            "terminated": np.zeros((buf_size, n_envs, 1)),
            "truncated": np.zeros((buf_size, n_envs, 1)),
            "a": np.random.rand(buf_size, n_envs, 5),
        }
    )
    rb["truncated"][-1] = 1
    epb = EpisodeBuffer(buf_size * n_envs, minimum_episode_length=2, n_envs=n_envs)
    epb.add(rb)
    assert (rb["a"][:, 0] == epb._buf[0]["a"]).all()
    assert (rb["a"][:, 1] == epb._buf[1]["a"]).all()
    assert (rb["a"][:, 2] == epb._buf[2]["a"]).all()
