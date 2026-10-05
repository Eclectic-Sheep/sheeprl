import os

import fsspec
import pytest
from omegaconf import OmegaConf

from sheeprl.utils import fs
from sheeprl.utils.utils import dotdict


@pytest.fixture()
def memory_dir():
    """A directory of the memory filesystem of fsspec, as a remote one, removed after the test."""
    path = "memory://pytest_fs"
    yield path
    memory = fsspec.filesystem("memory")
    if memory.exists(path):
        memory.rm(path, recursive=True)


def test_the_paths_of_a_remote_filesystem_keep_their_protocol_and_slashes():
    assert fs.join("s3://bucket/runs", "ppo/CartPole", "run") == "s3://bucket/runs/ppo/CartPole/run"
    # A URL or an absolute path replaces what precedes it, as in `os.path.join`
    assert fs.join("logs/runs", "s3://bucket/x") == "s3://bucket/x"
    assert fs.join("logs/runs", os.path.abspath("x")) == os.path.abspath("x")
    ckpt = "s3://bucket/runs/ppo/run/version_0/checkpoint/ckpt_8_0.ckpt"
    assert fs.run_config(ckpt) == "s3://bucket/runs/ppo/run/version_0/config.yaml"
    assert fs.parent(ckpt, 4) == "s3://bucket/runs/ppo" and fs.basename(fs.parent(ckpt, 2)) == "version_0"
    assert fs.relative("s3://bucket/runs") == "bucket/runs" and fs.relative(os.path.abspath("x")).endswith("x")


def test_the_configurations_and_the_checkpoints_of_a_run_on_a_remote_filesystem(memory_dir):
    folder = fs.join(memory_dir, "run", "checkpoint")
    fs.makedirs(folder)
    fs.save_yaml({"env": {"id": "CartPole-v1"}}, fs.join(memory_dir, "run", "config.yaml"))
    assert OmegaConf.to_container(fs.load_yaml(fs.join(memory_dir, "run", "config.yaml"))) == {
        "env": {"id": "CartPole-v1"}
    }
    memory = fsspec.filesystem("memory")
    for step in (16, 128, 32):
        memory.pipe(fs.join(folder, f"ckpt_{step}_0.ckpt"), b"")
    # Ordered by their step, not by their names or times
    assert [fs.basename(c) for c in fs.checkpoints(folder)] == ["ckpt_16_0.ckpt", "ckpt_32_0.ckpt", "ckpt_128_0.ckpt"]
    # A checkpoint is removed with the files of its replay buffers
    memory.pipe(fs.buffer_path(fs.join(folder, "ckpt_16_0.ckpt"), 1), b"")
    memory.pipe(fs.buffer_path(fs.join(folder, "ckpt_32_0.ckpt"), 1), b"")
    fs.remove_checkpoint(fs.checkpoints(folder)[0], folder)
    assert [fs.basename(p) for p in memory.ls(folder, detail=False)] == ["ckpt_128_0.ckpt", "ckpt_32_0.ckpt"]
    assert [fs.basename(p) for p in memory.ls(fs.join(memory_dir, "run", "checkpoint_buffers"), detail=False)] == [
        "ckpt_32_buffer_rank_1.pt"
    ]


def test_the_memory_mapped_buffers_are_in_a_local_directory():
    cfg = dotdict({"buffer": {"memmap": True, "memmap_dir": None}})
    assert fs.memmap_dir(cfg, "logs/runs/a/version_0", 1) == os.path.join(
        "logs/runs/a/version_0", "memmap_buffer", "rank_1"
    )
    with pytest.raises(ValueError, match="buffer.memmap_dir"):
        fs.memmap_dir(cfg, "s3://bucket/runs/a/version_0", 0)
    # In the path of the log directory, which keeps the runs apart
    cfg.buffer.memmap_dir = "/scratch"
    assert fs.memmap_dir(cfg, "s3://bucket/runs/a/version_0", 0) == os.path.join(
        "/scratch", "bucket/runs/a/version_0", "memmap_buffer", "rank_0"
    )
    # Without memory-mapped buffers, a remote log directory needs no local one
    cfg = dotdict({"buffer": {"memmap": False}})
    fs.memmap_dir(cfg, "s3://bucket/runs/a/version_0", 0)


def test_the_videos_of_a_run_on_a_remote_filesystem_are_uploaded(memory_dir):
    # Recorded in a local temporary directory, then copied to the run on the remote filesystem: `RecordVideo` wrote in
    # a local directory named after the URL
    import gymnasium as gym

    from sheeprl.utils.env import UploadedRecordVideo

    env = UploadedRecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"), fs.join(memory_dir, "videos"))
    local_folder = env.video_folder
    env.reset(seed=0)
    for _ in range(20):
        env.step(env.action_space.sample())
    env.close()
    videos = fsspec.filesystem("memory").ls(fs.join(memory_dir, "videos"), detail=False)
    assert len(videos) == 1 and videos[0].endswith(".mp4")
    assert not os.path.exists(local_folder)


def test_the_buffers_of_a_checkpoint_are_in_a_file_per_process_loaded_when_indexed(tmp_path, monkeypatch):
    import torch

    paths = [fs.buffer_path(str(tmp_path / "checkpoint" / "ckpt_8_0.ckpt"), rank) for rank in range(2)]
    # The same files for every process, whose checkpoint path ends with its rank
    assert paths[1] == fs.buffer_path(str(tmp_path / "checkpoint" / "ckpt_8_1.ckpt"), 1)
    assert os.path.basename(paths[1]) == "ckpt_8_buffer_rank_1.pt"
    assert os.path.dirname(paths[1]) == str(tmp_path / "checkpoint_buffers")
    fs.makedirs(tmp_path / "checkpoint_buffers")
    for rank, path in enumerate(paths):
        torch.save({"rank": rank}, path)
    loaded = []
    monkeypatch.setattr(fs.BufferFiles, "_load", staticmethod(lambda path: loaded.append(path) or torch.load(path)))
    buffers = fs.BufferFiles(paths)
    assert isinstance(buffers, list) and len(buffers) == 2
    assert buffers[1] == {"rank": 1} and loaded == [paths[1]]
