"""The paths of the runs, on the local filesystem or on a remote one (`log_root`, e.g. `s3://bucket/runs`), through
fsspec: the logs, the configurations and the checkpoints of a run can live where every node of a training reaches
them, without a shared filesystem."""

from __future__ import annotations

import os
import posixpath
from typing import Any, List

from lightning.fabric.utilities.cloud_io import get_filesystem
from omegaconf import DictConfig, OmegaConf


def is_url(path: str | os.PathLike) -> bool:
    """Whether `path` is the URL of a filesystem (e.g. `s3://bucket/runs`), not a local path."""
    return "://" in str(path)


def join(path: str | os.PathLike, *parts: str | os.PathLike) -> str:
    """`path` joined with `parts`, as `os.path.join` (an absolute part or a URL replaces what precedes it), with `/` in
    the URLs."""
    path = str(path)
    for part in map(str, parts):
        if is_url(part) or os.path.isabs(part):
            path = part
        elif is_url(path):
            path = posixpath.join(path, part)
        else:
            path = os.path.join(path, part)
    return path


def parent(path: str | os.PathLike, levels: int = 1) -> str:
    """The directory `levels` levels above `path`."""
    path = str(path)
    for _ in range(levels):
        path = posixpath.dirname(path.rstrip("/")) if is_url(path) else os.path.dirname(os.path.normpath(path))
    return path


def basename(path: str | os.PathLike) -> str:
    path = str(path)
    return posixpath.basename(path.rstrip("/")) if is_url(path) else os.path.basename(os.path.normpath(path))


def relative(path: str | os.PathLike) -> str:
    """`path` without its protocol, its drive and its root: e.g. to place it under another directory."""
    path = str(path)
    if is_url(path):
        path = path.split("://", 1)[1]
    return os.path.splitdrive(path)[1].lstrip("/\\")


def makedirs(path: str | os.PathLike) -> None:
    get_filesystem(str(path)).makedirs(str(path), exist_ok=True)


def save_yaml(config: Any, path: str | os.PathLike, resolve: bool = True) -> None:
    with get_filesystem(str(path)).open(str(path), "w") as f:
        OmegaConf.save(config, f, resolve=resolve)


def load_yaml(path: str | os.PathLike) -> DictConfig:
    with get_filesystem(str(path)).open(str(path), "r") as f:
        return OmegaConf.load(f)


def run_config(checkpoint_path: str | os.PathLike) -> str:
    """The configuration of the run of a checkpoint: `<log_dir>/config.yaml` for `<log_dir>/checkpoint/<ckpt>`."""
    return join(parent(checkpoint_path, 2), "config.yaml")


def checkpoints(folder: str | os.PathLike) -> List[str]:
    """The checkpoints in `folder`, the oldest first: ordered by their policy step (`ckpt_<step>_<rank>.ckpt`), since
    the times of the files are not kept by every filesystem."""
    fs = get_filesystem(str(folder))
    paths = fs.glob(join(str(folder), "*.ckpt"))

    def step(path: str) -> int:
        try:
            return int(posixpath.basename(path).split("_")[1])
        except (IndexError, ValueError):
            return -1

    return sorted(paths, key=step)


def remove_checkpoint(path: str, folder: str | os.PathLike) -> None:
    """Remove the checkpoint `path`, one of `checkpoints(folder)`, and the files of its replay buffers."""
    filesystem = get_filesystem(str(folder))
    pattern = f"{_stem(posixpath.basename(path))}_buffer_rank_*.pt"
    buffers = filesystem.glob(join(parent(folder), "checkpoint_buffers", pattern))
    for file in (path, *buffers):
        filesystem.rm(file)


def memmap_dir(cfg: Any, log_dir: str, rank: int) -> str:
    """The local directory of the memory-mapped buffers of the process of `rank`: in the log directory of the run, or,
    with `buffer.memmap_dir`, in that directory (in the path of the log directory, which keeps the runs apart). The
    memory-mapped files need a local directory: a run logged on a remote filesystem needs `buffer.memmap_dir`."""
    base = cfg.buffer.get("memmap_dir", None)
    if base is None:
        if is_url(log_dir) and cfg.buffer.get("memmap", False):
            raise ValueError(
                f"The memory-mapped buffers need a local directory, but the run is logged in '{log_dir}': set "
                "`buffer.memmap_dir`"
            )
        base = log_dir
    else:
        base = os.path.join(str(base), relative(log_dir))
    return os.path.join(base, "memmap_buffer", f"rank_{rank}")


def upload(local_path: str | os.PathLike, path: str | os.PathLike) -> None:
    """Copy the local file `local_path` to `path` (e.g. on a remote filesystem)."""
    get_filesystem(str(path)).put_file(str(local_path), str(path))


def buffer_path(checkpoint_path: str | os.PathLike, rank: int) -> str:
    """The file of the replay buffer of the process of `rank` saved with a checkpoint (`ckpt_<step>_<rank>.ckpt`):
    `<log_dir>/checkpoint_buffers/ckpt_<step>_buffer_rank_<rank>.pt`, the same for every process (the folder of the
    checkpoints keeps only them). Local paths are absolute, for a resume from another directory."""
    name = f"{_stem(basename(checkpoint_path))}_buffer_rank_{rank}.pt"
    path = join(parent(checkpoint_path, 2), "checkpoint_buffers", name)
    return path if is_url(path) else os.path.abspath(path)


def _stem(name: str) -> str:
    """`ckpt_<step>` for the checkpoint `ckpt_<step>_<rank>.ckpt`."""
    stem = name[: -len(".ckpt")] if name.endswith(".ckpt") else name
    head, _, tail = stem.rpartition("_")
    return head if head and tail.isdigit() else stem


class BufferFiles(list):
    """The replay buffers of the processes of a checkpoint, one file each (`buffer_path`), saved by every process
    instead of being gathered by the first one: a process loads only its own, when indexed
    (`state["rb"][fabric.global_rank]`). Iterated, it gives the paths of the files (as Fabric traverses a checkpoint).
    """

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, slice):
            return [self._load(path) for path in super().__getitem__(idx)]
        return self._load(super().__getitem__(idx))

    @staticmethod
    def _load(path: str) -> Any:
        import torch

        with get_filesystem(path).open(path, "rb") as f:
            return torch.load(f, weights_only=False)
