from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.utilities.cloud_io import get_filesystem
from torch import Tensor

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils import fs
from sheeprl.utils.memmap import copied_on_pickle


class CheckpointCallback:
    """Callback to checkpoint the training: the models, the optimizers and the replay buffers.

    `on_checkpoint_coupled` is called by all the processes: the process of rank 0 gets the buffers of all the processes
    and saves the state of the training.

    When the buffer is added to the state of the checkpoint, it is assumed that the episode is truncated.
    """

    def __init__(self, keep_last: int | None = None, memmap_buffer: str = "reference") -> None:
        """
        Args:
            keep_last: the number of checkpoints to keep, the latest ones (all of them when `None`).
            memmap_buffer: how the memory-mapped replay buffers are saved (`checkpoint.memmap_buffer`): `reference`
                saves the paths of their files, `copy` also their data (`sheeprl.utils.memmap.copied_on_pickle`).
        """
        if memmap_buffer not in ("reference", "copy"):
            raise ValueError(f"`memmap_buffer` must be 'reference' or 'copy', got '{memmap_buffer}'")
        self.keep_last = keep_last
        self.memmap_buffer = memmap_buffer

    def on_checkpoint_coupled(
        self,
        fabric: Fabric,
        ckpt_path: str,
        state: Dict[str, Any],
        replay_buffer: Optional[Any] = None,
    ):
        if replay_buffer is not None:
            # The batches being prefetched by a `ReplayStore` are gathered first
            getattr(replay_buffer, "wait", lambda: None)()
            rb_state = self._ckpt_rb(replay_buffer)
            # Every process saves its own buffer next to the checkpoint (`sheeprl.utils.fs.buffer_path`), instead of
            # gathering them in the first one: the checkpoint references their files, and a resume loads only its own.
            # The memory-mapped buffers are saved with their data with `memmap_buffer=copy`
            path = fs.buffer_path(ckpt_path, fabric.global_rank)
            fs.makedirs(fs.parent(path))
            with copied_on_pickle(self.memmap_buffer == "copy"), get_filesystem(path).open(path, "wb") as f:
                torch.save(replay_buffer, f)
            # The checkpoint is written once the buffers of all the processes are
            fabric.barrier()
            state["rb"] = fs.BufferFiles([fs.buffer_path(ckpt_path, rank) for rank in range(fabric.world_size)])
        # The version that saved the checkpoint: the checkpoints are not compatible across major versions
        from sheeprl import __version__

        state["sheeprl_version"] = __version__
        fabric.save(ckpt_path, state)
        if replay_buffer is not None:
            self._experiment_consistent_rb(replay_buffer, rb_state)
        if fabric.is_global_zero and self.keep_last:
            self._delete_old_checkpoints(fs.parent(ckpt_path))

    def _ckpt_rb(self, rb: Any) -> Any:
        """Make the replay buffer (the `ReplayBuffer` of a `ReplayStore`) consistent for the checkpoint: the last step
        of every environment is truncated, because the state of the environments is not saved in the checkpoint.

        Returns:
            The true `truncated` of the last steps, to restore after the checkpoint (`_experiment_consistent_rb`).
        """
        rb: ReplayBuffer = getattr(rb, "storage", rb)
        rows, envs = (rb.positions - 1) % rb.buffer_size, np.arange(rb.n_envs)
        state = _copy(rb["truncated"][rows, envs])
        rb["truncated"][rows, envs] = 1
        # A memory-mapped buffer is saved by reference to its files, where the truncation is undone after the
        # checkpoint: the loaded buffer writes it again when it is used
        rb._checkpoint_truncation = rb.is_memmap
        return state

    def _experiment_consistent_rb(self, rb: Any, state: Any) -> None:
        """Restore the true `truncated` of the last steps, after the checkpoint (it undoes `_ckpt_rb`)."""
        rb: ReplayBuffer = getattr(rb, "storage", rb)
        rb["truncated"][(rb.positions - 1) % rb.buffer_size, np.arange(rb.n_envs)] = state
        rb._checkpoint_truncation = False

    def _delete_old_checkpoints(self, ckpt_folder: str):
        """Keep the last `keep_last` checkpoints of `ckpt_folder`, also on a remote filesystem (`sheeprl.utils.fs`)."""
        ckpts = fs.checkpoints(ckpt_folder)
        for ckpt in ckpts[: max(len(ckpts) - self.keep_last, 0)]:
            fs.remove_checkpoint(ckpt, ckpt_folder)


def _copy(row: Any) -> Any:
    """A copy of a row of a buffer: a NumPy array, or a tensor of a buffer in the memory of a device."""
    return row.clone() if isinstance(row, Tensor) else row.copy()
