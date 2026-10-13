from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from lightning.fabric import Fabric
from lightning.fabric.utilities.cloud_io import get_filesystem
from torch import Tensor

from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer, ReplayBuffer
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
        replay_buffer: Optional[Union["EnvIndependentReplayBuffer", "ReplayBuffer", "EpisodeBuffer"]] = None,
    ):
        if replay_buffer is not None:
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
        fabric.save(ckpt_path, state)
        if replay_buffer is not None:
            self._experiment_consistent_rb(replay_buffer, rb_state)
        if fabric.is_global_zero and self.keep_last:
            self._delete_old_checkpoints(fs.parent(ckpt_path))

    def _ckpt_rb(
        self, rb: ReplayBuffer | EnvIndependentReplayBuffer | EpisodeBuffer
    ) -> Tensor | Sequence[Tensor] | Sequence[Sequence[Tensor]]:
        """Modify the replay buffer in order to be consistent for the checkpoint.
        There could be 3 cases, depending on the buffers:

        1. The `ReplayBuffer` or `SequentialReplayBuffer`: a done is inserted in the last pos because the
            state of the environment is not saved in the checkpoint.
        2. The `EnvIndependentReplayBuffer`: for each buffer, the done in the last position is set to True
            (for the same reason of the point 1.).
        3. The `EpisodeBuffer`: the open episodes are discarded  because the
            state of the environment is not saved in the checkpoint.

        Args:
            rb (ReplayBuffer | EnvIndependentReplayBuffer | EpisodeBuffer): the buffer.

        Returns:
            The original state of the buffer.
        """
        if isinstance(rb, ReplayBuffer):
            # clone the true done
            state = rb["truncated"][(rb._pos - 1) % rb.buffer_size, :].copy()
            # substitute the last done with all True values (all the environment are truncated)
            rb["truncated"][(rb._pos - 1) % rb.buffer_size, :] = 1
            # A memory-mapped buffer is saved by reference to its files, where the truncation is undone after the
            # checkpoint: the loaded buffer writes it again when it is used
            rb._checkpoint_truncation = rb.is_memmap
        elif isinstance(rb, EnvIndependentReplayBuffer):
            state = []
            for b in rb.buffer:
                state.append(b["truncated"][(b._pos - 1) % b.buffer_size, :].copy())
                b["truncated"][(b._pos - 1) % b.buffer_size, :] = 1
                b._checkpoint_truncation = b.is_memmap
        elif isinstance(rb, EpisodeBuffer):
            # remove open episodes from the buffer because the state of the environment is not saved
            state = rb._open_episodes
            rb._open_episodes = [[] for _ in range(rb.n_envs)]
        return state

    def _experiment_consistent_rb(
        self,
        rb: ReplayBuffer | EnvIndependentReplayBuffer | EpisodeBuffer,
        state: Tensor | Sequence[Tensor] | Sequence[Sequence[Tensor]],
    ):
        """Restore the state of the buffer consistent with the execution of the experiment.
        I.e., it undoes the changes in the _ckpt_rb function.

        Args:
            rb (ReplayBuffer | EnvIndependentReplayBuffer | EpisodeBuffer): the buffer.
            state (Tensor | Sequence[Tensor] | Sequence[Sequence[Tensor]]): the original state of the buffer.
        """
        if isinstance(rb, ReplayBuffer):
            # reinsert the true dones in the buffer
            rb["truncated"][(rb._pos - 1) % rb.buffer_size, :] = state
            rb._checkpoint_truncation = False
        elif isinstance(rb, EnvIndependentReplayBuffer):
            for i, b in enumerate(rb.buffer):
                b["truncated"][(b._pos - 1) % b.buffer_size, :] = state[i]
                b._checkpoint_truncation = False
        elif isinstance(rb, EpisodeBuffer):
            # reinsert the open episodes to continue the training
            rb._open_episodes = state

    def _delete_old_checkpoints(self, ckpt_folder: str):
        """Keep the last `keep_last` checkpoints of `ckpt_folder`, also on a remote filesystem (`sheeprl.utils.fs`)."""
        ckpts = fs.checkpoints(ckpt_folder)
        for ckpt in ckpts[: max(len(ckpts) - self.keep_last, 0)]:
            fs.remove_checkpoint(ckpt, ckpt_folder)
