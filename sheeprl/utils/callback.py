from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, Optional, Sequence, Union

from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives import TorchCollective
from torch import Tensor

from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer, ReplayBuffer


class CheckpointCallback:
    """Callback to checkpoint the training.
    Three methods are defined to checkpoint the models, the optimizers, and the replay buffers during the training:
        1. `on_checkpoint_coupled`: The method called by all processes in coupled algorithms,
            the process on rank-0 gets the buffers from all the processes and saves the state of the training.
        2. `on_checkpoint_player`: called by the player process of decoupled algorithms (rank-0),
            it receives the state from the trainer of rank-1 and, if required, adds the replay_buffer to the state.
        3. `on_checkpoint_trainer`: called by the rank-1 trainer process of decoupled algorithms that
            sends the state to the player process (rank-0).

    When the buffer is added to the state of the checkpoint, it is assumed that the episode is truncated.
    """

    def __init__(self, keep_last: int | None = None) -> None:
        self.keep_last = keep_last

    def on_checkpoint_coupled(
        self,
        fabric: Fabric,
        ckpt_path: str,
        state: Dict[str, Any],
        replay_buffer: Optional[Union["EnvIndependentReplayBuffer", "ReplayBuffer", "EpisodeBuffer"]] = None,
    ):
        if replay_buffer is not None:
            rb_state = self._ckpt_rb(replay_buffer)
            state["rb"] = replay_buffer
            if fabric.world_size > 1:
                # We need to collect the buffers from all the ranks
                # The collective it is needed because the `gather_object` function is not implemented in Fabric
                checkpoint_collective = TorchCollective()
                # gloo is the torch.distributed backend that works on cpu
                checkpoint_collective.create_group(backend="gloo", ranks=list(range(fabric.world_size)))
                gathered_rb = [None for _ in range(fabric.world_size)]
                if fabric.global_rank == 0:
                    checkpoint_collective.gather_object(replay_buffer, gathered_rb)
                    state["rb"] = gathered_rb
                else:
                    checkpoint_collective.gather_object(replay_buffer, None)
        fabric.save(ckpt_path, state)
        if replay_buffer is not None:
            self._experiment_consistent_rb(replay_buffer, rb_state)
        if fabric.is_global_zero and self.keep_last:
            self._delete_old_checkpoints(pathlib.Path(ckpt_path).parent)

    def on_checkpoint_player(
        self,
        fabric: Fabric,
        player_trainer_collective: TorchCollective,
        ckpt_path: str,
        replay_buffer: Optional["ReplayBuffer"] = None,
        ratio_state_dict: Dict[str, Any] | None = None,
    ):
        state = [None]
        player_trainer_collective.broadcast_object_list(state, src=1)
        state = state[0]
        if replay_buffer is not None:
            rb_state = self._ckpt_rb(replay_buffer)
            state["rb"] = replay_buffer
        if ratio_state_dict is not None:
            state["ratio"] = ratio_state_dict
        fabric.save(ckpt_path, state)
        if replay_buffer is not None:
            self._experiment_consistent_rb(replay_buffer, rb_state)
        if fabric.is_global_zero and self.keep_last:
            self._delete_old_checkpoints(pathlib.Path(ckpt_path).parent)

    def on_checkpoint_trainer(
        self, fabric: Fabric, player_trainer_collective: TorchCollective, state: Dict[str, Any], ckpt_path: str
    ):
        if fabric.global_rank == 1:
            player_trainer_collective.broadcast_object_list([state], src=1)
        fabric.save(ckpt_path, state)

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
        elif isinstance(rb, EnvIndependentReplayBuffer):
            state = []
            for b in rb.buffer:
                state.append(b["truncated"][(b._pos - 1) % b.buffer_size, :].copy())
                b["truncated"][(b._pos - 1) % b.buffer_size, :] = 1
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
        elif isinstance(rb, EnvIndependentReplayBuffer):
            for i, b in enumerate(rb.buffer):
                b["truncated"][(b._pos - 1) % b.buffer_size, :] = state[i]
        elif isinstance(rb, EpisodeBuffer):
            # reinsert the open episodes to continue the training
            rb._open_episodes = state

    def _delete_old_checkpoints(self, ckpt_folder: pathlib.Path):
        ckpts = list(sorted(ckpt_folder.glob("*.ckpt"), key=os.path.getmtime))
        if len(ckpts) > self.keep_last:
            to_delete = ckpts[: -self.keep_last]
            [f.unlink() for f in to_delete]
