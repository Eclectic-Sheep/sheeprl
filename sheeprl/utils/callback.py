from typing import Any, Dict, Optional, Union

import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives import TorchCollective

from sheeprl.data.buffers import AsyncReplayBuffer, EpisodeBuffer, ReplayBuffer


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

    def on_checkpoint_coupled(
        self,
        fabric: Fabric,
        ckpt_path: str,
        state: Dict[str, Any],
        replay_buffer: Optional[Union["AsyncReplayBuffer", "ReplayBuffer", "EpisodeBuffer"]] = None,
    ):
        if replay_buffer is not None:
            if isinstance(replay_buffer, ReplayBuffer):
                # clone the true done
                true_done = replay_buffer["dones"][(replay_buffer._pos - 1) % replay_buffer.buffer_size, :].clone()
                # substitute the last done with all True values (all the environment are truncated)
                replay_buffer["dones"][(replay_buffer._pos - 1) % replay_buffer.buffer_size, :] = True
            elif isinstance(replay_buffer, AsyncReplayBuffer):
                true_dones = []
                for b in replay_buffer.buffer:
                    true_dones.append(b["dones"][(b._pos - 1) % b.buffer_size, :].clone())
                    b["dones"][(b._pos - 1) % b.buffer_size, :] = True
            state["rb"] = replay_buffer
            if fabric.world_size > 1:
                # We need to collect the buffers from all the ranks
                # The collective it is needed because the `gather_object` function is not implemented in Fabric
                checkpoint_collective = TorchCollective()
                # gloo is the torch.distributed backend that works on cpu
                if replay_buffer.device == torch.device("cpu"):
                    backend = "gloo"
                else:
                    backend = "nccl"
                checkpoint_collective.create_group(backend=backend, ranks=list(range(fabric.world_size)))
                gathered_rb = [None for _ in range(fabric.world_size)]
                if fabric.global_rank == 0:
                    checkpoint_collective.gather_object(replay_buffer, gathered_rb)
                    state["rb"] = gathered_rb
                else:
                    checkpoint_collective.gather_object(replay_buffer, None)
        fabric.save(ckpt_path, state)
        if replay_buffer is not None and isinstance(replay_buffer, ReplayBuffer):
            # reinsert the true dones in the buffer
            replay_buffer["dones"][(replay_buffer._pos - 1) % replay_buffer.buffer_size, :] = true_done
        elif isinstance(replay_buffer, AsyncReplayBuffer):
            for i, b in enumerate(replay_buffer.buffer):
                b["dones"][(b._pos - 1) % b.buffer_size, :] = true_dones[i]

    def on_checkpoint_player(
        self,
        fabric: Fabric,
        player_trainer_collective: TorchCollective,
        ckpt_path: str,
        replay_buffer: Optional["ReplayBuffer"] = None,
    ):
        state = [None]
        player_trainer_collective.broadcast_object_list(state, src=1)
        state = state[0]
        if replay_buffer is not None:
            # clone the true done
            true_done = replay_buffer["dones"][(replay_buffer._pos - 1) % replay_buffer.buffer_size, :].clone()
            # substitute the last done with all True values (all the environment are truncated)
            replay_buffer["dones"][(replay_buffer._pos - 1) % replay_buffer.buffer_size, :] = True
            state["rb"] = replay_buffer
        fabric.save(ckpt_path, state)
        if replay_buffer is not None:
            # reinsert the true dones in the buffer
            replay_buffer["dones"][(replay_buffer._pos - 1) % replay_buffer.buffer_size, :] = true_done

    def on_checkpoint_trainer(
        self, fabric: Fabric, player_trainer_collective: TorchCollective, state: Dict[str, Any], ckpt_path: str
    ):
        if fabric.global_rank == 1:
            player_trainer_collective.broadcast_object_list([state], src=1)
        fabric.save(ckpt_path, state)
