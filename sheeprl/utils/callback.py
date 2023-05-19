from typing import Any, Dict, Optional

import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives import TorchCollective

from sheeprl.data.buffers import ReplayBuffer


class CheckpointCallback:
    def on_checkpoint_coupled(
        self, fabric: Fabric, ckpt_path: str, state: Dict[str, Any], rb: Optional["ReplayBuffer"] = None
    ):
        if rb is not None:
            true_done = rb["dones"][(rb._pos - 1) % rb.buffer_size, :].clone()
            rb["dones"][(rb._pos - 1) % rb.buffer_size, :] = True
            state["rb"] = rb
            if fabric.world_size > 1:
                # We need to collect the buffers from all the ranks
                # The collective it is needed because the `gather_object` function is not implemented in Fabric
                checkpoint_collective = TorchCollective()
                # gloo is the torch.distributed backend that works on cpu
                if rb.device == torch.device("cpu"):
                    backend = "gloo"
                else:
                    backend = "nccl"
                checkpoint_collective.create_group(backend=backend, ranks=list(range(fabric.world_size)))
                gathered_rb = [None for _ in range(fabric.world_size)]
                if fabric.global_rank == 0:
                    checkpoint_collective.gather_object(rb, gathered_rb)
                    state["rb"] = gathered_rb
                else:
                    checkpoint_collective.gather_object(rb, None)
        fabric.save(ckpt_path, state)
        if rb is not None:
            rb["dones"][(rb._pos - 1) % rb.buffer_size, :] = true_done

    def on_checkpoint_player(
        self,
        fabric: Fabric,
        player_trainer_collective: TorchCollective,
        ckpt_path: str,
        rb: Optional["ReplayBuffer"] = None,
    ):
        state = [None]
        player_trainer_collective.broadcast_object_list(state, src=1)
        state = state[0]
        if rb is not None:
            true_done = rb["dones"][(rb._pos - 1) % rb.buffer_size, :].clone()
            rb["dones"][(rb._pos - 1) % rb.buffer_size, :] = True
            state["rb"] = rb
        fabric.save(ckpt_path, state)
        if rb is not None:
            rb["dones"][(rb._pos - 1) % rb.buffer_size, :] = true_done

    def on_checkpoint_trainer(self, player_trainer_collective: TorchCollective, state: Dict[str, Any]):
        player_trainer_collective.broadcast_object_list([state], src=1)
