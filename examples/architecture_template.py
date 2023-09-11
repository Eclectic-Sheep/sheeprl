"""
The following snippet is a template for the architecture of a distributed RL algorithm.
The main function is responsible for spawning the processes for the buffer, players, and trainers. There processes are
defined in the functions buffer, player, and trainer, respectively.
The buffer process is responsible for collecting the data from the players, sampling batches and sending each batch to
 the trainers.
The player process is responsible for playing the game and collecting the data from the environment.
The trainer process is responsible for receiving the data from the buffer, performing optimization, and sending the
updated parameters to the players.

The number of players and trainers is defined by the user by specifying the value of the `num_players` and `num_trainers`
variables. The number of buffers is always 1.
In total there are `num_players` + `num_trainers` + 1 processes.

Processes communicate through collectives. The collectives are defined in the main function and passed to the processes
as arguments. A schema of the collectives can be found is the `assets/images/architecture_template.png` file.

To run this script, execute the following command:
`lightning run model --devices=<num_processes> examples/architecture_template.py`
where num_processes is computed as descrbed above.
"""

import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.strategies import DDPStrategy


def player(
    world_collective: TorchCollective,
    buffer_players_collective: TorchCollective,
    players_trainer_collective: TorchCollective,
):
    """Player process.
    It receives updated parameters from the trainer and sends the collected data to the buffer.

    Args:
        world_collective: collective for the world group.
        buffer_players_collective: collective for the players group and the buffer, for sharing the collected data.
            The first rank is for the buffer.
        players_trainer_collective: collective for a trainer and the players, for sharing the parameters.
            The last rank is for the trainer.
    """
    params = torch.empty(4)
    players_trainer_collective.broadcast(params, src=world_collective.world_size - 1)  # Broadcast uses global rank
    if players_trainer_collective.rank == 0:
        print(f"Player {world_collective.rank}: Params received from trainer -> {params}")

    print(f"Player {world_collective.rank}: Playing the game and collecting data...")
    time.sleep(0.5 * world_collective.rank)
    local_buffer = torch.ones(4) * world_collective.rank

    print(f"Player {world_collective.rank}: Sending collected data to buffer...")
    buffer_players_collective.gather_object(local_buffer, None, dst=0)


def trainer(
    world_collective: TorchCollective,
    buffer_trainers_collective: TorchCollective,
    players_trainer_collective: TorchCollective,
    optimization_pg,
):
    """Trainer process.
    Send updated parameters to the players and receive the collected data from the buffer. Perform optimization with the
    received data.

    Args:
        world_collective: collective for the world group.
        buffer_trainers_collective: collective for the buffer and the trainers, for sharing the collected data.
            The first rank is for the buffer.
        players_trainer_collective: collective for a trainer and the players, for sharing the parameters.
            The last rank is for the trainer.
    """
    fabric = Fabric(strategy=DDPStrategy(process_group=optimization_pg))
    if players_trainer_collective.rank == players_trainer_collective.world_size - 1:
        params = torch.ones(4) * 2
        print(f"Trainer {world_collective.rank}: " f"Sending updated params to all players...")
        players_trainer_collective.broadcast(params, src=world_collective.world_size - 1)  # broadcast uses global rank

    print(f"Trainer {world_collective.rank}: Waiting for collected data from buffer...")
    data_to_be_received = [None]
    buffer_trainers_collective.scatter_object_list(
        data_to_be_received, None, src=0
    )  # scatter_object_list uses global rank
    sampled_data = data_to_be_received[0]
    if buffer_trainers_collective.rank == 1:
        print(f"Trainer {world_collective.rank}: Data received: {sampled_data}")

    print(f"Trainer {world_collective.rank}: Optimizing the model with the received data...")
    l = nn.Linear(4, 1)
    l = fabric.setup_module(l)
    o = l(sampled_data)
    F.mse_loss(o, torch.rand_like(o)).backward()


def buffer(
    world_collective: TorchCollective,
    buffer_players_collective: TorchCollective,
    buffer_trainers_collective: TorchCollective,
):
    """Buffer process.
    Receive data from the players and send it to the trainers.

    Args:
        world_collective: collective for the world group.
        buffer_players_collective: collective for the players group and the buffer, for receiving the collected data.
            The first rank is for the buffer.
        buffer_trainers_collective: collective for the buffer and the trainers, for sending the collected data.
            The first rank is for the buffer.
    """
    print(f"Buffer {world_collective.rank}: Waiting for collected data from players...")
    buffers = [None for _ in range(buffer_players_collective.world_size)]
    buffer_players_collective.gather_object([None], buffers, dst=0)  # gather_object uses global rank
    for i, b in enumerate(buffers):
        print(f"Buffer-{i}:", b)

    print(f"Buffer {world_collective.rank}: Sending collected data to trainers...")
    sampled_buffers = [torch.ones(4) * i for i in range(buffer_trainers_collective.world_size - 1)]
    buffer_trainers_collective.scatter_object_list(
        [None], [None] + sampled_buffers, src=0
    )  # scatter_object_list uses global rank


def main():
    # Ranks semantic:
    # rank-0: buffer
    # rank-1, ..., rank-num_players: players
    # rank-(num_players+1), ..., rank-(num_players+num_trainers): trainers
    num_players = 2
    num_trainers = 2

    devices = os.environ.get("LT_DEVICES", None)
    if devices is None or devices in ("1", "2"):
        raise RuntimeError(
            "Please run the script with the number of devices greater than 2: "
            "`lightning run model --devices=3 examples/architecture_template.py ...`"
        )

    world_collective = TorchCollective()
    world_collective.setup(
        backend="nccl" if os.environ.get("LT_ACCELERATOR", None) in ("gpu", "cuda") else "gloo",
        timeout=timedelta(days=1),
    )

    # Create a global group for all the processes
    world_collective.create_group(timeout=timedelta(days=1))
    global_rank = world_collective.rank

    # Trainers collective to train the model in paraller with all the other trainers, needed for the optimization_pg
    trainers_collective = TorchCollective()
    trainers_collective.create_group(
        ranks=list(range(num_players + 1, num_players + num_trainers + 1)),
        timeout=timedelta(days=1),
    )
    optimization_pg = trainers_collective.group

    # Players-Buffer collective, to share data between players and buffer
    buffer_players_collective = TorchCollective()
    buffer_players_collective.create_group(ranks=list(range(0, num_players + 1)), timeout=timedelta(days=1))

    # Trainers-Buffer collective, to share data between buffer and trainers
    buffer_trainers_collective = TorchCollective()
    buffer_trainers_collective.create_group(
        ranks=[0] + list(range(num_players + 1, num_players + num_trainers + 1)),
        timeout=timedelta(days=1),
    )

    # Trainer-Players collective, to share the updated parameters between trainer and players
    players_trainer_collective = TorchCollective()
    players_trainer_collective.create_group(
        ranks=list(range(1, num_players + 1)) + [num_players + num_trainers],
        timeout=timedelta(days=1),
    )

    if global_rank == 0:
        buffer(world_collective, buffer_players_collective, buffer_trainers_collective)
    elif 1 <= global_rank <= num_players:
        player(world_collective, buffer_players_collective, players_trainer_collective)
    else:
        trainer(
            world_collective,
            buffer_trainers_collective,
            players_trainer_collective,
            optimization_pg,
        )


if __name__ == "__main__":
    main()
