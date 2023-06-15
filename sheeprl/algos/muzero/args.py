from dataclasses import dataclass

from sheeprl.algos.args import StandardArgs
from sheeprl.utils.parser import Arg


@dataclass
class MuzeroArgs(StandardArgs):
    # processes
    num_players: int = Arg(default=1, help="The number of paraller players' processes.")
    num_trainers: int = Arg(default=1, help="The number of parallel trainers' processes.")
    # environment
    capture_video: bool = Arg(
        default=False, help="Whether to capture videos of the agent performances (check out `videos` folder)"
    )
    mask_vel: bool = Arg(default=False, help="Whether to mask the velocity of in the observation.")
    # players
    total_steps: int = Arg(default=1e6, help="Total timesteps of the experiments.")
    learning_starts: int = Arg(default=100, help="timestep to start learning")
    max_trajectory_len: int = Arg(default=1000, help="The maximum length of a trajectory.")
    num_simulations: int = Arg(default=50, help="The number of MCTS simulations to run for each action.")
    gamma: float = Arg(default=0.997, help="The discount factor.")
    dirichlet_alpha: float = Arg(default=0.25, help="The alpha parameter of the Dirichlet distribution.")
    exploration_fraction: float = Arg(default=0.25, help="The fraction of the exploration phase.")
    checkpoint_every: int = Arg(default=-1, help="How often to make the checkpoint, -1 to deactivate the checkpoint.")
    # trainer
    lr: float = Arg(default=0.05, help="The learning rate of the optimizer.")
    weight_decay: float = Arg(default=1e-4, help="The weight decay of the optimizer.")
    anneal_lr: bool = Arg(default=True, help="Whether to anneal the learning rate.")
    decay_period: int = Arg(default=350_000, help="The period of the learning rate annealing.")
    decay_factor: float = Arg(default=0.1, help="The factor of the learning rate annealing.")
    update_epochs: int = Arg(default=5, help="The number of epochs to train the network.")
    # buffer
    buffer_capacity: int = Arg(default=100_000, help="The maximum number of trajectories in the replay buffer.")
    chunk_sequence_len: int = Arg(default=5, help="The length of a chunk of trajectory in a batch.")
    chunks_per_batch: int = Arg(default=32, help="The number of chunks in a batch.")
