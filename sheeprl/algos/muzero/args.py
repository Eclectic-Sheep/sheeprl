from dataclasses import dataclass
from typing import Optional, Tuple

from sheeprl.algos.args import StandardArgs
from sheeprl.utils.parser import Arg


@dataclass
class MuzeroArgs(StandardArgs):
    action_space_size: int = Arg()
    max_moves: int = Arg()
    discount: float = Arg()
    dirichlet_alpha: float = Arg()
    num_simulations: int = Arg()
    batch_size: int = Arg()
    td_steps: int = Arg()
    num_actors: int = Arg()
    lr_init: float = Arg()
    lr_decay_steps: float = Arg()
    known_bounds: Optional[Tuple] = Arg()
    root_exploration_fraction = 0.25
    pb_c_base = 19652
    pb_c_init = 1.25
    training_steps = int(1000e3)
    checkpoint_interval = int(1e3)
    window_size = int(1e6)
    num_unroll_steps = 5
    weight_decay = 1e-4
    momentum = 0.9
    lr_decay_rate = 0.1
