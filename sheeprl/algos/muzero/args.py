from dataclasses import dataclass

from sheeprl.algos.args import StandardArgs
from sheeprl.utils.parser import Arg


@dataclass
class MuzeroArgs(StandardArgs):
    num_players: int = Arg(default=1, help="The number of paraller players' processes.")
    num_trainers: int = Arg(default=1, help="The number of parallel trainers' processes.")
