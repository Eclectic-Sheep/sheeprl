from dataclasses import dataclass

from sheeprl.algos.sac.args import SACArgs
from sheeprl.utils.parser import Arg


@dataclass
class DROQArgs(SACArgs):
    dropout: float = Arg(default=0.01, help="the dropout probability for the critic network")
    gradient_steps: int = Arg(default=20, help="the number of gradient steps per each environment interaction")
