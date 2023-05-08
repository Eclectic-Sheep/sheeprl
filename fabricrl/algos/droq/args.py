from dataclasses import dataclass

from fabricrl.algos.sac.args import SACArgs
from fabricrl.utils.parser import Arg


@dataclass
class DROQArgs(SACArgs):
    dropout: float = Arg(default=0.01, help="the dropout probability for the critic network")
    gradient_steps: int = Arg(default=20, help="the number of gradient steps per each environment interaction")
