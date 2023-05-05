from dataclasses import dataclass

from fabricrl.algos.sac.args import SACArgs
from fabricrl.utils.parser import Arg


@dataclass
class DROQArgs(SACArgs):
    q_lr: float = Arg(default=3e-2, help="the learning rate of the critic network optimizer")
    alpha_lr: float = Arg(default=3e-2, help="the learning rate of the policy network optimizer")
    policy_lr: float = Arg(default=3e-2, help="the learning rate of the entropy coefficient optimizer")
    gradient_steps: int = Arg(default=20, help="the number of gradient steps per each environment interaction")
