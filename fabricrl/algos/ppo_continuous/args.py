from dataclasses import dataclass

from fabricrl.algos.ppo.args import PPOArgs
from fabricrl.utils.parser import Arg


@dataclass
class PPOContinuousArgs(PPOArgs):
    env_id: str = Arg(default="LunarLanderContinuous-v2", help="the id of the environment")
