from dataclasses import dataclass

from sheeprl.algos.ppo.args import PPOArgs
from sheeprl.utils.parser import Arg


@dataclass
class PPOAtariArgs(PPOArgs):
    env_id: str = Arg(default="PongNoFrameskip-v4", help="the id of the environment")
    frame_stack: int = Arg(default=4, help="how many frames to stack")
