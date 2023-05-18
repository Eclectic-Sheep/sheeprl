from dataclasses import dataclass

from sheeprl.algos.ppo.args import PPOArgs
from sheeprl.utils.parser import Arg


@dataclass
class PPOAtariArgs(PPOArgs):
    env_id: str = Arg(default="PongNoFrameskip-v4", help="the id of the environment")
    frame_stack: int = Arg(default=4, help="how many frames to stack")
    screen_size: int = Arg(
        default=64, help="the dimension of the image rendered by the environment (screen_size x screen_size)"
    )


@dataclass
class PPOPixelContinuousArgs(PPOArgs):
    env_id: str = Arg(default="CarRacing-v2", help="the id of the environment")
    action_repeat: int = Arg(default=2, help="how many actions to repeat")
    screen_size: int = Arg(
        default=64, help="the dimension of the image rendered by the environment (screen_size x screen_size)"
    )
