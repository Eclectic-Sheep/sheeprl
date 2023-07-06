from dataclasses import dataclass

from sheeprl.algos.ppo.args import PPOArgs
from sheeprl.utils.parser import Arg


@dataclass
class PPOAtariArgs(PPOArgs):
    env_id: str = Arg(default="PongNoFrameskip-v4", help="the id of the environment")
    action_repeat: int = Arg(default=4, help="how many actions to repeat")
    frame_stack: int = Arg(default=4, help="how many frames to stack. 0 to disable.")
    screen_size: int = Arg(
        default=64, help="the dimension of the image rendered by the environment (screen_size x screen_size)"
    )
    features_dim: int = Arg(default=512, help="the features dimension after the NatureCNN layer")


@dataclass
class PPOPixelContinuousArgs(PPOAtariArgs):
    env_id: str = Arg(
        default="CarRacing-v2",
        help="the id of the environment. The environment observation space must be a single array, or a dict of arrays.",
    )
