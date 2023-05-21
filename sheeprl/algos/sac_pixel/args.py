from dataclasses import dataclass

from sheeprl.algos.sac.args import SACArgs
from sheeprl.utils.parser import Arg


@dataclass
class SACPixelContinuousArgs(SACArgs):
    env_id: str = Arg(default="CarRacing-v2", help="the id of the environment")
    action_repeat: int = Arg(default=2, help="how many actions to repeat. Must be greater than 0.")
    frame_stack: int = Arg(default=4, help="how many frames to stack. 0 to disable.")
    screen_size: int = Arg(
        default=64, help="the dimension of the image rendered by the environment (screen_size x screen_size)"
    )
    features_dim: int = Arg(
        default=256,
        help="The features dimension after the convolutional layer. "
        "This will also be the dimension of the mlps for both the actor and the critic models",
    )
