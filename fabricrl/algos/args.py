from dataclasses import dataclass
from typing import Optional

from fabricrl.utils.parser import Arg


@dataclass
class StandardArgs:
    exp_name: Optional[str] = Arg(default="default", help="the name of this experiment")
    seed: Optional[int] = Arg(default=42, help="seed of the experiment")
    torch_deterministic: Optional[bool] = Arg(
        default=False, help="if toggled, " "`torch.backends.cudnn.deterministic=True`"
    )
    env_id: Optional[str] = Arg(default="CartPole-v1", help="the id of the environment")
    num_envs: Optional[int] = Arg(default=4, help="the number of parallel game environments")
