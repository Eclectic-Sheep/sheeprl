from dataclasses import dataclass

from sheeprl.utils.parser import Arg, DataClassType


@dataclass
class StandardArgs(DataClassType):
    exp_name: str = Arg(default="default", help="the name of this experiment")
    seed: int = Arg(default=42, help="seed of the experiment")
    dry_run: bool = Arg(default=False, help="whether to dry-run the script and exit")
    torch_deterministic: bool = Arg(default=False, help="if toggled, " "`torch.backends.cudnn.deterministic=True`")
    env_id: str = Arg(default="CartPole-v1", help="the id of the environment")
    num_envs: int = Arg(default=4, help="the number of parallel game environments")
