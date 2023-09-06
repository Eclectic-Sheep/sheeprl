import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

from sheeprl.utils.parser import Arg


@dataclass
class StandardArgs:
    exp_name: str = Arg(default="default", help="the name of this experiment")
    seed: int = Arg(default=42, help="seed of the experiment")
    dry_run: bool = Arg(default=False, help="whether to dry-run the script and exit")
    torch_deterministic: bool = Arg(default=False, help="if toggled, " "`torch.backends.cudnn.deterministic=True`")
    env_id: str = Arg(default="CartPole-v1", help="the id of the environment")
    num_envs: int = Arg(default=4, help="the number of parallel game environments")
    sync_env: bool = Arg(default=False, help="whether to use SyncVectorEnv instead of AsyncVectorEnv")
    root_dir: Optional[str] = Arg(
        default=None,
        help="the name of the root folder of the log directory of this experiment",
    )
    run_name: Optional[str] = Arg(default=None, help="the folder name of this run")
    action_repeat: int = Arg(default=1, help="the number of action repeat")
    memmap_buffer: bool = Arg(
        default=False,
        help="whether to move the buffer to the shared memory. "
        "Useful for pixel-based off-policy methods with large buffer size (>=1e6).",
    )
    checkpoint_every: int = Arg(default=100, help="how often to make the checkpoint, -1 to deactivate the checkpoint")
    checkpoint_path: Optional[str] = Arg(default=None, help="the path of the checkpoint from which you want to restart")

    screen_size: int = Arg(default=64, help="the size of the pixel-from observations (if any)")
    frame_stack: int = Arg(default=-1, help="how many frame to stack (only for pixel-like observations)")
    frame_stack_dilation: int = Arg(default=1, help="the dilation between the stacked frames, 1 no dilation")
    max_episode_steps: int = Arg(
        default=-1,
        help="the maximum duration in terms of number of steps of an episode, -1 to disable. "
        "This value will be divided by the `action_repeat` value during the environment creation.",
    )

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)
        if __name == "log_dir":
            file_name = os.path.join(__value, "args.json")
            os.makedirs(__value, exist_ok=True)
            json.dump(asdict(self), open(file_name, "w"))
