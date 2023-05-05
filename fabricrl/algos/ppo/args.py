from dataclasses import dataclass

from fabricrl.algos.args import StandardArgs
from fabricrl.utils.parser import Arg


@dataclass
class PPOArgs(StandardArgs):
    share_data: bool = Arg(default=False, help="Toggle sharing data between processes")
    per_rank_batch_size: int = Arg(default=64, help="the batch size for each rank")
    total_steps: int = Arg(default=2**16, help="total timesteps of the experiments")
    rollout_steps: int = Arg(default=128, help="the number of steps to run in each environment per policy rollout")
    capture_video: bool = Arg(
        default=False, help="whether to capture videos of the agent performances (check out `videos` folder)"
    )
    mask_vel: bool = Arg(default=False, help="whether to mask velocities")
    lr: float = Arg(default=1e-3, help="the learning rate of the optimizer")
    anneal_lr: bool = Arg(default=False, help="Toggle learning rate annealing for policy and value networks")
    gamma: float = Arg(default=0.99, help="the discount factor gamma")
    gae_lambda: float = Arg(default=0.95, help="the lambda for the general advantage estimation")
    update_epochs: int = Arg(default=10, help="the K epochs to update the policy")
    envs_batch_size: int = Arg(default=2, help="the number of environments to be batched during a single PPO epoch")
    normalize_advantages: bool = Arg(default=False, help="Toggles advantages normalization")
    clip_coef: float = Arg(default=0.2, help="the surrogate clipping coefficient")
    clip_vloss: bool = Arg(
        default=False, help="Toggles whether or not to use a clipped loss for the value function, as per the paper."
    )
    ent_coef: float = Arg(default=0.0, help="coefficient of the entropy")
    vf_coef: float = Arg(default=1.0, help="coefficient of the value function")
    max_grad_norm: float = Arg(default=0.5, help="the maximum norm for the gradient clipping")
