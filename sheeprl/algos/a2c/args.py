from dataclasses import dataclass

from sheeprl.algos.args import StandardArgs
from sheeprl.utils.parser import Arg


@dataclass
class A2CArgs(StandardArgs):
    actor_hidden_sizes: int = Arg(default=128, help="the dimension of the hidden sizes of the actor network")
    critic_hidden_sizes: int = Arg(default=128, help="the dimension of the hidden sizes of the critic network")
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
    loss_reduction: str = Arg(
        default="sum", metadata={"choices": ("mean", "sum", "none")}, help="Which reduction to use"
    )
    normalize_advantages: bool = Arg(default=False, help="Toggles advantages normalization")
    ent_coef: float = Arg(default=0.0, help="coefficient of the entropy")
    anneal_ent_coef: bool = Arg(default=False, help="whether to linearly anneal the entropy coefficient to zero")
    vf_coef: float = Arg(default=1.0, help="coefficient of the value function")
    max_grad_norm: float = Arg(default=0.0, help="the maximum norm for the gradient clipping")
    checkpoint_every: int = Arg(default=-1, help="how often to make the checkpoint, -1 to deactivate the checkpoint")
    use_rmsprop: bool = Arg(
        default=True, help="whether to use the Tensorflow modified version version of the RMSProp optimizer"
    )
