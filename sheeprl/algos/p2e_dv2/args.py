from dataclasses import dataclass

from sheeprl.algos.dreamer_v2.args import DreamerV2Args
from sheeprl.utils.parser import Arg


@dataclass
class P2EDV2Args(DreamerV2Args):
    # override
    hidden_size: int = Arg(default=400, help="the hidden size for the transition and representation model")
    recurrent_state_size: int = Arg(default=400, help="the dimension of the recurrent state")

    # P2E args
    num_ensembles: int = Arg(default=10, help="the number of the ensembles to compute the intrinsic reward")
    ensemble_lr: float = Arg(default=3e-4, help="the learning rate of the optimizer of the ensembles")
    ensemble_eps: float = Arg(default=1e-5, help="the epsilon of the Adam optimizer of the ensembles")
    ensemble_clip_gradients: float = Arg(default=100, help="how much to clip the gradient norms of the ensembles")
    intrinsic_reward_multiplier: float = Arg(default=1, help="how much scale the intrinsic rewards")
    exploration_steps: int = Arg(
        default=int(5e6),
        help="the total number of exploration steps. If this number is "
        "less than the total number of steps, then for `total_steps - exploration_steps` steps the actor will be finetuned. "
        "Otherwise the actor will be learned in a zero-shot setting.",
    )
