from dataclasses import dataclass

from fabricrl.algos.ppo.args import PPOArgs
from fabricrl.utils.parser import Arg


@dataclass
class RecurrentPPOArgs(PPOArgs):
    share_data: bool = Arg(default=False, help="Toggle sharing data between processes")
    per_rank_batch_size: int = Arg(default=64, help="the sequence length for each rank")
    per_rank_num_batches: int = Arg(
        default=4, help="the number of sequences in a single batch during a single PPO training epoch"
    )
