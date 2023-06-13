from dataclasses import dataclass
from typing import Optional

from sheeprl.algos.ppo.args import PPOArgs
from sheeprl.utils.parser import Arg


@dataclass
class RecurrentPPOArgs(PPOArgs):
    share_data: bool = Arg(default=False, help="Toggle sharing data between processes")
    per_rank_batch_size: int = Arg(default=64, help="the sequence length for each rank")
    per_rank_num_batches: int = Arg(
        default=4, help="the number of sequences in a single batch during a single PPO training epoch"
    )
    reset_recurrent_state_on_done: bool = Arg(
        default=False, help="If present the recurrent state will be reset when a done is received"
    )
    lstm_hidden_size: int = Arg(default=64, help="the dimension of the LSTM hidden size")
    actor_pre_lstm_hidden_size: Optional[int] = Arg(
        default=64,
        help="the dimension of the hidden sizes of the pre-lstm single-layer actor network. "
        "If None, no pre-lstm network will be used",
    )
    critic_pre_lstm_hidden_size: Optional[int] = Arg(
        default=64,
        help="the dimension of the hidden sizes of the pre-lstm single-layer critic network. "
        "If None, no pre-lstm network will be used",
    )
