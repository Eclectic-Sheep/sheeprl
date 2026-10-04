"""The shared core of the SheepRL algorithms: one training loop (`run`) and one interface (`Algorithm`).

See `sheeprl/algos/ppo/ppo.py` for an algorithm written on it.
"""

from sheeprl.core.algorithm import Algorithm, Player, TrainState
from sheeprl.core.cadence import Cadence
from sheeprl.core.evaluation import evaluate, log_models_from_checkpoint
from sheeprl.core.loop import load_trained_state, run
from sheeprl.core.runner import EnvRunner, EnvStep
from sheeprl.core.schedule import TrainSchedule
from sheeprl.core.store import Rollout, load_replay_buffer
from sheeprl.core.update import all_reduce_gradients, autocast, setup_module, update
from sheeprl.data.store import ReplayStore
from sheeprl.utils.model import ema_

__all__ = [
    "Algorithm",
    "Cadence",
    "EnvRunner",
    "EnvStep",
    "Player",
    "ReplayStore",
    "Rollout",
    "TrainSchedule",
    "TrainState",
    "all_reduce_gradients",
    "autocast",
    "evaluate",
    "ema_",
    "load_replay_buffer",
    "load_trained_state",
    "log_models_from_checkpoint",
    "run",
    "setup_module",
    "update",
]
