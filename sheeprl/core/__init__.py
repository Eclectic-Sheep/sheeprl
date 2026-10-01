"""The shared core of the SheepRL algorithms: one training loop (`run`) and one interface (`Algorithm`).

See `sheeprl/algos/ppo/ppo.py` for an algorithm written on it.
"""

from sheeprl.core.algorithm import Algorithm, Player, TrainState
from sheeprl.core.cadence import Cadence
from sheeprl.core.loop import run
from sheeprl.core.runner import EnvRunner, EnvStep
from sheeprl.core.schedule import TrainSchedule
from sheeprl.core.store import Rollout
from sheeprl.core.update import all_reduce_gradients, autocast, setup_module, update

__all__ = [
    "Algorithm",
    "Cadence",
    "EnvRunner",
    "EnvStep",
    "Player",
    "Rollout",
    "TrainSchedule",
    "TrainState",
    "all_reduce_gradients",
    "autocast",
    "run",
    "setup_module",
    "update",
]
