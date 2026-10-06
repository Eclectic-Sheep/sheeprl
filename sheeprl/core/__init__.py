"""The shared core of the SheepRL algorithms: one training loop (`run`) and one interface (`Algorithm`).

See `sheeprl/algos/ppo/ppo.py` for an algorithm written on it.
"""

from sheeprl.core.algorithm import Algorithm, Player, TrainState
from sheeprl.core.cadence import Cadence
from sheeprl.core.collector import Act, Collector, Policy, Writer
from sheeprl.core.evaluation import evaluate, log_models_from_checkpoint
from sheeprl.core.loop import load_trained_state, run
from sheeprl.core.runner import Environment, EnvStep, Episode, GymEnvironment
from sheeprl.core.schedule import TrainSchedule
from sheeprl.core.store import load_replay_buffer, rollout_store
from sheeprl.core.update import all_reduce_gradients, autocast, setup_module, update
from sheeprl.data.store import ReplayStore
from sheeprl.utils.model import ema_

__all__ = [
    "Act",
    "Algorithm",
    "Cadence",
    "Collector",
    "EnvStep",
    "Environment",
    "Episode",
    "GymEnvironment",
    "Player",
    "Policy",
    "ReplayStore",
    "TrainSchedule",
    "TrainState",
    "Writer",
    "all_reduce_gradients",
    "autocast",
    "evaluate",
    "ema_",
    "load_replay_buffer",
    "load_trained_state",
    "log_models_from_checkpoint",
    "rollout_store",
    "run",
    "setup_module",
    "update",
]
