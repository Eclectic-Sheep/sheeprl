from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.ppo.utils import AGGREGATOR_KEYS as ppo_aggregator_keys
from sheeprl.algos.ppo.utils import MODELS_TO_REGISTER as ppo_models_to_register
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


AGGREGATOR_KEYS = ppo_aggregator_keys
MODELS_TO_REGISTER = ppo_models_to_register


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    from sheeprl.algos.ppo_recurrent.ppo_recurrent import PPORecurrent
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    return log_trained_models(
        fabric, env, cfg, state, PPORecurrent(fabric, cfg.to_log), lambda trained: {"agent": trained.agent}
    )
