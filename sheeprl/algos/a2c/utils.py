from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
from lightning import Fabric

from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/value_loss", "Loss/policy_loss"}
MODELS_TO_REGISTER = {"agent"}


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    from sheeprl.algos.a2c.a2c import A2C
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    return log_trained_models(
        fabric, env, cfg, state, A2C(fabric, cfg.to_log), lambda trained: {"agent": trained.agent}
    )
