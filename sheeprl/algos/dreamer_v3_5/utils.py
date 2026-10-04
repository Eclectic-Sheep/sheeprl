from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.dreamer_v3.utils import Moments, prepare_obs, test  # noqa: F401
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/replay_value_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Actor/entropy",
    "Values/return",
    "Values/value",
    "Grads/agent",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic", "moments"}


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    from sheeprl.algos.dreamer_v3_5.dreamer_v3_5 import DreamerV3_5
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    return log_trained_models(
        fabric,
        env,
        cfg,
        state,
        DreamerV3_5(fabric, cfg.to_log),
        lambda trained: {
            "world_model": trained.world_model,
            "actor": trained.actor,
            "critic": trained.critic,
            "target_critic": trained.target_critic,
            "moments": trained.moments,
        },
    )
