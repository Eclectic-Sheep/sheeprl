from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.dreamer_v1.utils import AGGREGATOR_KEYS as AGGREGATOR_KEYS_DV1
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss_task",
    "Loss/policy_loss_task",
    "Loss/value_loss_exploration",
    "Loss/policy_loss_exploration",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "Loss/ensemble_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Params/exploration_amount_task",
    "Params/exploration_amount_exploration",
    "Rewards/intrinsic",
    "Values_exploration/predicted_values",
    "Values_exploration/lambda_values",
    "Grads/world_model",
    "Grads/actor_task",
    "Grads/critic_task",
    "Grads/actor_exploration",
    "Grads/critic_exploration",
    "Grads/ensemble",
}.union(AGGREGATOR_KEYS_DV1)
MODELS_TO_REGISTER = {
    "world_model",
    "ensembles",
    "actor_exploration",
    "critic_exploration",
    "actor_task",
    "critic_task",
}


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    from sheeprl.algos.p2e_dv1.p2e_dv1_exploration import P2EDV1Exploration
    from sheeprl.algos.p2e_dv1.p2e_dv1_finetuning import P2EDV1Finetuning
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    exploration = "exploration" in cfg.to_log.algo.name

    def models(trained: Any) -> Dict[str, Any]:
        models = {
            "world_model": trained.world_model,
            "actor_task": trained.actor_task,
            "critic_task": trained.critic_task,
        }
        if exploration:
            models["ensembles"] = trained.ensembles
            models["actor_exploration"] = trained.actor_exploration
            models["critic_exploration"] = trained.critic_exploration
        return models

    algorithm = P2EDV1Exploration if exploration else P2EDV1Finetuning
    return log_trained_models(fabric, env, cfg, state, algorithm(fabric, cfg.to_log), models)
