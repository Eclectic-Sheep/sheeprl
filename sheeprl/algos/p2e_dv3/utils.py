from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.dreamer_v3.utils import AGGREGATOR_KEYS as AGGREGATOR_KEYS_DV3
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/policy_loss_task",
    "Loss/value_loss_task",
    "Loss/policy_loss_exploration",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "Loss/ensemble_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor_task",
    "Grads/critic_task",
    "Grads/actor_exploration",
    "Grads/ensemble",
    # General key name for the exploration critics.
    "Loss/value_loss_exploration",
    "Values_exploration/predicted_values",
    "Values_exploration/lambda_values",
    "Grads/critic_exploration",
    "Rewards/intrinsic",
}.union(AGGREGATOR_KEYS_DV3)
MODELS_TO_REGISTER = {
    "world_model",
    "ensembles",
    "actor_exploration",
    "critic_exploration_intrinsic",
    "target_critic_exploration_intrinsic",
    "moments_exploration_intrinsic",
    "critic_exploration_extrinsic",
    "target_critic_exploration_extrinsic",
    "moments_exploration_extrinsic",
    "actor_task",
    "critic_task",
    "target_critic_task",
    "moments_task",
}


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    from sheeprl.algos.p2e_dv3.p2e_dv3_exploration import P2EDV3Exploration
    from sheeprl.algos.p2e_dv3.p2e_dv3_finetuning import P2EDV3Finetuning
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    exploration = "exploration" in cfg.to_log.algo.name

    def models(trained: Any) -> Dict[str, Any]:
        models = {
            "world_model": trained.world_model,
            "actor_task": trained.actor_task,
            "critic_task": trained.critic_task,
            "target_critic_task": trained.target_critic_task,
            "moments_task": trained.moments_task,
        }
        if exploration:
            models["ensembles"] = trained.ensembles
            models["actor_exploration"] = trained.actor_exploration
            for k, critic in trained.critics_exploration.items():
                models[f"critic_exploration_{k}"] = critic["module"]
                models[f"target_critic_exploration_{k}"] = critic["target_module"]
                models[f"moments_exploration_{k}"] = critic["moments"]
        return models

    algorithm = P2EDV3Exploration if exploration else P2EDV3Finetuning
    return log_trained_models(fabric, env, cfg, state, algorithm(fabric, cfg.to_log), models)
