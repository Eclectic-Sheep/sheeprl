from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.dreamer_v3.utils import AGGREGATOR_KEYS as AGGREGATOR_KEYS_DV3
from sheeprl.core import Algorithm
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

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


def trained_algorithm(fabric: Fabric, cfg: Dict[str, Any]) -> Algorithm:
    """The algorithm of a P2E-DV3 run (`cfg`), to rebuild its trained models: the exploration, or the finetuning
    without the exploration it started from (its configuration already holds the values of the exploration)."""
    from sheeprl.algos.p2e_dv3.p2e_dv3_exploration import P2EDV3Exploration
    from sheeprl.algos.p2e_dv3.p2e_dv3_finetuning import P2EDV3Finetuning

    if "finetuning" in cfg.algo.name:
        return P2EDV3Finetuning(fabric, cfg)
    return P2EDV3Exploration(fabric, cfg)


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    from sheeprl.core import load_trained_state

    # The models are built as by the training, with its configuration
    algo = trained_algorithm(fabric, cfg.to_log)
    trained = load_trained_state(fabric, cfg.to_log, algo, state, env.observation_space, env.action_space)
    models = {
        "world_model": trained.world_model,
        "actor_task": trained.actor_task,
        "critic_task": trained.critic_task,
        "target_critic_task": trained.target_critic_task,
        "moments_task": trained.moments_task,
    }
    if "exploration" in cfg.to_log.algo.name:
        models["ensembles"] = trained.ensembles
        models["actor_exploration"] = trained.actor_exploration
        for k, critic in trained.critics_exploration.items():
            models[f"critic_exploration_{k}"] = critic.module
            models[f"target_critic_exploration_{k}"] = critic.target_module
            models[f"moments_exploration_{k}"] = critic.moments

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        for name, model in models.items():
            model_info[name] = mlflow.pytorch.log_model(unwrap_fabric(model), artifact_path=name)
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
