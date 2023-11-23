from __future__ import annotations

from typing import Any, Dict, Sequence

import gymnasium as gym
import mlflow
from lightning import Fabric
from mlflow.models.model import ModelInfo

from sheeprl.algos.dreamer_v1.utils import AGGREGATOR_KEYS as AGGREGATOR_KEYS_DV1
from sheeprl.algos.p2e_dv1.agent import build_agent
from sheeprl.utils.utils import unwrap_fabric

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
) -> Sequence[ModelInfo]:
    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    (
        world_model,
        ensembles,
        actor_task,
        critic_task,
        actor_exploration,
        critic_exploration,
    ) = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["ensembles"] if "exploration" in cfg.algo.name else None,
        state["actor_task"],
        state["critic_task"],
        state["actor_exploration"] if "exploration" in cfg.algo.name else None,
        state["critic_exploration"] if "exploration" in cfg.algo.name else None,
    )

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor_task"] = mlflow.pytorch.log_model(unwrap_fabric(actor_task), artifact_path="actor_task")
        model_info["critic_task"] = mlflow.pytorch.log_model(unwrap_fabric(critic_task), artifact_path="critic_task")
        if "exploration" in cfg.algo.name:
            model_info["ensembles"] = mlflow.pytorch.log_model(unwrap_fabric(ensembles), artifact_path="ensembles")
            model_info["actor_exploration"] = mlflow.pytorch.log_model(
                unwrap_fabric(actor_exploration), artifact_path="actor_exploration"
            )
            model_info["critic_exploration"] = mlflow.pytorch.log_model(
                unwrap_fabric(critic_exploration), artifact_path="critic_exploration"
            )
        mlflow.log_dict(cfg.to_log, "config.json")

    return model_info
