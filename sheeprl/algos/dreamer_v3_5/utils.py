from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.dreamer_v3.utils import Moments, prepare_obs, test  # noqa: F401
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

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
    import mlflow  # noqa

    from sheeprl.algos.dreamer_v3_5.agent import build_agent

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    world_model, actor, critic, target_critic, _ = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["actor"],
        state["critic"],
        state["target_critic"],
    )
    moments = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments.load_state_dict(state["moments"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(
            unwrap_fabric(world_model), name="world_model", serialization_format="pickle"
        )
        model_info["actor"] = mlflow.pytorch.log_model(
            unwrap_fabric(actor), name="actor", serialization_format="pickle"
        )
        model_info["critic"] = mlflow.pytorch.log_model(
            unwrap_fabric(critic), name="critic", serialization_format="pickle"
        )
        model_info["target_critic"] = mlflow.pytorch.log_model(
            unwrap_fabric(target_critic), name="target_critic", serialization_format="pickle"
        )
        model_info["moments"] = mlflow.pytorch.log_model(moments, name="moments", serialization_format="pickle")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
