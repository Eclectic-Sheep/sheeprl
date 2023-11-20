from typing import Any, Dict, Sequence

import gymnasium as gym
import mlflow
from lightning import Fabric
from mlflow.models.model import ModelInfo
from torch import nn

from sheeprl.algos.dreamer_v3.utils import AGGREGATOR_KEYS as AGGREGATOR_KEYS_DV3
from sheeprl.algos.dreamer_v3.utils import Moments
from sheeprl.algos.p2e_dv3.agent import build_agent
from sheeprl.models.models import MLP
from sheeprl.utils.utils import unwrap_fabric

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
    "Params/exploration_amount_task",
    "Params/exploration_amount_exploration",
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
) -> Sequence[ModelInfo]:
    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    world_model, actor_task, critic_task, target_critic_task, actor_exploration, critics_exploration = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["actor_task"],
        state["critic_task"],
        state["target_critic_task"],
        state["actor_exploration"] if "exploration" in cfg.algo.name else None,
        state["critics_exploration"] if "exploration" in cfg.algo.name else None,
    )
    moments_task = Moments(
        fabric,
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments_task.load_state_dict(state["moments_task"])

    if "exploration" in cfg.algo.name:
        ens_list = []
        cfg_ensembles = cfg.algo.ensembles
        for i in range(cfg_ensembles.n):
            fabric.seed_everything(cfg.seed + i)
            ens_list.append(
                MLP(
                    input_dims=int(
                        sum(actions_dim)
                        + cfg.algo.world_model.recurrent_model.recurrent_state_size
                        + cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
                    ),
                    output_dim=cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size,
                    hidden_sizes=[cfg_ensembles.dense_units] * cfg_ensembles.mlp_layers,
                    activation=eval(cfg_ensembles.dense_act),
                    flatten_dim=None,
                    layer_args={"bias": not cfg.algo.ensembles.layer_norm},
                    norm_layer=(
                        [nn.LayerNorm for _ in range(cfg_ensembles.mlp_layers)] if cfg_ensembles.layer_norm else None
                    ),
                    norm_args=(
                        [{"normalized_shape": cfg_ensembles.dense_units} for _ in range(cfg_ensembles.mlp_layers)]
                        if cfg_ensembles.layer_norm
                        else None
                    ),
                )
            )
        ensembles = nn.ModuleList(ens_list)
        ensembles.load_state_dict(state["ensembles"])

        moments_exploration = {
            k: Moments(
                fabric,
                cfg.algo.actor.moments.decay,
                cfg.algo.actor.moments.max,
                cfg.algo.actor.moments.percentile.low,
                cfg.algo.actor.moments.percentile.high,
            )
            for k in critics_exploration.keys()
        }
        for k, m in moments_exploration.items():
            m.load_state_dict(state[f"moments_exploration_{k}"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run_id, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor_task"] = mlflow.pytorch.log_model(unwrap_fabric(actor_task), artifact_path="actor_task")
        model_info["critic_task"] = mlflow.pytorch.log_model(unwrap_fabric(critic_task), artifact_path="critic_task")
        model_info["target_critic_task"] = mlflow.pytorch.log_model(
            target_critic_task, artifact_path="target_critic_task"
        )
        model_info["moments_task"] = mlflow.pytorch.log_model(moments_task, artifact_path="moments_task")
        if "exploration" in cfg.algo.name:
            model_info["ensembles"] = mlflow.pytorch.log_model(ensembles, artifact_path="ensembles")
            model_info["actor_exploration"] = mlflow.pytorch.log_model(
                unwrap_fabric(actor_exploration), artifact_path="actor_exploration"
            )
            for k in critics_exploration.keys():
                model_info[f"critics_exploration_{k}"] = mlflow.pytorch.log_model(
                    critics_exploration[k]["module"], artifact_path=f"critics_exploration_{k}"
                )
                model_info[f"critics_exploration_{k}"] = mlflow.pytorch.log_model(
                    critics_exploration[k]["target_module"], artifact_path=f"critics_exploration_{k}"
                )
                model_info[f"moments_exploration_{k}"] = mlflow.pytorch.log_model(
                    moments_exploration[k], artifact_path=f"moments_exploration_{k}"
                )

    return model_info
