from typing import Any, Dict, Sequence

import gymnasium as gym
import mlflow
from lightning import Fabric
from mlflow.models.model import ModelInfo
from torch import nn

from sheeprl.algos.dreamer_v1.utils import AGGREGATOR_KEYS as AGGREGATOR_KEYS_DV1
from sheeprl.algos.p2e_dv1.agent import build_agent
from sheeprl.models.models import MLP
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
        state["actor_task"],
        state["critic_task"],
        state["actor_exploration"] if "exploration" in cfg.algo.name else None,
        state["critic_exploration"] if "exploration" in cfg.algo.name else None,
    )

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

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run_id, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor_task"] = mlflow.pytorch.log_model(unwrap_fabric(actor_task), artifact_path="actor_task")
        model_info["critic_task"] = mlflow.pytorch.log_model(unwrap_fabric(critic_task), artifact_path="critic_task")
        if "exploration" in cfg.algo.name:
            model_info["ensembles"] = mlflow.pytorch.log_model(ensembles, artifact_path="ensembles")
            model_info["actor_exploration"] = mlflow.pytorch.log_model(
                unwrap_fabric(actor_exploration), artifact_path="actor_exploration"
            )
            model_info["critic_exploration"] = mlflow.pytorch.log_model(
                unwrap_fabric(critic_exploration), artifact_path="critic_exploration"
            )

    return model_info
