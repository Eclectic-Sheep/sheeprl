import copy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import gymnasium
import hydra
import torch
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from lightning.pytorch.utilities.seed import isolate_rng
from torch import nn

from sheeprl.algos.dreamer_v2.agent import Actor as DV2Actor
from sheeprl.algos.dreamer_v2.agent import MinedojoActor as DV2MinedojoActor
from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.agent import build_agent as dv2_build_agent
from sheeprl.models.models import MLP
from sheeprl.utils.utils import init_weights

# In order to use the hydra.utils.get_class method, in this way the user can
# specify in the configs the name of the class without having to know where
# to go to retrieve the class
Actor = DV2Actor
MinedojoActor = DV2MinedojoActor


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    world_model_state: Optional[Dict[str, torch.Tensor]] = None,
    ensembles_state: Optional[Dict[str, torch.Tensor]] = None,
    actor_task_state: Optional[Dict[str, torch.Tensor]] = None,
    critic_task_state: Optional[Dict[str, torch.Tensor]] = None,
    target_critic_task_state: Optional[Dict[str, torch.Tensor]] = None,
    actor_exploration_state: Optional[Dict[str, torch.Tensor]] = None,
    critic_exploration_state: Optional[Dict[str, torch.Tensor]] = None,
    target_critic_exploration_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, _FabricModule, nn.Module, _FabricModule, _FabricModule, nn.Module]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        action_dim (int): the dimension of the actions.
        is_continuous (bool): whether or not the actions are continuous.
        cfg (DictConfig): the configs of P2E_DV2.
        obs_space (Dict[str, Any]): The observations space of the environment.
        world_model_state (Dict[str, Tensor], optional): the state of the world model.
            Default to None.
        ensembles_state (Dict[str, Tensor], optional): the state of the ensembles.
            Default to None.
        actor_task_state (Dict[str, Tensor], optional): the state of the actor_task.
            Default to None.
        critic_task_state (Dict[str, Tensor], optional): the state of the critic_task.
            Default to None.
        target_critic_task_state (Dict[str, Tensor], optional): the state of the target
            critic_task. Default to None.
        actor_exploration_state (Dict[str, Tensor], optional): the state of the actor_exploration.
            Default to None.
        critic_exploration_state (Dict[str, Tensor], optional): the state of the critic_exploration.
            Default to None.
        target_critic_exploration_state (Dict[str, Tensor], optional): the state of the target
            critic_exploration. Default to None.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and
            reward models and the continue model.
        The ensembles (_FabricModule): for estimating the intrinsic reward.
        The actor_task (_FabricModule): for learning the task.
        The critic_task (_FabricModule): for predicting the values of the task.
        The target_critic_task (nn.Module): takes a EMA of the critic_task weights.
        The actor_exploration (_FabricModule): for exploring the environment.
        The critic_exploration (_FabricModule): for predicting the values of the exploration.
        The target_critic_exploration (nn.Module): takes a EMA of the critic_exploration weights.
    """
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic

    # Sizes
    stochastic_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    latent_state_size = stochastic_size + world_model_cfg.recurrent_model.recurrent_state_size

    # Create exploration models
    world_model, actor_exploration, critic_exploration, target_critic_exploration = dv2_build_agent(
        fabric,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        cfg=cfg,
        obs_space=obs_space,
        world_model_state=world_model_state,
        actor_state=actor_exploration_state,
        critic_state=critic_exploration_state,
        target_critic_state=target_critic_exploration_state,
    )

    # Create task models
    actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)
    actor_task: Union[Actor, MinedojoActor] = actor_cls(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        mlp_layers=actor_cfg.mlp_layers,
        dense_units=actor_cfg.dense_units,
        activation=eval(actor_cfg.dense_act),
        distribution_cfg=cfg.distribution,
        layer_norm=actor_cfg.layer_norm,
    )
    critic_task = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
        activation=eval(critic_cfg.dense_act),
        flatten_dim=None,
        norm_layer=[nn.LayerNorm for _ in range(critic_cfg.mlp_layers)] if critic_cfg.layer_norm else None,
        norm_args=[{"normalized_shape": critic_cfg.dense_units} for _ in range(critic_cfg.mlp_layers)]
        if critic_cfg.layer_norm
        else None,
    )
    actor_task.apply(init_weights)
    critic_task.apply(init_weights)

    # Load task models from checkpoint
    if actor_task_state:
        actor_task.load_state_dict(actor_task_state)
    if critic_task_state:
        critic_task.load_state_dict(critic_task_state)

    # Setup task models with Fabric
    actor_task = fabric.setup_module(actor_task)
    critic_task = fabric.setup_module(critic_task)
    target_critic_task = copy.deepcopy(critic_task.module)
    if target_critic_task_state:
        target_critic_task.load_state_dict(target_critic_task_state)

    # initialize the ensembles with different seeds to be sure they have different weights
    ens_list = []
    with isolate_rng():
        for i in range(cfg.algo.ensembles.n):
            fabric.seed_everything(cfg.seed + i)
            ens_list.append(
                MLP(
                    input_dims=int(
                        sum(actions_dim)
                        + cfg.algo.world_model.recurrent_model.recurrent_state_size
                        + cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
                    ),
                    output_dim=cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size,
                    hidden_sizes=[cfg.algo.ensembles.dense_units] * cfg.algo.ensembles.mlp_layers,
                    activation=eval(cfg.algo.ensembles.dense_act),
                    flatten_dim=None,
                    norm_layer=(
                        [nn.LayerNorm for _ in range(cfg.algo.ensembles.mlp_layers)]
                        if cfg.algo.ensembles.layer_norm
                        else None
                    ),
                    norm_args=(
                        [
                            {"normalized_shape": cfg.algo.ensembles.dense_units}
                            for _ in range(cfg.algo.ensembles.mlp_layers)
                        ]
                        if cfg.algo.ensembles.layer_norm
                        else None
                    ),
                ).apply(init_weights)
            )
    ensembles = nn.ModuleList(ens_list)
    if ensembles_state:
        ensembles.load_state_dict(ensembles_state)
    for i in range(len(ensembles)):
        ensembles[i] = fabric.setup_module(ensembles[i])

    return (
        world_model,
        ensembles,
        actor_task,
        critic_task,
        target_critic_task,
        actor_exploration,
        critic_exploration,
        target_critic_exploration,
    )
