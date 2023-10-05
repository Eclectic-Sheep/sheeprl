from typing import Any, Dict, Optional, Sequence, Tuple, Union

import gymnasium
import hydra
import torch
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule

from sheeprl.algos.dreamer_v1.agent import WorldModel
from sheeprl.algos.dreamer_v1.agent import build_models as dv1_build_models
from sheeprl.algos.dreamer_v2.agent import Actor as DV2Actor
from sheeprl.algos.dreamer_v2.agent import MinedojoActor as DV2MinedojoActor
from sheeprl.models.models import MLP
from sheeprl.utils.utils import init_weights

# In order to use the hydra.utils.get_class method, in this way the user can
# specify in the configs the name of the class without having to know where
# to go to retrieve the class
Actor = DV2Actor
MinedojoActor = DV2MinedojoActor


def build_models(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    world_model_state: Optional[Dict[str, torch.Tensor]] = None,
    actor_task_state: Optional[Dict[str, torch.Tensor]] = None,
    critic_task_state: Optional[Dict[str, torch.Tensor]] = None,
    actor_exploration_state: Optional[Dict[str, torch.Tensor]] = None,
    critic_exploration_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, _FabricModule, _FabricModule]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        actions_dim (Sequence[int]): the dimension of the actions.
        is_continuous (bool): whether or not the actions are continuous.
        cfg (DictConfig): the hyper-parameters of DreamerV1.
        obs_space (Dict[str, Any]): the observation space.
        world_model_state (Dict[str, Tensor], optional): the state of the world model.
            Default to None.
        actor_task_state (Dict[str, Tensor], optional): the state of the actor_task.
            Default to None.
        critic_task_state (Dict[str, Tensor], optional): the state of the critic_task.
            Default to None.
        actor_exploration_state (Dict[str, Tensor], optional): the state of the actor_exploration.
            Default to None.
        critic_exploration_state (Dict[str, Tensor], optional): the state of the critic_exploration.
            Default to None.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and
        reward models and the continue model.
        The actor_task (_FabricModule).
        The critic_task (_FabricModule).
        The actor_exploration (_FabricModule).
        The critic_exploration (_FabricModule).
    """
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic

    # Sizes
    latent_state_size = world_model_cfg.stochastic_size + world_model_cfg.recurrent_model.recurrent_state_size

    # Create exploration models
    world_model, actor_exploration, critic_exploration = dv1_build_models(
        fabric,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        cfg=cfg,
        obs_space=obs_space,
        world_model_state=world_model_state,
        actor_state=actor_exploration_state,
        critic_state=critic_exploration_state,
    )
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
        layer_norm=False,
    )
    critic_task = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
        activation=eval(critic_cfg.dense_act),
        flatten_dim=None,
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

    return world_model, actor_task, critic_task, actor_exploration, critic_exploration
