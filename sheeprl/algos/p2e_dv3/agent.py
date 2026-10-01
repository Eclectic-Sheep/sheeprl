import copy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import hydra
import torch
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from lightning.pytorch.utilities.seed import isolate_rng
from torch import nn

from sheeprl.algos.dreamer_v3.agent import Actor as DV3Actor
from sheeprl.algos.dreamer_v3.agent import MinedojoActor as DV3MinedojoActor
from sheeprl.algos.dreamer_v3.agent import PlayerDV3, WorldModel
from sheeprl.algos.dreamer_v3.agent import build_agent as dv3_build_agent
from sheeprl.algos.dreamer_v3.agent import build_models as dv3_build_models
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
from sheeprl.models.models import MLP
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.utils import unwrap_fabric

# In order to use the hydra.utils.get_class method, in this way the user can
# specify in the configs the name of the class without having to know where
# to go to retrieve the class
Actor = DV3Actor
MinedojoActor = DV3MinedojoActor


def build_exploration_actor(
    actions_dim: Sequence[int], is_continuous: bool, cfg: Dict[str, Any]
) -> Union[Actor, MinedojoActor]:
    """The exploration actor, with the default initialization of PyTorch (see `init_exploration_actor`)."""
    actor_cfg = cfg.algo.actor
    world_model_cfg = cfg.algo.world_model
    latent_state_size = (
        world_model_cfg.stochastic_size * world_model_cfg.discrete_size
        + world_model_cfg.recurrent_model.recurrent_state_size
    )
    actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)
    return actor_cls(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        dense_units=actor_cfg.dense_units,
        activation=hydra.utils.get_class(actor_cfg.dense_act),
        mlp_layers=actor_cfg.mlp_layers,
        distribution_cfg=cfg.distribution,
        layer_norm_cls=hydra.utils.get_class(actor_cfg.layer_norm.cls),
        layer_norm_kw=actor_cfg.layer_norm.kw,
        unimix=cfg.algo.unimix,
    )


def init_exploration_actor(actor_exploration: nn.Module, cfg: Dict[str, Any]) -> None:
    """The initialization of the exploration actor, done after the exploration critics are created."""
    actor_exploration.apply(init_weights)
    if cfg.algo.hafner_initialization:
        actor_exploration.mlp_heads.apply(uniform_init_weights(1.0))


def build_exploration_critics(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The exploration critics with a positive weight, initialized: for each one its `weight`, its `reward_type`
    (`intrinsic` or `task`) and the critic (`module`). At least one must be intrinsic."""
    world_model_cfg = cfg.algo.world_model
    critic_cfg = cfg.algo.critic
    latent_state_size = (
        world_model_cfg.stochastic_size * world_model_cfg.discrete_size
        + world_model_cfg.recurrent_model.recurrent_state_size
    )
    critics_exploration = {}
    intrinsic_critics = 0
    critic_ln_cls = hydra.utils.get_class(critic_cfg.layer_norm.cls)
    for k, v in cfg.algo.critics_exploration.items():
        if v.weight > 0:
            if v.reward_type == "intrinsic":
                intrinsic_critics += 1
            critics_exploration[k] = {
                "weight": v.weight,
                "reward_type": v.reward_type,
                "module": MLP(
                    input_dims=latent_state_size,
                    output_dim=critic_cfg.bins,
                    hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
                    activation=hydra.utils.get_class(critic_cfg.dense_act),
                    flatten_dim=None,
                    layer_args={"bias": critic_ln_cls == nn.Identity},
                    norm_layer=critic_ln_cls,
                    norm_args={
                        **critic_cfg.layer_norm.kw,
                        "normalized_shape": critic_cfg.dense_units,
                    },
                ),
            }
            critics_exploration[k]["module"].apply(init_weights)
            if cfg.algo.hafner_initialization:
                critics_exploration[k]["module"].model[-1].apply(uniform_init_weights(0.0))
    if intrinsic_critics == 0:
        raise RuntimeError("You must specify at least one intrinsic critic (`reward_type='intrinsic'`)")
    return critics_exploration


def build_ensembles(fabric: Fabric, actions_dim: Sequence[int], cfg: Dict[str, Any]) -> nn.ModuleList:
    """The ensembles that predict the next stochastic state, whose disagreement is the intrinsic reward. Each one is
    initialized with its own seed (`seed + i`), without changing the random state of the run."""
    ens_list = []
    cfg_ensembles = cfg.algo.ensembles
    ensembles_ln_cls = hydra.utils.get_class(cfg_ensembles.layer_norm.cls)
    with isolate_rng():
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
                    activation=hydra.utils.get_class(cfg_ensembles.dense_act),
                    flatten_dim=None,
                    layer_args={"bias": ensembles_ln_cls == nn.Identity},
                    norm_layer=ensembles_ln_cls,
                    norm_args={
                        **cfg_ensembles.layer_norm.kw,
                        "normalized_shape": cfg_ensembles.dense_units,
                    },
                ).apply(init_weights)
            )
    return nn.ModuleList(ens_list)


def build_models(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: Dict[str, Any],
) -> Tuple[WorldModel, nn.Module, nn.Module, nn.Module, Dict[str, Dict[str, Any]], nn.ModuleList]:
    """Create all the models of P2E-DV3 with their initial weights, in the same order as `build_agent`. They are not
    set up with Fabric (only the RSSM is moved to the device).

    Returns:
        The world model, the task actor, the task critic, the exploration actor, the exploration critics (see
        `build_exploration_critics`) and the ensembles.
    """
    world_model, actor_task, critic_task = dv3_build_models(fabric.device, actions_dim, is_continuous, cfg, obs_space)
    actor_exploration = build_exploration_actor(actions_dim, is_continuous, cfg)
    critics_exploration = build_exploration_critics(cfg)
    init_exploration_actor(actor_exploration, cfg)
    ensembles = build_ensembles(fabric, actions_dim, cfg)
    return world_model, actor_task, critic_task, actor_exploration, critics_exploration, ensembles


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: Dict[str, Any],
    world_model_state: Optional[Dict[str, torch.Tensor]] = None,
    ensembles_state: Optional[Dict[str, torch.Tensor]] = None,
    actor_task_state: Optional[Dict[str, torch.Tensor]] = None,
    critic_task_state: Optional[Dict[str, torch.Tensor]] = None,
    target_critic_task_state: Optional[Dict[str, torch.Tensor]] = None,
    actor_exploration_state: Optional[Dict[str, torch.Tensor]] = None,
    critics_exploration_state: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[
    WorldModel, nn.ModuleList, _FabricModule, _FabricModule, _FabricModule, _FabricModule, Dict[str, Any], PlayerDV3
]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        action_dim (int): the dimension of the actions.
        is_continuous (bool): whether or not the actions are continuous.
        cfg (Dict[str, Any]): the configs of P2E_DV3.
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
        critics_exploration_state (Dict[str, Dict[str, Any]], optional): the state of the critics_exploration.
            Default to None.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and
            reward models and the continue model.

        The ensembles (_FabricModule): for estimating the intrinsic reward.
        The actor_task (_FabricModule): for learning the task.
        The critic_task (_FabricModule): for predicting the values of the task.
        The target_critic_task (nn.Module): takes a EMA of the critic_task weights.
        The actor_exploration (_FabricModule): for exploring the environment.
        The critics_exploration (_FabricModule): for predicting the values of the exploration.
        The critics_exploration (Dict[str, Dict[str, Any]]): python dictionary containing all the exploration critics.
            The critic is under the 'module' key, whereas, the target critic is under the 'target_critic' key.
    """
    # Create task models
    world_model, actor_task, critic_task, target_critic_task, player = dv3_build_agent(
        fabric,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        cfg=cfg,
        obs_space=obs_space,
        world_model_state=world_model_state,
        actor_state=actor_task_state,
        critic_state=critic_task_state,
        target_critic_state=target_critic_task_state,
    )

    # Create exploration models
    actor_exploration = build_exploration_actor(actions_dim, is_continuous, cfg)
    single_device_fabric = get_single_device_fabric(fabric)
    critics_exploration = build_exploration_critics(cfg)
    for k in critics_exploration:
        if critics_exploration_state:
            critics_exploration[k]["module"].load_state_dict(critics_exploration_state[k]["module"])
        critics_exploration[k]["module"] = fabric.setup_module(critics_exploration[k]["module"])
        critics_exploration[k]["target_module"] = copy.deepcopy(critics_exploration[k]["module"].module)
        if critics_exploration_state:
            critics_exploration[k]["target_module"].load_state_dict(critics_exploration_state[k]["target_module"])
        critics_exploration[k]["target_module"] = single_device_fabric.setup_module(
            critics_exploration[k]["target_module"]
        )
    init_exploration_actor(actor_exploration, cfg)

    # Load exploration models from checkpoint
    if actor_exploration_state:
        actor_exploration.load_state_dict(actor_exploration_state)

    # Setup exploration models with Fabric
    actor_exploration = fabric.setup_module(actor_exploration)

    # Set requires_grad=False for all target critics
    target_critic_task.requires_grad_(False)
    for c in critics_exploration.values():
        c["target_module"].requires_grad_(False)

    # initialize the ensembles with different seeds to be sure they have different weights
    ensembles = build_ensembles(fabric, actions_dim, cfg)
    if ensembles_state:
        ensembles.load_state_dict(ensembles_state)
    for i in range(len(ensembles)):
        ensembles[i] = fabric.setup_module(ensembles[i])

    # Setup player agent
    if cfg.algo.player.actor_type == "exploration":
        fabric_player = get_single_device_fabric(fabric)
        player_actor = unwrap_fabric(actor_exploration)
        player.actor = fabric_player.setup_module(player_actor)
        for agent_p, p in zip(actor_exploration.parameters(), player.actor.parameters()):
            p.data = agent_p.data

    return (
        world_model,
        ensembles,
        actor_task,
        critic_task,
        target_critic_task,
        actor_exploration,
        critics_exploration,
        player,
    )
