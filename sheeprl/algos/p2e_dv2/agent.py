import copy
from typing import Any, Dict, Optional, Sequence, Tuple

from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn

from sheeprl.algos.dreamer_v2.agent import Actor, MinedojoActor, WorldModel
from sheeprl.algos.dreamer_v2.agent import build_models as dv2_build_models
from sheeprl.algos.p2e_dv2.args import P2EDV2Args
from sheeprl.models.models import MLP
from sheeprl.utils.utils import init_weights


def build_models(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    args: P2EDV2Args,
    obs_space: Dict[str, Any],
    cnn_keys: Sequence[str],
    mlp_keys: Sequence[str],
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_task_state: Optional[Dict[str, Tensor]] = None,
    critic_task_state: Optional[Dict[str, Tensor]] = None,
    target_critic_task_state: Optional[Dict[str, Tensor]] = None,
    actor_exploration_state: Optional[Dict[str, Tensor]] = None,
    critic_exploration_state: Optional[Dict[str, Tensor]] = None,
    target_critic_exploration_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, nn.Module, _FabricModule, _FabricModule, nn.Module]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        action_dim (int): the dimension of the actions.
        observation_shape (Tuple[int, ...]): the shape of the observations.
        is_continuous (bool): whether or not the actions are continuous.
        args (P2EDV2Args): the hyper-parameters of Dreamer_v1.
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
        The target_critic_task (nn.Module).
        The actor_exploration (_FabricModule).
        The critic_exploration (_FabricModule).
        The target_critic_exploration (nn.Module).
    """
    if args.cnn_channels_multiplier <= 0:
        raise ValueError(f"cnn_channels_multiplier must be greater than zero, given {args.cnn_channels_multiplier}")
    if args.dense_units <= 0:
        raise ValueError(f"dense_units must be greater than zero, given {args.dense_units}")
    try:
        dense_act = getattr(nn, args.dense_act)
    except AttributeError:
        raise ValueError(
            f"Invalid value for dense_act, given {args.dense_act}, "
            "must be one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        )

    # Sizes
    stochastic_size = args.stochastic_size * args.discrete_size
    latent_state_size = stochastic_size + args.recurrent_state_size

    # Create exploration models
    world_model, actor_exploration, critic_exploration, target_critic_exploration = dv2_build_models(
        fabric,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        args=args,
        obs_space=obs_space,
        cnn_keys=cnn_keys,
        mlp_keys=mlp_keys,
        world_model_state=world_model_state,
        actor_state=actor_exploration_state,
        critic_state=critic_exploration_state,
        target_critic_state=target_critic_exploration_state,
    )

    # Create task models
    if "minedojo" in args.env_id:
        actor_task = MinedojoActor(
            latent_state_size,
            actions_dim,
            is_continuous,
            args.actor_init_std,
            args.actor_min_std,
            args.dense_units,
            dense_act,
            args.mlp_layers,
            distribution=args.actor_distribution,
            layer_norm=args.layer_norm,
        )
    else:
        actor_task = Actor(
            latent_state_size,
            actions_dim,
            is_continuous,
            args.actor_init_std,
            args.actor_min_std,
            args.dense_units,
            dense_act,
            args.mlp_layers,
            distribution=args.actor_distribution,
            layer_norm=args.layer_norm,
        )
    critic_task = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
        norm_layer=[nn.LayerNorm for _ in range(args.mlp_layers)] if args.layer_norm else None,
        norm_args=[{"normalized_shape": args.dense_units} for _ in range(args.mlp_layers)] if args.layer_norm else None,
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

    return (
        world_model,
        actor_task,
        critic_task,
        target_critic_task,
        actor_exploration,
        critic_exploration,
        target_critic_exploration,
    )
