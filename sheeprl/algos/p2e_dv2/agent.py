import copy
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn

from sheeprl.algos.dreamer_v2.agent import (
    RSSM,
    Actor,
    MinedojoActor,
    MultiDecoder,
    MultiEncoder,
    RecurrentModel,
    WorldModel,
)
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
    actor_exploration_state: Optional[Dict[str, Tensor]] = None,
    critic_exploration_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, nn.Module, _FabricModule, _FabricModule, nn.Module, int]:
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
        The world model (WorldModel): composed by the encoder, rssm, observation and reward models and the continue model.
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
        cnn_act = getattr(nn, args.cnn_act)
    except:
        raise ValueError(
            f"Invalid value for cnn_act, given {args.cnn_act}, must be one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        )

    try:
        dense_act = getattr(nn, args.dense_act)
    except:
        raise ValueError(
            f"Invalid value for dense_act, given {args.dense_act}, must be one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        )

    # Define models
    encoder = MultiEncoder(
        obs_space,
        cnn_keys,
        mlp_keys,
        args.cnn_channels_multiplier,
        args.mlp_layers,
        args.dense_units,
        cnn_act,
        dense_act,
        fabric.device,
        args.layer_norm,
    )
    stochastic_size = args.stochastic_size * args.discrete_size
    recurrent_model = RecurrentModel(
        int(np.sum(actions_dim)) + stochastic_size,
        args.recurrent_state_size,
        args.dense_units,
        layer_norm=args.layer_norm,
    )
    representation_model = MLP(
        input_dims=args.recurrent_state_size + encoder.cnn_output_dim + encoder.mlp_output_dim,
        output_dim=stochastic_size,
        hidden_sizes=[args.hidden_size],
        activation=dense_act,
        flatten_dim=None,
        norm_layer=[nn.LayerNorm] if args.layer_norm else None,
        norm_args=[{"normalized_shape": args.hidden_size}] if args.layer_norm else None,
    )
    transition_model = MLP(
        input_dims=args.recurrent_state_size,
        output_dim=stochastic_size,
        hidden_sizes=[args.hidden_size],
        activation=dense_act,
        flatten_dim=None,
        norm_layer=[nn.LayerNorm] if args.layer_norm else None,
        norm_args=[{"normalized_shape": args.hidden_size}] if args.layer_norm else None,
    )
    rssm = RSSM(
        recurrent_model.apply(init_weights),
        representation_model.apply(init_weights),
        transition_model.apply(init_weights),
        args.discrete_size,
    )
    observation_model = MultiDecoder(
        obs_space,
        cnn_keys,
        mlp_keys,
        args.cnn_channels_multiplier,
        args.stochastic_size * args.discrete_size + args.recurrent_state_size,
        encoder.cnn_output_dim,
        encoder.cnn_input_dim,
        args.mlp_layers,
        args.dense_units,
        cnn_act,
        dense_act,
        fabric.device,
        args.layer_norm,
    )
    reward_model = MLP(
        input_dims=stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
        norm_layer=[nn.LayerNorm for _ in range(args.mlp_layers)] if args.layer_norm else None,
        norm_args=[{"normalized_shape": args.dense_units} for _ in range(args.mlp_layers)] if args.layer_norm else None,
    )
    if args.use_continues:
        continue_model = MLP(
            input_dims=stochastic_size + args.recurrent_state_size,
            output_dim=1,
            hidden_sizes=[args.dense_units] * args.mlp_layers,
            activation=dense_act,
            flatten_dim=None,
            norm_layer=[nn.LayerNorm for _ in range(args.mlp_layers)] if args.layer_norm else None,
            norm_args=[{"normalized_shape": args.dense_units} for _ in range(args.mlp_layers)]
            if args.layer_norm
            else None,
        )
    world_model = WorldModel(
        encoder.apply(init_weights),
        rssm,
        observation_model.apply(init_weights),
        reward_model.apply(init_weights),
        continue_model.apply(init_weights) if args.use_continues else None,
    )
    if "minedojo" in args.env_id:
        actor_task = MinedojoActor(
            stochastic_size + args.recurrent_state_size,
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
            stochastic_size + args.recurrent_state_size,
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
        input_dims=stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
        norm_layer=[nn.LayerNorm for _ in range(args.mlp_layers)] if args.layer_norm else None,
        norm_args=[{"normalized_shape": args.dense_units} for _ in range(args.mlp_layers)] if args.layer_norm else None,
    )
    actor_task.apply(init_weights)
    critic_task.apply(init_weights)

    if "minedojo" in args.env_id:
        actor_exploration = MinedojoActor(
            stochastic_size + args.recurrent_state_size,
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
        actor_exploration = Actor(
            stochastic_size + args.recurrent_state_size,
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
    critic_exploration = MLP(
        input_dims=stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
        norm_layer=[nn.LayerNorm for _ in range(args.mlp_layers)] if args.layer_norm else None,
        norm_args=[{"normalized_shape": args.dense_units} for _ in range(args.mlp_layers)] if args.layer_norm else None,
    )
    actor_exploration.apply(init_weights)
    critic_exploration.apply(init_weights)

    # Load models from checkpoint
    if world_model_state:
        world_model.load_state_dict(world_model_state)
    if actor_task_state:
        actor_task.load_state_dict(actor_task_state)
    if critic_task_state:
        critic_task.load_state_dict(critic_task_state)
    if actor_exploration_state:
        actor_exploration.load_state_dict(actor_exploration_state)
    if critic_exploration_state:
        critic_exploration.load_state_dict(critic_exploration_state)

    # Setup models with Fabric
    world_model.encoder = fabric.setup_module(world_model.encoder)
    world_model.observation_model = fabric.setup_module(world_model.observation_model)
    world_model.reward_model = fabric.setup_module(world_model.reward_model)
    world_model.rssm.recurrent_model = fabric.setup_module(world_model.rssm.recurrent_model)
    world_model.rssm.representation_model = fabric.setup_module(world_model.rssm.representation_model)
    world_model.rssm.transition_model = fabric.setup_module(world_model.rssm.transition_model)
    if world_model.continue_model:
        world_model.continue_model = fabric.setup_module(world_model.continue_model)
    actor_task = fabric.setup_module(actor_task)
    critic_task = fabric.setup_module(critic_task)
    actor_exploration = fabric.setup_module(actor_exploration)
    critic_exploration = fabric.setup_module(critic_exploration)
    target_critic_task = copy.deepcopy(critic_task.module)
    target_critic_exploration = copy.deepcopy(critic_exploration.module)

    return (
        world_model,
        actor_task,
        critic_task,
        target_critic_task,
        actor_exploration,
        critic_exploration,
        target_critic_exploration,
    )
