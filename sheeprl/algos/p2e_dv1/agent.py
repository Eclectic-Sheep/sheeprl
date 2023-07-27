from typing import Any, Dict, Optional, Sequence, Tuple

from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn

from sheeprl.algos.dreamer_v1.agent import RSSM, RecurrentModel, WorldModel
from sheeprl.algos.dreamer_v2.agent import Actor, CNNDecoder, CNNEncoder, MinedojoActor, MLPDecoder, MLPEncoder
from sheeprl.algos.p2e_dv1.args import P2EDV1Args
from sheeprl.models.models import MLP, MultiDecoder, MultiEncoder
from sheeprl.utils.utils import init_weights


def build_models(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    args: P2EDV1Args,
    obs_space: Dict[str, Any],
    cnn_keys: Sequence[str],
    mlp_keys: Sequence[str],
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_task_state: Optional[Dict[str, Tensor]] = None,
    critic_task_state: Optional[Dict[str, Tensor]] = None,
    actor_exploration_state: Optional[Dict[str, Tensor]] = None,
    critic_exploration_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, _FabricModule, _FabricModule]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        actions_dim (Sequence[int]): the dimension of the actions.
        observation_shape (Tuple[int, ...]): the shape of the observations.
        is_continuous (bool): whether or not the actions are continuous.
        args (DreamerV1Args): the hyper-parameters of DreamerV1.
        obs_space (Dict[str, Any]): the observation space.
        cnn_keys (Sequence[str]): the keys of the observation space to encode through the cnn encoder.
        mlp_keys (Sequence[str]): the keys of the observation space to encode through the mlp encoder.
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
        The actor_exploration (_FabricModule).
        The critic_exploration (_FabricModule).
    """
    # minecraft environment does not support grayscale observations
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
    mlp_splits = [obs_space[k].shape[0] for k in mlp_keys]
    cnn_channels = [obs_space[k].shape[0] for k in cnn_keys]
    image_size = obs_space[cnn_keys[0]].shape[-2:]
    cnn_encoder = (
        CNNEncoder(
            cnn_keys,
            cnn_channels,
            image_size,
            args.cnn_channels_multiplier,
            False,
            cnn_act,
        )
        if cnn_keys is not None and len(cnn_keys) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(mlp_keys, mlp_splits, args.mlp_layers, args.dense_units, dense_act, False)
        if mlp_keys is not None and len(mlp_keys) > 0
        else None
    )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder, fabric.device)

    recurrent_model = RecurrentModel(sum(actions_dim) + args.stochastic_size, args.recurrent_state_size)
    representation_model = MLP(
        input_dims=args.recurrent_state_size + encoder.cnn_output_dim + encoder.mlp_output_dim,
        output_dim=args.stochastic_size * 2,
        hidden_sizes=[args.hidden_size],
        activation=dense_act,
        flatten_dim=None,
    )
    transition_model = MLP(
        input_dims=args.recurrent_state_size,
        output_dim=args.stochastic_size * 2,
        hidden_sizes=[args.hidden_size],
        activation=dense_act,
        flatten_dim=None,
    )
    rssm = RSSM(
        recurrent_model.apply(init_weights),
        representation_model.apply(init_weights),
        transition_model.apply(init_weights),
        args.min_std,
    )
    cnn_decoder = (
        CNNDecoder(
            cnn_keys,
            cnn_channels,
            args.cnn_channels_multiplier,
            args.stochastic_size + args.recurrent_state_size,
            cnn_encoder.output_dim,
            image_size,
            cnn_act,
            False,
        )
        if cnn_keys is not None and len(cnn_keys) > 0
        else None
    )
    mlp_decoder = (
        MLPDecoder(
            mlp_keys,
            mlp_splits,
            args.stochastic_size + args.recurrent_state_size,
            args.mlp_layers,
            args.dense_units,
            dense_act,
            False,
        )
        if mlp_keys is not None and len(mlp_keys) > 0
        else None
    )
    observation_model = MultiDecoder(cnn_decoder, mlp_decoder, fabric.device)
    reward_model = MLP(
        input_dims=args.stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
    )
    if args.use_continues:
        continue_model = MLP(
            input_dims=args.stochastic_size + args.recurrent_state_size,
            output_dim=1,
            hidden_sizes=[args.dense_units] * args.mlp_layers,
            activation=dense_act,
            flatten_dim=None,
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
            args.stochastic_size + args.recurrent_state_size,
            actions_dim,
            is_continuous,
            args.actor_init_std,
            args.actor_min_std,
            args.dense_units,
            dense_act,
            args.mlp_layers,
            distribution="tanh_normal",
            layer_norm=False,
        )
    else:
        actor_task = Actor(
            args.stochastic_size + args.recurrent_state_size,
            actions_dim,
            is_continuous,
            args.actor_init_std,
            args.actor_min_std,
            args.dense_units,
            dense_act,
            args.mlp_layers,
            distribution="tanh_normal",
            layer_norm=False,
        )
    critic_task = MLP(
        input_dims=args.stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
    )
    actor_task.apply(init_weights)
    critic_task.apply(init_weights)

    if "minedojo" in args.env_id:
        actor_exploration = MinedojoActor(
            args.stochastic_size + args.recurrent_state_size,
            actions_dim,
            is_continuous,
            args.actor_init_std,
            args.actor_min_std,
            args.dense_units,
            dense_act,
            args.mlp_layers,
            distribution="tanh_normal",
            layer_norm=False,
        )
    else:
        actor_exploration = Actor(
            args.stochastic_size + args.recurrent_state_size,
            actions_dim,
            is_continuous,
            args.actor_init_std,
            args.actor_min_std,
            args.dense_units,
            dense_act,
            args.mlp_layers,
            distribution="tanh_normal",
            layer_norm=False,
        )
    critic_exploration = MLP(
        input_dims=args.stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
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

    return world_model, actor_task, critic_task, actor_exploration, critic_exploration
