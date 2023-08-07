from typing import Any, Dict, Optional, Sequence, Tuple

from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn

from sheeprl.algos.dreamer_v1.agent import WorldModel
from sheeprl.algos.dreamer_v1.agent import build_models as dv1_build_models
from sheeprl.algos.dreamer_v2.agent import Actor, MinedojoActor
from sheeprl.algos.p2e_dv1.args import P2EDV1Args
from sheeprl.models.models import MLP
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
        dense_act = getattr(nn, args.dense_act)
    except:
        raise ValueError(
            f"Invalid value for dense_act, given {args.dense_act}, must be one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        )

    # Sizes
    latent_state_size = args.stochastic_size + args.recurrent_state_size

    # Create exploration models
    world_model, actor_exploration, critic_exploration = dv1_build_models(
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
    )
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
            distribution="tanh_normal",
            layer_norm=False,
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
            distribution="tanh_normal",
            layer_norm=False,
        )
    critic_task = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
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
