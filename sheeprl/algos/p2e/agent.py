from typing import Dict, Optional, Tuple

import torch
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn

from sheeprl.algos.dreamer_v1.agent import RSSM, Actor, Encoder, RecurrentModel, WorldModel
from sheeprl.algos.dreamer_v1.args import DreamerV1Args
from sheeprl.algos.dreamer_v1.utils import init_weights
from sheeprl.models.models import MLP, DeCNN


class RSSMP2E(RSSM):
    def __init__(
        self,
        recurrent_model: nn.Module,
        representation_model: nn.Module,
        transition_model: nn.Module,
        min_std: Optional[float] = 0.1,
    ) -> None:
        super().__init__(recurrent_model, representation_model, transition_model, min_std)

    def dynamic(
        self, stochastic_state: Tensor, recurrent_state: Tensor, action: Tensor, embedded_obs: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tensor, Tuple[Tensor, Tensor], Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the stochastic state from the recurrent output.
            Representation model: compute the stochasitc state from the recurrent state and from
                the embedded observations provided by the environment.
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551) and [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

        Args:
            stochastic_state (Tensor): the stochastic state.
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            action (Tensor): the action taken by the agent.
            embedded_obs (Tensor): the embedded observations provided by the environment.

        Returns:
            The actual mean and std (Tuple[Tensor, Tensor]): the actual mean and std of the distribution of the latent state.
            The recurrent state (Tuple[Tensor, ...]): the recurrent state of the recurrent model.
            The stochastic state (Tensor): computed by the representation model from the recurrent state and the embbedded observation, aka posterior.
            The predicted mean and std (Tuple[Tensor, Tensor]): the predicted mean and std of the distribution of the latent state.
            The predicted stochastic state (Tensor): computed by the transition model, aka prior.
        """
        recurrent_out, recurrent_state = self.recurrent_model(
            torch.cat((stochastic_state, action), -1), recurrent_state
        )
        predicted_state_mean_std, predicted_stochastic_state = self._transition(recurrent_out)
        state_mean_std, stochastic_state = self._representation(recurrent_state, embedded_obs)
        return state_mean_std, recurrent_state, stochastic_state, predicted_state_mean_std, predicted_stochastic_state


def build_models(
    fabric: Fabric,
    action_dim: int,
    observation_shape: Tuple[int, ...],
    is_continuous: bool,
    args: DreamerV1Args,
    world_model_state: Dict[str, Tensor] = None,
    actor_task_state: Dict[str, Tensor] = None,
    critic_task_state: Dict[str, Tensor] = None,
    actor_exploration_state: Dict[str, Tensor] = None,
    critic_exploration_state: Dict[str, Tensor] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, _FabricModule, _FabricModule]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        action_dim (int): the dimension of the actions.
        observation_shape (Tuple[int, ...]): the shape of the observations.
        is_continuous (bool): whether or not the actions are continuous.
        args (DreamerV1Args): the hyper-parameters of Dreamer_v1.
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
    n_obs_channels = 1 if args.grayscale_obs and "minedojo" not in args.env_id.lower() else 3
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
    encoder = Encoder(
        input_channels=n_obs_channels,
        hidden_channels=(torch.tensor([1, 2, 4, 8]) * args.cnn_channels_multiplier).tolist(),
        layer_args={"kernel_size": 4, "stride": 2},
        activation=cnn_act,
        observation_shape=observation_shape,
    )

    recurrent_model = RecurrentModel(action_dim + args.stochastic_size, args.recurrent_state_size)
    representation_model = MLP(
        input_dims=args.recurrent_state_size + encoder.output_size,
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
    rssm = RSSMP2E(
        recurrent_model.apply(init_weights),
        representation_model.apply(init_weights),
        transition_model.apply(init_weights),
        args.min_std,
    )
    observation_model = nn.Sequential(
        nn.Linear(args.stochastic_size + args.recurrent_state_size, encoder.output_size),
        nn.Unflatten(1, (encoder.output_size, 1, 1)),
        DeCNN(
            input_channels=encoder.output_size,
            hidden_channels=(torch.tensor([4, 2, 1]) * args.cnn_channels_multiplier).tolist() + [n_obs_channels],
            layer_args=[
                {"kernel_size": 5, "stride": 2},
                {"kernel_size": 5, "stride": 2},
                {"kernel_size": 6, "stride": 2},
                {"kernel_size": 6, "stride": 2},
            ],
            activation=[cnn_act, cnn_act, cnn_act, None],
        ),
    )
    reward_model = MLP(
        input_dims=args.stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.num_layers,
        activation=dense_act,
        flatten_dim=None,
    )
    if args.use_continues:
        continue_model = MLP(
            input_dims=args.stochastic_size + args.recurrent_state_size,
            output_dim=1,
            hidden_sizes=[args.dense_units] * args.num_layers,
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
    actor_task = Actor(
        args.stochastic_size + args.recurrent_state_size,
        action_dim,
        is_continuous,
        args.actor_mean_scale,
        args.actor_init_std,
        args.actor_min_std,
        args.dense_units,
        dense_act,
        args.num_layers,
    )
    critic_task = MLP(
        input_dims=args.stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.num_layers,
        activation=dense_act,
        flatten_dim=None,
    )
    actor_task.apply(init_weights)
    critic_task.apply(init_weights)

    actor_exploration = Actor(
        args.stochastic_size + args.recurrent_state_size,
        action_dim,
        is_continuous,
        args.actor_mean_scale,
        args.actor_init_std,
        args.actor_min_std,
        args.dense_units,
        dense_act,
        args.num_layers,
    )
    critic_exploration = MLP(
        input_dims=args.stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.num_layers,
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
