from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.distributions import Independent, Normal, OneHotCategorical, TanhTransform, TransformedDistribution

from sheeprl.algos.dreamer_v1.args import DreamerV1Args
from sheeprl.algos.dreamer_v1.utils import cnn_forward, compute_stochastic_state, init_weights
from sheeprl.models.models import CNN, MLP, DeCNN
from sheeprl.utils.model import ArgsType, ModuleType


class Encoder(nn.Module):
    """The wrapper class for the encoder.

    Args:
        input_channels (int): the number of channels in input.
        hidden_channels (Sequence[int]): the hidden channels of the CNN.
        layer_args (ArgsType): the args of the layers of the CNN.
        activation (Optional[Union[ModuleType, Sequence[ModuleType]]]): the activation function to use in the CNN.
            Default nn.ReLU.
        observation_shape (Tuple[int, int, int]): the dimension of the observations, channels first.
            Default to (3, 64, 64).
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        layer_args: ArgsType,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        observation_shape: Tuple[int, int, int] = (3, 64, 64),
    ) -> None:
        super().__init__()
        self._module = nn.Sequential(
            CNN(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                layer_args=layer_args,
                activation=activation,
            ),
            nn.Flatten(-3, -1),
        )
        self._observation_shape = observation_shape
        with torch.no_grad():
            self._output_size = self._module(torch.zeros(*observation_shape)).shape[-1]

    @property
    def output_size(self) -> None:
        return self._output_size

    def forward(self, x) -> Tensor:
        return self._module(x)


class RecurrentModel(nn.Module):
    """
    Recurrent model for the model-base Dreamer agent.

    Args:
        input_size (int): the input size of the model.
        recurrent_state_size (int): the size of the recurrent state.
        activation_fn (nn.Module): the activation function.
            Default to ELU.
    """

    def __init__(self, input_size: int, recurrent_state_size: int, activation_fn: nn.Module = nn.ELU) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_size, recurrent_state_size), activation_fn())
        self.rnn = nn.GRU(recurrent_state_size, recurrent_state_size)

    def forward(self, input: Tensor, recurrent_state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the next recurrent state from the latent state (stochastic and recurrent states) and the actions.

        Args:
            input (Tensor): the input tensor composed by the stochastic state and the actions concatenated together.
            recurrent_state (Tensor): the previous recurrent state.

        Returns:
            the computed recurrent output and recurrent state.
        """
        feat = self.mlp(input)
        self.rnn.flatten_parameters()
        out, recurrent_state = self.rnn(feat, recurrent_state)
        return out, recurrent_state


class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a multu-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        transition_model (nn.Module): the transition model described in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
            The model is composed by a multu-layer perceptron to predict the stochastic part of the latent state.
        min_std (float, optional): the minimum value of the standard deviation computed by the transition model.
            Default to 0.1.
    """

    def __init__(
        self,
        recurrent_model: nn.Module,
        representation_model: nn.Module,
        transition_model: nn.Module,
        min_std: Optional[float] = 0.1,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.min_std = min_std

    def dynamic(
        self,
        stochastic_state: Tensor,
        recurrent_state: Tensor,
        action: Tensor,
        embedded_obs: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tensor, Tuple[Tensor, Tensor]]:
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
            The actual stochastic state (Tensor): computed by the representation model from the recurrent state and the embbedded observation.
            The predicted mean and std (Tuple[Tensor, Tensor]): the predicted mean and std of the distribution of the latent state.
        """
        recurrent_out, recurrent_state = self.recurrent_model(
            torch.cat((stochastic_state, action), -1), recurrent_state
        )
        predicted_state_mean_std, _ = self._transition(recurrent_out)
        state_mean_std, stochastic_state = self._representation(recurrent_state, embedded_obs)
        return state_mean_std, recurrent_state, stochastic_state, predicted_state_mean_std

    def _representation(self, recurrent_state: Tensor, embedded_obs: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """Compute the distribution of the stochastic part of the latent state.

        Args:
            recurrent_state (Tensor): the recurrent state of the recurrent model, i.e.,
                what is called h or deterministic state in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            state_mean_std (Tensor, Tensor): the mean and the standard deviation of the distribution of the latent state.
            stochastic_state (Tensor): the sampled stochastic part of the latent state.
        """
        state_mean_std, stochastic_state = compute_stochastic_state(
            self.representation_model(torch.cat((recurrent_state, embedded_obs), -1)),
            event_shape=1,
            min_std=self.min_std,
        )
        return state_mean_std, stochastic_state

    def _transition(self, recurrent_out: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """
        Predict the stochastic part of the latent state (Transition Model).

        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.

        Returns:
            The predicted mean and the standard deviation of the distribution of the latent state (Tensor, Tensor).
            The stochastic part of the latent state (Tensor): the sampled stochastic parts of the latent state predicted by the transition model.
        """
        predicted_mean_std = self.transition_model(recurrent_out)
        return compute_stochastic_state(predicted_mean_std, event_shape=1, min_std=self.min_std)

    def imagination(self, stochastic_state: Tensor, recurrent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One-step imagination of the next latent state.
        It can be used several times to imagine trajectories in the latent space (Transition Model).

        Args:
            stochastic_state (Tensor): the stochastic part of the latent space.
                Shape (batch_size, 1, stochastic_size).
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            actions (Tensor): the actions taken by the agent.
                Shape (batch_size, 1, stochastic_size).

        Returns:
            The imagined stochastic state (Tuple[Tensor, Tensor]): the imagined stochastic state.
            The recurrent state (Tensor).
        """
        recurrent_output, recurrent_state = self.recurrent_model(
            torch.cat((stochastic_state, actions), -1), recurrent_state
        )
        _, imagined_stochastic_state = self._transition(recurrent_output)
        return imagined_stochastic_state, recurrent_state


class Actor(nn.Module):
    """
    The wrapper class of the Dreamer_v1 Actor model.

    Args:
        latent_state_size (int): the dimension of the latent state (stochastic size + recurrent_state_size).
        action_dim (int): the dimension in output of the actor.
            The number of actions if continuous, the dimension of the action if discrete.
        is_continuous (bool): whether or not the actions are continuous.
        mean_scale (float): how much to scale the mean.
            Default to 1.
        init_std (float): the amount to sum to the input of the softplus function for the standard deviation.
            Default to 5.
        min_std (float): the minimum standard deviation for the actions.
            Default to 0.1.
        dense_units (int): the dimension of the hidden dense layers.
            Default to 400.
        dense_act (int): the activation function to apply after the dense layers.
            Default to nn.ELU.
        num_layers (int): the number of MLP layers.
            Default to 4.
    """

    def __init__(
        self,
        latent_state_size: int,
        action_dim: int,
        is_continuous: bool,
        mean_scale: float = 5.0,
        init_std: float = 5.0,
        min_std: float = 1e-4,
        dense_units: int = 400,
        dense_act: nn.Module = nn.ELU,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=action_dim * 2 if is_continuous else action_dim,
            hidden_sizes=[dense_units] * num_layers,
            activation=dense_act,
            flatten_dim=None,
        )
        self.action_dim = action_dim
        self.is_continuous = is_continuous
        self.mean_scale = torch.tensor(mean_scale)
        self.init_std = torch.tensor(init_std)
        self.raw_init_std = torch.log(torch.exp(self.init_std) - 1)
        self.min_std = min_std

    def forward(self, state: Tensor, is_training: bool = True) -> Tensor:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
        """
        out: Tensor = self.model(state)
        if self.is_continuous:
            mean, std = torch.chunk(out, 2, -1)
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = F.softplus(std + self.raw_init_std) + self.min_std
            actions_dist = Normal(mean, std)
            actions_dist = Independent(TransformedDistribution(actions_dist, TanhTransform()), 1)
            if is_training:
                actions = actions_dist.rsample()
            else:
                sample = actions_dist.sample((100,))
                log_prob = actions_dist.log_prob(sample)
                actions = sample[log_prob.argmax(0)].view(1, 1, -1)
        else:
            actions_dist = OneHotCategorical(logits=out)
            if is_training:
                actions = actions_dist.sample() + actions_dist.probs - actions_dist.probs.detach()
            else:
                actions = actions_dist.mode
        return actions


class WorldModel(nn.Module):
    """
    Wrapper class for the World model.

    Args:
        encoder (_FabricModule): the encoder.
        rssm (RSSM): the rssm.
        observation_model (_FabricModule): the observation model.
        reward_model (_FabricModule): the reward model.
        continue_model (_FabricModule, optional): the continue model.
    """

    def __init__(
        self,
        encoder: _FabricModule,
        rssm: RSSM,
        observation_model: _FabricModule,
        reward_model: _FabricModule,
        continue_model: Optional[_FabricModule],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.continue_model = continue_model


class Player(nn.Module):
    """
    The model of the Dreamer_v1 player.

    Args:
        encoder (_FabricModule): the encoder.
        recurrent_model (_FabricModule): the recurrent model.
        representation_model (_FabricModule): the representation model.
        actor (_FabricModule): the actor.
        action_dim (int): the dimension of the actions.
        expl_amout (float): the exploration amout to use during training.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        device (torch.device): the device to work on.
    """

    def __init__(
        self,
        encoder: _FabricModule,
        recurrent_model: _FabricModule,
        representation_model: _FabricModule,
        actor: _FabricModule,
        action_dim: int,
        expl_amount: float,
        num_envs: int,
        stochastic_size: int,
        recurrent_state_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.actor = actor
        self.device = device

        self.expl_amount = expl_amount
        self.action_dim = action_dim
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
        self.num_envs = num_envs

        self.init_states()

    def init_states(self) -> None:
        """
        Initialize the states and the actions for the ended environments.
        """
        self.actions = torch.zeros(1, self.num_envs, self.action_dim, device=self.device)
        self.stochastic_state = torch.zeros(1, self.num_envs, self.stochastic_size, device=self.device)
        self.recurrent_state = torch.zeros(1, self.num_envs, self.recurrent_state_size, device=self.device)

    def get_exploration_action(self, obs: Tensor, is_continuous: bool) -> Tensor:
        """
        Return the actions with a certain amount of noise for exploration.

        Args:
            obs (Tensor): the current observations.
            is_continuous (bool): whether or not the actions are continuous.

        Returns:
            The actions the agent has to perform.
        """
        actions = self.get_greedy_action(obs)
        if is_continuous:
            actions = torch.clip(Normal(actions, self.expl_amount).sample(), -1, 1)
        else:
            sample = OneHotCategorical(logits=torch.zeros_like(actions)).sample().to(self.device)
            actions = torch.where(torch.rand(actions.shape[:1], device=self.device) < self.expl_amount, sample, actions)
        return actions

    def get_greedy_action(self, obs: Tensor, is_training: bool = True) -> Tensor:
        """
        Return the greedy actions.

        Args:
            obs (Tensor): the current observations.
            is_training (bool): whether it is training.
                Default to True.

        Returns:
            The actions the agent has to perform.
        """
        embedded_obs: Tensor = cnn_forward(self.encoder, obs.clone(), obs.shape[-3:], (-1,))
        _, self.recurrent_state = self.recurrent_model(
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        _, self.stochastic_state = compute_stochastic_state(
            self.representation_model(torch.cat((self.recurrent_state, embedded_obs), -1))
        )
        self.actions = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), is_training)
        return self.actions


def build_models(
    fabric: Fabric,
    action_dim: int,
    observation_shape: Tuple[int, ...],
    is_continuous: bool,
    args: DreamerV1Args,
    world_model_state: Dict[str, Tensor] = None,
    actor_state: Dict[str, Tensor] = None,
    critic_state: Dict[str, Tensor] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        action_dim (int): the dimension of the actions.
        observation_shape (Tuple[int, ...]): the shape of the observations.
        is_continuous (bool): whether or not the actions are continuous.
        args (DreamerV1Args): the hyper-parameters of Dreamer_v1.
        world_model_state (Dict[str, Tensor]): the state of the world model.
            Default to None.
        actor_state (Dict[str, Tensor]): the state of the actor.
            Default to None.
        critic_state (Dict[str, Tensor]): the state of the critic.
            Default to None.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and reward models and the continue model.
        The actor (_FabricModule).
        The critic (_FabricModule).
    """
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
    rssm = RSSM(
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
    actor = Actor(
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
    critic = MLP(
        input_dims=args.stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.num_layers,
        activation=dense_act,
        flatten_dim=None,
    )
    actor.apply(init_weights)
    critic.apply(init_weights)

    # Load models from checkpoint
    if world_model_state:
        world_model.load_state_dict(world_model_state)
    if actor_state:
        actor.load_state_dict(actor_state)
    if critic_state:
        critic.load_state_dict(critic_state)

    # Setup models with Fabric
    world_model.encoder = fabric.setup_module(world_model.encoder)
    world_model.observation_model = fabric.setup_module(world_model.observation_model)
    world_model.reward_model = fabric.setup_module(world_model.reward_model)
    world_model.rssm.recurrent_model = fabric.setup_module(world_model.rssm.recurrent_model)
    world_model.rssm.representation_model = fabric.setup_module(world_model.rssm.representation_model)
    world_model.rssm.transition_model = fabric.setup_module(world_model.rssm.transition_model)
    if world_model.continue_model:
        world_model.continue_model = fabric.setup_module(world_model.continue_model)
    actor = fabric.setup_module(actor)
    critic = fabric.setup_module(critic)

    return world_model, actor, critic
