import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
    TanhTransform,
    TransformedDistribution,
)

from sheeprl.algos.dreamer_v1.utils import cnn_forward
from sheeprl.algos.dreamer_v2.args import DreamerV2Args
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state, init_weights
from sheeprl.models.models import CNN, MLP, DeCNN, LayerNormGRUCell
from sheeprl.utils.distribution import TruncatedNormal
from sheeprl.utils.model import ModuleType


class MultiEncoder(nn.Module):
    def __init__(
        self,
        obs_space: Dict[str, Any],
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        cnn_channels_multiplier: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        cnn_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ELU,
        mlp_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ELU,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.cnn_keys = cnn_keys
        self.mlp_keys = mlp_keys
        self.mlp_input_dim = sum([obs_space[k].shape[0] for k in mlp_keys])
        cnn_input_channels = sum([obs_space[k].shape[0] for k in cnn_keys])
        self.cnn_input_dim = (cnn_input_channels, *obs_space[cnn_keys[0]].shape[1:])
        if self.cnn_keys != []:
            self.cnn_encoder = nn.Sequential(
                CNN(
                    input_channels=cnn_input_channels,
                    hidden_channels=(torch.tensor([1, 2, 4, 8]) * cnn_channels_multiplier).tolist(),
                    layer_args={"kernel_size": 4, "stride": 2},
                    activation=cnn_act,
                ),
                nn.Flatten(-3, -1),
            )
            with torch.no_grad():
                self.cnn_output_dim = self.cnn_encoder(torch.zeros(*self.cnn_input_dim)).shape[-1]
        else:
            self.cnn_output_dim = 0

        if self.mlp_keys != []:
            self.mlp_encoder = MLP(self.mlp_input_dim, None, [dense_units] * mlp_layers, activation=mlp_act)
            self.mlp_output_dim = dense_units
        else:
            self.mlp_output_dim = 0

    def forward(self, obs):
        cnn_out = torch.tensor((), device=self.device)
        mlp_out = torch.tensor((), device=self.device)
        if self.cnn_keys != []:
            cnn_input = torch.cat([obs[k] for k in self.cnn_keys], -3)  # channels dimension
            cnn_out = cnn_forward(self.cnn_encoder, cnn_input, cnn_input.shape[-3:], (-1,))
        if self.mlp_keys != []:
            mlp_input = torch.cat([obs[k] for k in self.mlp_keys], -1).type(torch.float32)
            mlp_out = self.mlp_encoder(mlp_input)
        return torch.cat((cnn_out, mlp_out), -1)


class MultiDecoder(nn.Module):
    def __init__(
        self,
        obs_space: Dict[str, Any],
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        cnn_channels_multiplier: int,
        mlp_output_dim: int,
        latent_state_size: int,
        cnn_decoder_input_dim: int,
        cnn_decoder_output_dim: Tuple[int, int, int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        cnn_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ELU,
        mlp_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ELU,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.mlp_splits = [obs_space[k].shape[0] for k in mlp_keys]
        self.cnn_splits = [obs_space[k].shape[0] for k in cnn_keys]
        self.cnn_keys = cnn_keys
        self.mlp_keys = mlp_keys
        self.cnn_decoder_output_dim = cnn_decoder_output_dim
        if self.cnn_keys != []:
            self.cnn_decoder = nn.Sequential(
                nn.Linear(latent_state_size, cnn_decoder_input_dim),
                nn.Unflatten(1, (cnn_decoder_input_dim, 1, 1)),
                DeCNN(
                    input_channels=cnn_decoder_input_dim,
                    hidden_channels=(torch.tensor([4, 2, 1]) * cnn_channels_multiplier).tolist()
                    + [cnn_decoder_output_dim[0]],
                    layer_args=[
                        {"kernel_size": 5, "stride": 2},
                        {"kernel_size": 5, "stride": 2},
                        {"kernel_size": 6, "stride": 2},
                        {"kernel_size": 6, "stride": 2},
                    ],
                    activation=[cnn_act, cnn_act, cnn_act, None],
                ),
            )
        if self.mlp_keys != []:
            self.mlp_decoder = MLP(latent_state_size, mlp_output_dim, [dense_units] * mlp_layers, activation=mlp_act)

    def forward(self, latent_states: Tensor) -> Dict[str, Tensor]:
        reconstructed_obs = {}
        if self.cnn_keys != []:
            cnn_out = cnn_forward(
                self.cnn_decoder, latent_states, (latent_states.shape[-1],), self.cnn_decoder_output_dim
            )
            reconstructed_obs.update(
                {k: rec_obs for k, rec_obs in zip(self.cnn_keys, torch.split(cnn_out, self.cnn_splits, -3))}
            )
        if self.mlp_keys != []:
            mlp_out = self.mlp_decoder(latent_states)
            reconstructed_obs.update(
                {k: rec_obs for k, rec_obs in zip(self.mlp_keys, torch.split(mlp_out, self.mlp_splits, -1))}
            )
        return reconstructed_obs


class RecurrentModel(nn.Module):
    """
    Recurrent model for the model-base Dreamer agent.

    Args:
        input_size (int): the input size of the model.
        dense_units (int): the number of dense units.
        recurrent_state_size (int): the size of the recurrent state.
        activation_fn (nn.Module): the activation function.
            Default to ELU.
        layer_norm (bool, optional): whether to use the LayerNorm inside the GRU.
            Defaults to True.
    """

    def __init__(
        self, input_size: int, recurrent_state_size: int, dense_units: int, activation_fn: nn.Module = nn.ELU
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_size, dense_units), activation_fn())
        self.rnn = LayerNormGRUCell(dense_units, recurrent_state_size, bias=True, batch_first=False, layer_norm=True)

    def forward(self, input: Tensor, recurrent_state: Tensor) -> Tensor:
        """
        Compute the next recurrent state from the latent state (stochastic and recurrent states) and the actions.

        Args:
            input (Tensor): the input tensor composed by the stochastic state and the actions concatenated together.
            recurrent_state (Tensor): the previous recurrent state.

        Returns:
            the computed recurrent output and recurrent state.
        """
        feat = self.mlp(input)
        out = self.rnn(feat, recurrent_state)
        return out


class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multu-layer perceptron to predict the stochastic part of the latent state.
        min_std (float, optional): the minimum value of the standard deviation computed by the transition model.
            Default to 0.1.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
    """

    def __init__(
        self,
        recurrent_model: nn.Module,
        representation_model: nn.Module,
        transition_model: nn.Module,
        min_std: Optional[float] = 0.1,
        discrete: Optional[int] = 32,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.min_std = min_std
        self.discrete = discrete

    def dynamic(
        self, posterior: Tensor, recurrent_state: Tensor, action: Tensor, embedded_obs: Tensor, is_first: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the prior from the recurrent output.
            Representation model: compute the posterior from the recurrent state and from
                the embedded observations provided by the environment.
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551) and [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

        Args:
            posterior (Tensor): the stochastic state computed by the representation model (posterior). It is expected
                to be of dimension `[stoch_size, self.discrete]`, which by default is `[32, 32]`.
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            action (Tensor): the action taken by the agent.
            embedded_obs (Tensor): the embedded observations provided by the environment.
            is_first (Tensor): if this is the first step in the episode.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model.
            The posterior stochastic state (Tensor): computed by the representation model
            The prior stochastic state (Tensor): computed by the transition model
            The logits of the posterior state (Tensor): computed by the transition model from the recurrent state.
            The logits of the prior state (Tensor): computed by the transition model from the recurrent state.
            from the recurrent state and the embbedded observation.
        """
        action = (1 - is_first) * action
        posterior = (1 - is_first) * posterior.view(*posterior.shape[:-2], -1)
        recurrent_state = (1 - is_first) * recurrent_state
        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        prior_logits, prior = self._transition(recurrent_state)
        posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)
        return recurrent_state, posterior, prior, posterior_logits, prior_logits

    def _representation(self, recurrent_state: Tensor, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_state (Tensor): the recurrent state of the recurrent model, i.e.,
                what is called h or deterministic state in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

    def _transition(self, recurrent_out: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits = self.transition_model(recurrent_out)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

    def imagination(self, prior: Tensor, recurrent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One-step imagination of the next latent state.
        It can be used several times to imagine trajectories in the latent space (Transition Model).

        Args:
            prior (Tensor): the prior state.
            recurrent_state (Tensor): the recurrent state of the recurrent model.
            actions (Tensor): the actions taken by the agent.

        Returns:
            The imagined prior state (Tuple[Tensor, Tensor]): the imagined prior state.
            The recurrent state (Tensor).
        """
        recurrent_state = self.recurrent_model(torch.cat((prior, actions), -1), recurrent_state)
        _, imagined_prior = self._transition(recurrent_state)
        return imagined_prior, recurrent_state


class Actor(nn.Module):
    """
    The wrapper class of the Dreamer_v2 Actor model.

    Args:
        latent_state_size (int): the dimension of the latent state (stochastic size + recurrent_state_size).
        actions_dim (Sequence[int]): the dimension in output of the actor.
            The number of actions if continuous, the dimension of the action if discrete.
        is_continuous (bool): whether or not the actions are continuous.
        init_std (float): the amount to sum to the input of the softplus function for the standard deviation.
            Default to 5.
        min_std (float): the minimum standard deviation for the actions.
            Default to 0.1.
        dense_units (int): the dimension of the hidden dense layers.
            Default to 400.
        dense_act (int): the activation function to apply after the dense layers.
            Default to nn.ELU.
        distribution (str): the distribution for the action. Possible values are: `auto`, `discrete`, `normal`,
            `tanh_normal` and `trunc_normal`. If `auto`, then the distribution will be `discrete` if the
            space is a discrete one, `trunc_normal` otherwise.
            Defaults to `auto`.
    """

    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        is_continuous: bool,
        init_std: float = 0.0,
        min_std: float = 0.1,
        dense_units: int = 400,
        dense_act: nn.Module = nn.ELU,
        mlp_layers: int = 4,
        distribution: str = "auto",
    ) -> None:
        super().__init__()
        self.distribution = distribution.lower()
        if self.distribution not in ("auto", "normal", "tanh_normal", "discrete", "trunc_normal"):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, `normal`, `tanh_normal` and `trunc_normal`. "
                f"Found: {self.distribution}"
            )
        if self.distribution == "discrete" and is_continuous:
            raise ValueError("You have choose a discrete distribution but `is_continuous` is true")
        if self.distribution == "auto":
            if is_continuous:
                self.distribution = "trunc_normal"
            else:
                self.distribution = "discrete"
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=np.sum(actions_dim) * 2 if is_continuous else np.sum(actions_dim),
            hidden_sizes=[dense_units] * mlp_layers,
            activation=dense_act,
            flatten_dim=None,
        )
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.init_std = torch.tensor(init_std)
        self.min_std = min_std

    def forward(self, state: Tensor, is_training: bool = True) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        if self.is_continuous:
            mean, std = torch.chunk(out, 2, -1)
            if self.distribution == "tanh_normal":
                mean = 5 * torch.tanh(mean / 5)
                std = F.softplus(std + self.init_std) + self.min_std
                actions_dist = Normal(mean, std)
                actions_dist = Independent(TransformedDistribution(actions_dist, TanhTransform()), 1)
            elif self.distribution == "normal":
                actions_dist = Normal(mean, std)
                actions_dist = Independent(actions_dist, 1)
            elif self.distribution == "trunc_normal":
                std = 2 * torch.sigmoid((std + self.init_std) / 2) + self.min_std
                dist = TruncatedNormal(torch.tanh(mean), std, -1, 1)
                actions_dist = Independent(dist, 1)
            if is_training:
                actions = actions_dist.rsample()
            else:
                sample = actions_dist.sample((100,))
                log_prob = actions_dist.log_prob(sample)
                actions = sample[log_prob.argmax(0)].view(1, 1, -1)
            actions = [actions]
            actions_dist = [actions_dist]
        else:
            actions_logits = torch.split(out, self.actions_dim, -1)
            actions_dist: List[Distribution] = []
            actions: List[Tensor] = []
            for logits in actions_logits:
                actions_dist.append(OneHotCategoricalStraightThrough(logits=logits))
                if is_training:
                    actions.append(actions_dist[-1].rsample())
                else:
                    actions.append(actions_dist[-1].mode)
        return tuple(actions), tuple(actions_dist)


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
        actions_dim (Sequence[int]): the dimension of the actions.
        expl_amout (float): the exploration amout to use during training.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        device (torch.device): the device to work on.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
            Defaults to 32.
    """

    def __init__(
        self,
        encoder: _FabricModule,
        recurrent_model: _FabricModule,
        representation_model: _FabricModule,
        actor: _FabricModule,
        actions_dim: Sequence[int],
        expl_amount: float,
        num_envs: int,
        stochastic_size: int,
        recurrent_state_size: int,
        device: torch.device,
        discrete_size: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.actor = actor
        self.device = device
        self.expl_amount = expl_amount
        self.actions_dim = actions_dim
        self.stochastic_size = stochastic_size
        self.discrete_size = discrete_size
        self.recurrent_state_size = recurrent_state_size
        self.num_envs = num_envs
        self.init_states()

    def init_states(self) -> None:
        """
        Initialize the states and the actions for the ended environments.
        """
        self.actions = torch.zeros(1, self.num_envs, np.sum(self.actions_dim), device=self.device)
        self.stochastic_state = torch.zeros(
            1, self.num_envs, self.stochastic_size * self.discrete_size, device=self.device
        )
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
            self.actions = torch.cat(actions, -1)
            if self.expl_amount > 0.0:
                self.actions = torch.clip(Normal(self.actions, self.expl_amount).sample(), -1, 1)
            expl_actions = [self.actions]
        else:
            expl_actions = []
            for act in actions:
                sample = OneHotCategorical(logits=torch.zeros_like(act)).sample().to(self.device)
                expl_actions.append(
                    torch.where(torch.rand(act.shape[:1], device=self.device) < self.expl_amount, sample, act)
                )
            self.actions = torch.cat(expl_actions, -1)
        return tuple(expl_actions)

    def get_greedy_action(self, obs: Tensor, is_training: bool = True) -> Sequence[Tensor]:
        """
        Return the greedy actions.

        Args:
            obs (Tensor): the current observations.
            is_training (bool): whether it is training.
                Default to True.

        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.encoder(obs)
        self.recurrent_state = self.recurrent_model(
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        posterior_logits = self.representation_model(torch.cat((self.recurrent_state, embedded_obs), -1))
        stochastic_state = compute_stochastic_state(posterior_logits, discrete=self.discrete_size)
        self.stochastic_state = stochastic_state.view(
            *stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), is_training)
        self.actions = torch.cat(actions, -1)
        return actions


def build_models(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    args: DreamerV2Args,
    obs_space: Dict[str, Any],
    cnn_keys: Sequence[str],
    mlp_keys: Sequence[str],
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_state: Optional[Dict[str, Tensor]] = None,
    critic_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, torch.nn.Module]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        actions_dim (Sequence[int]): the dimension of the actions.
        observation_shape (Tuple[int, ...]): the shape of the observations.
        is_continuous (bool): whether or not the actions are continuous.
        args (DreamerV1Args): the hyper-parameters of Dreamer_v1.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and reward models and the continue model.
        The actor (_FabricModule).
        The critic (_FabricModule).
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
    )
    stochastic_size = args.stochastic_size * args.discrete_size
    recurrent_model = RecurrentModel(np.sum(actions_dim) + stochastic_size, args.recurrent_state_size, args.dense_units)
    representation_model = MLP(
        input_dims=args.recurrent_state_size + encoder.cnn_output_dim + encoder.mlp_output_dim,
        output_dim=stochastic_size,
        hidden_sizes=[args.hidden_size],
        activation=dense_act,
        flatten_dim=None,
    )
    transition_model = MLP(
        input_dims=args.recurrent_state_size,
        output_dim=stochastic_size,
        hidden_sizes=[args.hidden_size],
        activation=dense_act,
        flatten_dim=None,
    )
    rssm = RSSM(
        recurrent_model.apply(init_weights),
        representation_model.apply(init_weights),
        transition_model.apply(init_weights),
        args.min_std,
        args.discrete_size,
    )
    observation_model = MultiDecoder(
        obs_space,
        cnn_keys,
        mlp_keys,
        args.cnn_channels_multiplier,
        encoder.mlp_input_dim,
        args.stochastic_size * args.discrete_size + args.recurrent_state_size,
        encoder.cnn_output_dim,
        encoder.cnn_input_dim,
        args.mlp_layers,
        args.dense_units,
        cnn_act,
        dense_act,
        fabric.device,
    )
    reward_model = MLP(
        input_dims=stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
    )
    if args.use_continues:
        continue_model = MLP(
            input_dims=stochastic_size + args.recurrent_state_size,
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
    actor = Actor(
        stochastic_size + args.recurrent_state_size,
        actions_dim,
        is_continuous,
        args.actor_init_std,
        args.actor_min_std,
        args.dense_units,
        dense_act,
        args.mlp_layers,
    )
    critic = MLP(
        input_dims=stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
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
    target_critic = copy.deepcopy(critic.module)

    return world_model, actor, critic, target_critic
