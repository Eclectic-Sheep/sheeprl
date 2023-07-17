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
from sheeprl.algos.dreamer_v2.agent import Actor, MinedojoActor, RecurrentModel, WorldModel
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state, init_weights
from sheeprl.algos.dreamer_v3.args import DreamerV3Args
from sheeprl.models.models import CNN, MLP, DeCNN
from sheeprl.utils.distribution import TruncatedNormal
from sheeprl.utils.model import Conv2dSame, LayerNormChannelLast, ModuleType


class MultiEncoder(nn.Module):
    def __init__(
        self,
        obs_space: Dict[str, Any],
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        cnn_channels_multiplier: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        cnn_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.SiLU,
        mlp_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.SiLU,
        device: Union[str, torch.device] = "cpu",
        layer_norm: bool = False,
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
                    cnn_layer=Conv2dSame,
                    layer_args={"kernel_size": 4, "stride": 2},
                    activation=cnn_act,
                    norm_layer=[LayerNormChannelLast for _ in range(4)] if layer_norm else None,
                    norm_args=[{"normalized_shape": (2**i) * cnn_channels_multiplier} for i in range(4)]
                    if layer_norm
                    else None,
                ),
                nn.Flatten(-3, -1),
            )
            with torch.no_grad():
                self.cnn_output_dim = self.cnn_encoder(torch.zeros(1, *self.cnn_input_dim)).shape[-1]
        else:
            self.cnn_output_dim = 0

        if self.mlp_keys != []:
            self.mlp_encoder = MLP(
                self.mlp_input_dim,
                None,
                [dense_units] * mlp_layers,
                activation=mlp_act,
                norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
                norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
            )
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
        latent_state_size: int,
        cnn_decoder_input_dim: int,
        cnn_decoder_output_dim: Tuple[int, int, int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        cnn_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.SiLU,
        mlp_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.SiLU,
        device: Union[str, torch.device] = "cpu",
        layer_norm: bool = False,
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
                nn.Unflatten(1, (-1, 4, 4)),
                DeCNN(
                    input_channels=8 * cnn_channels_multiplier,
                    hidden_channels=(torch.tensor([4, 2, 1]) * cnn_channels_multiplier).tolist()
                    + [cnn_decoder_output_dim[0]],
                    layer_args={"kernel_size": 4, "stride": 2, "padding": 1},
                    activation=[cnn_act, cnn_act, cnn_act, None],
                    norm_layer=[LayerNormChannelLast for _ in range(3)] + [None] if layer_norm else None,
                    norm_args=[{"normalized_shape": (2 ** (4 - i - 2)) * cnn_channels_multiplier} for i in range(3)]
                    + [None]
                    if layer_norm
                    else None,
                ),
            )
        if self.mlp_keys != []:
            self.mlp_decoder = MLP(
                latent_state_size,
                None,
                [dense_units] * mlp_layers,
                activation=mlp_act,
                norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
                norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
            )
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.mlp_splits])

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
            reconstructed_obs.update({k: head(mlp_out) for k, head in zip(self.mlp_keys, self.mlp_heads)})
        return reconstructed_obs


class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multu-layer perceptron to predict the stochastic part of the latent state.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
    """

    def __init__(
        self,
        recurrent_model: nn.Module,
        representation_model: nn.Module,
        transition_model: nn.Module,
        discrete: int = 32,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.discrete = discrete
        self.unimix = unimix

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

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = torch.log(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

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
        logits: Tensor = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

    def _transition(self, recurrent_out: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits: Tensor = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
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


def build_models(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    args: DreamerV3Args,
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
        actor = MinedojoActor(
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
        actor = Actor(
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
    critic = MLP(
        input_dims=stochastic_size + args.recurrent_state_size,
        output_dim=1,
        hidden_sizes=[args.dense_units] * args.mlp_layers,
        activation=dense_act,
        flatten_dim=None,
        norm_layer=[nn.LayerNorm for _ in range(args.mlp_layers)] if args.layer_norm else None,
        norm_args=[{"normalized_shape": args.dense_units} for _ in range(args.mlp_layers)] if args.layer_norm else None,
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
