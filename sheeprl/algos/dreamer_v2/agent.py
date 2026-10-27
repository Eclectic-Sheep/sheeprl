from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gymnasium
import hydra
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
    TransformedDistribution,
)

from sheeprl.algos.dreamer_policy import DreamerPolicy
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state, init_weights
from sheeprl.core.actions import ActionCodec
from sheeprl.models.io import MultiDecoder, build_encoder
from sheeprl.models.models import CNN, MLP, DeCNN, LayerNormChannelLast, LayerNormGRUCell
from sheeprl.utils.compile import compiled_policy
from sheeprl.utils.distribution import SafeTanhTransform, TruncatedNormal
from sheeprl.utils.utils import dotdict

# The epsilon of the LayerNorms: the one of the `LayerNormalization` of Keras, which the official implementation uses
# (https://github.com/danijar/dreamerv2), also inside its GRU
LAYER_NORM_EPS = 1e-3
from sheeprl.utils.fabric import get_single_device_fabric, setup_module
from sheeprl.utils.model import ModuleType, cnn_forward


def _activation(activation: Union[str, ModuleType]) -> ModuleType:
    return hydra.utils.get_class(activation) if isinstance(activation, str) else activation


def encoder_output_shape(image_shape: Sequence[int], channels_multiplier: int) -> Tuple[int, int, int]:
    """The shape of the feature maps of `CNNEncoder` on images of shape `image_shape`: four convolutions with kernel 4
    and stride 2 (64 x 64 images to 2 x 2 maps) of [1, 2, 4, 8] * `channels_multiplier` channels."""
    _, height, width = image_shape
    for _ in range(4):
        height, width = (height - 4) // 2 + 1, (width - 4) // 2 + 1
    return 8 * channels_multiplier, height, width


class CNNEncoder(nn.Module):
    """The image encoder of DreamerV2 (and DreamerV1), on images of shape `input_shape` (channels, height, width): four
    convolutions with kernel 4 and stride 2, of [1, 2, 4, 8] * `channels_multiplier` channels, optionally normalized
    (`layer_norm`) and activated; the feature maps flattened."""

    def __init__(
        self,
        input_shape: Sequence[int],
        channels_multiplier: int,
        layer_norm: bool = False,
        activation: Union[str, ModuleType] = nn.ELU,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(d) for d in input_shape)
        if len(self.input_shape) != 3:
            raise ValueError(
                f"The image encoder of DreamerV2 takes images (channels, height, width), got {input_shape}"
            )
        self.model = nn.Sequential(
            CNN(
                input_channels=self.input_shape[0],
                hidden_channels=(torch.tensor([1, 2, 4, 8]) * channels_multiplier).tolist(),
                layer_args={"kernel_size": 4, "stride": 2},
                activation=_activation(activation),
                norm_layer=[LayerNormChannelLast for _ in range(4)] if layer_norm else None,
                norm_args=(
                    [{"normalized_shape": (2**i) * channels_multiplier, "eps": LAYER_NORM_EPS} for i in range(4)]
                    if layer_norm
                    else None
                ),
            ),
            nn.Flatten(-3, -1),
        )
        with torch.no_grad():
            self.output_dim = self.model(torch.zeros(1, *self.input_shape)).shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        return cnn_forward(self.model, x, x.shape[-3:], (-1,))


class MLPEncoder(nn.Module):
    """The vector encoder of DreamerV2 (and DreamerV1), on vectors of shape `input_shape`: `mlp_layers` linear layers of
    `dense_units` units, optionally normalized (`layer_norm`) and activated."""

    def __init__(
        self,
        input_shape: Sequence[int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        layer_norm: bool = False,
        activation: Union[str, ModuleType] = nn.ELU,
    ) -> None:
        super().__init__()
        input_shape = tuple(int(d) for d in input_shape)
        if len(input_shape) != 1:
            raise ValueError(f"The vector encoder of DreamerV2 takes vectors, got the shape {input_shape}")
        self.model = MLP(
            input_shape[0],
            None,
            [dense_units] * mlp_layers,
            activation=_activation(activation),
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=(
                [{"normalized_shape": dense_units, "eps": LAYER_NORM_EPS} for _ in range(mlp_layers)]
                if layer_norm
                else None
            ),
        )
        self.output_dim = dense_units

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class CNNDecoder(nn.Module):
    """The almost-exact inverse of `CNNEncoder`: a linear layer from the latent state to the size of the features of the
    encoder, then four transposed convolutions (kernels 5, 5, 6 and 6, stride 2) from 1 x 1 to the 64 x 64 images of
    shape `output_shape`, the last one without normalization and activation."""

    def __init__(
        self,
        latent_size: int,
        output_shape: Sequence[int],
        channels_multiplier: int,
        layer_norm: bool = False,
        activation: Union[str, ModuleType] = nn.ELU,
    ) -> None:
        super().__init__()
        self.output_shape = tuple(int(d) for d in output_shape)
        if self.output_shape[1:] != (64, 64):
            raise ValueError(
                f"The image decoder of DreamerV2 reconstructs 64 x 64 images, got the shape {self.output_shape}"
            )
        cnn_encoder_output_dim = int(np.prod(encoder_output_shape(self.output_shape, channels_multiplier)))
        hidden_channels = [4 * channels_multiplier, 2 * channels_multiplier, channels_multiplier]
        n_hidden = len(hidden_channels)
        activation = _activation(activation)
        self.model = nn.Sequential(
            nn.Linear(latent_size, cnn_encoder_output_dim),
            nn.Unflatten(1, (cnn_encoder_output_dim, 1, 1)),
            DeCNN(
                input_channels=cnn_encoder_output_dim,
                hidden_channels=hidden_channels + [self.output_shape[0]],
                layer_args=[
                    {"kernel_size": 5, "stride": 2},
                    {"kernel_size": 5, "stride": 2},
                    {"kernel_size": 6, "stride": 2},
                    {"kernel_size": 6, "stride": 2},
                ],
                activation=[activation] * n_hidden + [None],
                norm_layer=[LayerNormChannelLast] * n_hidden + [None] if layer_norm else None,
                norm_args=(
                    [{"normalized_shape": c, "eps": LAYER_NORM_EPS} for c in hidden_channels] + [None]
                    if layer_norm
                    else None
                ),
            ),
        )

    def forward(self, latent_states: Tensor) -> Tensor:
        return cnn_forward(self.model, latent_states, (latent_states.shape[-1],), self.output_shape)


class MLPDecoder(nn.Module):
    """The inverse of `MLPEncoder`: `mlp_layers` linear layers of `dense_units` units from the latent state, optionally
    normalized (`layer_norm`) and activated, then a linear head to `output_shape` (its values flat, then reshaped)."""

    def __init__(
        self,
        latent_size: int,
        output_shape: Sequence[int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        layer_norm: bool = False,
        activation: Union[str, ModuleType] = nn.ELU,
    ) -> None:
        super().__init__()
        self.model = MLP(
            latent_size,
            None,
            [dense_units] * mlp_layers,
            activation=_activation(activation),
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=(
                [{"normalized_shape": dense_units, "eps": LAYER_NORM_EPS} for _ in range(mlp_layers)]
                if layer_norm
                else None
            ),
        )
        self.output_shape = tuple(int(d) for d in output_shape)
        self.head = nn.Linear(dense_units, math.prod(self.output_shape))

    def forward(self, latent_states: Tensor) -> Tensor:
        out = self.head(self.model(latent_states))
        return out.reshape(*out.shape[:-1], *self.output_shape)


class RecurrentModel(nn.Module):
    """Recurrent model for the model-base Dreamer-V3 agent.
    This implementation uses the `sheeprl.models.models.LayerNormGRUCell`, which combines
    the standard GRUCell from PyTorch with the `nn.LayerNorm`, where the normalization is applied
    right after having computed the projection from the input to the weight space.

    Args:
        input_size (int): the input size of the model.
        recurrent_state_size (int): the size of the recurrent state.
        dense_units (int): the number of dense units.
        activation (nn.Module): the activation function.
            Default to ELU.
        layer_norm (bool): whether to use the LayerNorm inside the GRU.
            Defaults to True.
    """

    def __init__(
        self,
        input_size: int,
        recurrent_state_size: int,
        dense_units: int,
        activation: nn.Module = nn.ELU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=None,
            hidden_sizes=[dense_units],
            activation=activation,
            norm_layer=[nn.LayerNorm] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units, "eps": LAYER_NORM_EPS}] if layer_norm else None,
        )
        self.rnn = LayerNormGRUCell(
            dense_units,
            recurrent_state_size,
            bias=True,
            batch_first=False,
            layer_norm_cls=nn.LayerNorm,
            layer_norm_kw={"eps": LAYER_NORM_EPS},
        )

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
        recurrent_model (_FabricModule): the recurrent model of the RSSM model described in
        [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (_FabricModule): the representation model composed by a multi-layer perceptron
            to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (_FabricModule): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multu-layer perceptron to predict the stochastic part of the latent state.
        distribution_cfg (dotdict): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
    """

    def __init__(
        self,
        recurrent_model: _FabricModule,
        representation_model: _FabricModule,
        transition_model: _FabricModule,
        distribution_cfg: dotdict,
        discrete: Optional[int] = 32,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.discrete = discrete
        self.distribution_cfg = distribution_cfg

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
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
        and [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

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
                what is called h or deterministic state in
                [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
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
        distribution_cfg (dotdict): The configs of the distributions.
        init_std (float): the amount to sum to the input of the softplus function for the standard deviation.
            Default to 5.
        min_std (float): the minimum standard deviation for the actions.
            Default to 0.1.
        dense_units (int): the dimension of the hidden dense layers.
            Default to 400.
        activation (int): the activation function to apply after the dense layers.
            Default to nn.ELU.
        mlp_layers (int): the number of linear layers.
            Default to 4.
        layer_norm (bool): whether or not to use the layer norm.
            Default to False.
        expl_amount (float): the exploration amount to use during training.
            Default to 0.0.
        expl_decay (float): the exploration decay to use during training.
            Default to 0.0.
        expl_min (float): the exploration amount minimum to use during training.
            Default to 0.0.
    """

    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        is_continuous: bool,
        distribution_cfg: dotdict,
        init_std: float = 0.0,
        min_std: float = 0.1,
        dense_units: int = 400,
        activation: nn.Module = nn.ELU,
        mlp_layers: int = 4,
        layer_norm: bool = False,
        expl_amount: float = 0.0,
        expl_decay: float = 0.0,
        expl_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.distribution_cfg = distribution_cfg
        self.distribution = distribution_cfg.get("type", "auto").lower()
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
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            flatten_dim=None,
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=(
                [{"normalized_shape": dense_units, "eps": LAYER_NORM_EPS} for _ in range(mlp_layers)]
                if layer_norm
                else None
            ),
        )
        if is_continuous:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, sum(actions_dim) * 2)])
        else:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, action_dim) for action_dim in actions_dim])
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.init_std = torch.tensor(init_std)
        self.min_std = min_std
        self.distribution_cfg = distribution_cfg
        self._expl_amount = expl_amount
        self._expl_decay = expl_decay
        self._expl_min = expl_min

    def _get_expl_amount(self, step: int) -> Tensor:
        amount = self._expl_amount
        if self._expl_decay:
            # Halved every `expl_decay` steps
            amount *= 0.5 ** (float(step) / self._expl_decay)
        return max(amount, self._expl_min)

    def forward(
        self, state: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Dict[str, Tensor], optional): the action mask (which actions can be selected).
                Default to None.

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        pre_dist: List[Tensor] = [head(out) for head in self.mlp_heads]
        if self.is_continuous:
            mean, std = torch.chunk(pre_dist[0], 2, -1)
            if self.distribution == "tanh_normal":
                mean = 5 * torch.tanh(mean / 5)
                std = F.softplus(std + self.init_std) + self.min_std
                actions_dist = Normal(mean, std)
                actions_dist = Independent(TransformedDistribution(actions_dist, SafeTanhTransform()), 1)
            elif self.distribution == "normal":
                # The std is the output of the network: made positive as the one of `tanh_normal`
                std = F.softplus(std + self.init_std) + self.min_std
                actions_dist = Normal(mean, std)
                actions_dist = Independent(actions_dist, 1)
            elif self.distribution == "trunc_normal":
                std = 2 * torch.sigmoid((std + self.init_std) / 2) + self.min_std
                dist = TruncatedNormal(torch.tanh(mean), std, -1, 1)
                actions_dist = Independent(dist, 1)
            if not greedy:
                actions = actions_dist.rsample()
            else:
                # The most likely of 100 samples, for every state
                sample = actions_dist.sample((100,))
                best = actions_dist.log_prob(sample).argmax(0, keepdim=True)
                actions = sample.gather(0, best.unsqueeze(-1).expand(1, *sample.shape[1:])).squeeze(0)
            actions = [actions]
            actions_dist = [actions_dist]
        else:
            actions_dist: List[Distribution] = []
            actions: List[Tensor] = []
            for logits in pre_dist:
                actions_dist.append(OneHotCategoricalStraightThrough(logits=logits))
                if not greedy:
                    actions.append(actions_dist[-1].rsample())
                else:
                    actions.append(actions_dist[-1].mode)
        return tuple(actions), tuple(actions_dist)

    def add_exploration_noise(
        self, actions: Sequence[Tensor], step: int = 0, mask: Optional[Dict[str, Tensor]] = None
    ) -> Sequence[Tensor]:
        expl_amount = self._get_expl_amount(step)
        if self.is_continuous:
            actions = torch.cat(actions, -1)
            if expl_amount > 0.0:
                actions = torch.clip(Normal(actions, expl_amount).sample(), -1, 1)
            expl_actions = [actions]
        else:
            expl_actions = []
            for act in actions:
                sample = OneHotCategorical(logits=torch.zeros_like(act)).sample().to(act.device)
                # A random action with probability `expl_amount`, drawn for every environment
                explore = torch.rand(*act.shape[:-1], 1, device=act.device) < expl_amount
                expl_actions.append(torch.where(explore, sample, act))
        return tuple(expl_actions)


class MinedojoActor(Actor):
    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        is_continuous: bool,
        distribution_cfg: dotdict,
        init_std: float = 0,
        min_std: float = 0.1,
        dense_units: int = 400,
        activation: nn.Module = nn.ELU,
        mlp_layers: int = 4,
        layer_norm: bool = False,
        expl_amount: float = 0.0,
        expl_decay: float = 0.0,
        expl_min: float = 0.0,
    ) -> None:
        super().__init__(
            latent_state_size=latent_state_size,
            actions_dim=actions_dim,
            is_continuous=is_continuous,
            distribution_cfg=distribution_cfg,
            init_std=init_std,
            min_std=min_std,
            dense_units=dense_units,
            activation=activation,
            mlp_layers=mlp_layers,
            layer_norm=layer_norm,
            expl_amount=expl_amount,
            expl_decay=expl_decay,
            expl_min=expl_min,
        )

    def forward(
        self, state: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Dict[str, Tensor], optional): the action mask (which actions can be selected).
                Default to None.

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        actions_logits: List[Tensor] = [head(out) for head in self.mlp_heads]
        actions_dist: List[Distribution] = []
        actions: List[Tensor] = []
        functional_action = None
        for i, logits in enumerate(actions_logits):
            if mask is not None:
                if i == 0:
                    logits[torch.logical_not(mask["mask_action_type"].expand_as(logits))] = -torch.inf
                elif i == 1:
                    mask["mask_craft_smelt"] = mask["mask_craft_smelt"].expand_as(logits)
                    for t in range(functional_action.shape[0]):
                        for b in range(functional_action.shape[1]):
                            sampled_action = functional_action[t, b].item()
                            if sampled_action == 15:  # Craft action
                                logits[t, b][torch.logical_not(mask["mask_craft_smelt"][t, b])] = -torch.inf
                elif i == 2:
                    mask["mask_destroy"] = mask["mask_destroy"].expand_as(logits)
                    mask["mask_equip_place"] = mask["mask_equip_place"].expand_as(logits)
                    for t in range(functional_action.shape[0]):
                        for b in range(functional_action.shape[1]):
                            sampled_action = functional_action[t, b].item()
                            if sampled_action in (16, 17):  # Equip/Place action
                                logits[t, b][torch.logical_not(mask["mask_equip_place"][t, b])] = -torch.inf
                            elif sampled_action == 18:  # Destroy action
                                logits[t, b][torch.logical_not(mask["mask_destroy"][t, b])] = -torch.inf
            actions_dist.append(OneHotCategoricalStraightThrough(logits=logits))
            if not greedy:
                actions.append(actions_dist[-1].rsample())
            else:
                actions.append(actions_dist[-1].mode)
            if functional_action is None:
                functional_action = actions[0].argmax(dim=-1)  # [T, B]
        return tuple(actions), tuple(actions_dist)

    def add_exploration_noise(
        self, actions: Sequence[Tensor], step: int = 0, mask: Optional[Dict[str, Tensor]] = None
    ) -> Sequence[Tensor]:
        expl_amount = self._get_expl_amount(step)
        expl_actions = []
        functional_action = actions[0].argmax(dim=-1)
        for i, act in enumerate(actions):
            logits = torch.zeros_like(act)
            # Exploratory action must respect the constraints of the environment
            if mask is not None:
                if i == 0:
                    logits[torch.logical_not(mask["mask_action_type"].expand_as(logits))] = -torch.inf
                elif i == 1:
                    mask["mask_craft_smelt"] = mask["mask_craft_smelt"].expand_as(logits)
                    for t in range(functional_action.shape[0]):
                        for b in range(functional_action.shape[1]):
                            sampled_action = functional_action[t, b].item()
                            if sampled_action == 15:  # Craft action
                                logits[t, b][torch.logical_not(mask["mask_craft_smelt"][t, b])] = -torch.inf
                elif i == 2:
                    mask["mask_destroy"] = mask["mask_destroy"].expand_as(logits)
                    mask["mask_equip_place"] = mask["mask_equip_place"].expand_as(logits)
                    for t in range(functional_action.shape[0]):
                        for b in range(functional_action.shape[1]):
                            sampled_action = functional_action[t, b].item()
                            if sampled_action in {16, 17}:  # Equip/Place action
                                logits[t, b][torch.logical_not(mask["mask_equip_place"][t, b])] = -torch.inf
                            elif sampled_action == 18:  # Destroy action
                                logits[t, b][torch.logical_not(mask["mask_destroy"][t, b])] = -torch.inf
            sample = OneHotCategorical(logits=logits).sample()
            # A random action with probability `expl_amount`, drawn for every environment
            explore = torch.rand(*act.shape[:-1], 1, device=act.device) < expl_amount
            if i in {1, 2}:
                # The environments whose functional action was changed into one that takes arguments (craft, equip,
                # place, destroy) also change its arguments, to satisfy the constraints of the environment
                changed = actions[0].argmax(dim=-1) != expl_actions[0].argmax(dim=-1)
                with_arguments = torch.isin(
                    expl_actions[0].argmax(dim=-1), torch.tensor([15, 16, 17, 18], device=act.device)
                )
                explore = explore | (changed & with_arguments).unsqueeze(-1)
            expl_actions.append(torch.where(explore, sample, act))
            if mask is not None and i == 0:
                functional_action = expl_actions[0].argmax(dim=-1)
        return tuple(expl_actions)


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


class DreamerV2Policy(nn.Module, DreamerPolicy):
    """
    The model of the Dreamer_v2 policy.

    Args:
        encoder (nn.Module | _FabricModule): the encoder.
        recurrent_model (nn.Module | _FabricModule): the recurrent model.
        representation_model (nn.Module | _FabricModule): the representation model.
        actor (nn.Module | _FabricModule): the actor.
        actions_dim (Sequence[int]): the dimension of the actions.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        device (str | torch.device): the device where the model is stored.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
            Defaults to 32.
        actor_type (str, optional): which actor the policy is using ('task' or 'exploration').
            Default to None.
    """

    def __init__(
        self,
        encoder: nn.Module | _FabricModule,
        recurrent_model: nn.Module | _FabricModule,
        representation_model: nn.Module | _FabricModule,
        actor: nn.Module | _FabricModule,
        codec: ActionCodec,
        stochastic_size: int,
        recurrent_state_size: int,
        device: str | torch.device,
        discrete_size: int = 32,
        actor_type: str | None = None,
        keys: Sequence[str] = (),
        mask_keys: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.actor = actor
        self.codec = codec
        # The keys of the inputs of the encoder (by default) and the action masks the actor reads
        self.keys = tuple(keys) or tuple(encoder.keys)
        self.mask_keys = tuple(mask_keys)
        self.actions_dim = codec.dims
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
        self.device = device
        self.discrete_size = discrete_size
        self.actor_type = actor_type

    def init_states(self, num_envs: int) -> None:
        """Zero latent states and actions of `num_envs` environments."""
        self.num_envs = num_envs
        self.actions = torch.zeros(1, num_envs, np.sum(self.actions_dim), device=self.device)
        self.recurrent_state = torch.zeros(1, num_envs, self.recurrent_state_size, device=self.device)
        self.stochastic_state = torch.zeros(1, num_envs, self.stochastic_size * self.discrete_size, device=self.device)

    def reset_state(self, env_idxes: Sequence[int]) -> None:
        self.actions[:, env_idxes] = torch.zeros_like(self.actions[:, env_idxes])
        self.recurrent_state[:, env_idxes] = torch.zeros_like(self.recurrent_state[:, env_idxes])
        self.stochastic_state[:, env_idxes] = torch.zeros_like(self.stochastic_state[:, env_idxes])

    def get_actions(
        self,
        obs: Dict[str, Tensor],
        greedy: bool = False,
        mask: Optional[Dict[str, Tensor]] = None,
    ) -> Sequence[Tensor]:
        """
        Return the greedy actions.

        Args:
            obs (Dict[str, Tensor]): the current observations.
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Dict[str, Tensor], optional): the action mask (which actions can be selected).
                Default to None.

        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.encoder(obs, phase="act")
        self.recurrent_state = self.recurrent_model(
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        posterior_logits = self.representation_model(torch.cat((self.recurrent_state, embedded_obs), -1))
        stochastic_state = compute_stochastic_state(posterior_logits, discrete=self.discrete_size)
        self.stochastic_state = stochastic_state.view(
            *stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), greedy, mask)
        self.actions = torch.cat(actions, -1)
        return actions


def build_agent(
    fabric: Fabric,
    codec: ActionCodec,
    cfg: dotdict,
    obs_space: gymnasium.spaces.Dict,
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_state: Optional[Dict[str, Tensor]] = None,
    critic_state: Optional[Dict[str, Tensor]] = None,
    target_critic_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, _FabricModule, DreamerV2Policy]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        codec (ActionCodec): the actions of the agent and of the environments.
        cfg (dotdict): the configs: the encoder of `algo.inputs`, the decoder of `algo.outputs`.
        obs_space (Dict[str, Any]): the observation space.
        world_model_state (Dict[str, Tensor], optional): the state of the world model.
            Default to None.
        actor_state: (Dict[str, Tensor], optional): the state of the actor.
            Default to None.
        critic_state: (Dict[str, Tensor], optional): the state of the critic.
            Default to None.
        target_critic_state: (Dict[str, Tensor], optional): the state of the critic.
            Default to None.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and
        reward models and the continue model.
        The actor (_FabricModule).
        The critic (_FabricModule).
        The target critic (_FabricModule).
    """
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic
    codec.check("The actor of DreamerV2")
    actions_dim, is_continuous = codec.dims, codec.continuous

    # Sizes
    stochastic_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    latent_state_size = stochastic_size + world_model_cfg.recurrent_model.recurrent_state_size

    # Define models
    encoder = build_encoder(cfg, obs_space)
    recurrent_model = RecurrentModel(
        **world_model_cfg.recurrent_model,
        input_size=int(sum(actions_dim) + stochastic_size),
    )
    representation_model = MLP(
        input_dims=(world_model_cfg.recurrent_model.recurrent_state_size + encoder.output_dim),
        output_dim=stochastic_size,
        hidden_sizes=[world_model_cfg.representation_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.representation_model.dense_act),
        flatten_dim=None,
        norm_layer=[nn.LayerNorm] if world_model_cfg.representation_model.layer_norm else None,
        norm_args=(
            [{"normalized_shape": world_model_cfg.representation_model.hidden_size, "eps": LAYER_NORM_EPS}]
            if world_model_cfg.representation_model.layer_norm
            else None
        ),
    )
    transition_model = MLP(
        input_dims=world_model_cfg.recurrent_model.recurrent_state_size,
        output_dim=stochastic_size,
        hidden_sizes=[world_model_cfg.transition_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.transition_model.dense_act),
        flatten_dim=None,
        norm_layer=[nn.LayerNorm] if world_model_cfg.transition_model.layer_norm else None,
        norm_args=(
            [{"normalized_shape": world_model_cfg.transition_model.hidden_size, "eps": LAYER_NORM_EPS}]
            if world_model_cfg.transition_model.layer_norm
            else None
        ),
    )
    rssm = RSSM(
        recurrent_model=recurrent_model.apply(init_weights),
        representation_model=representation_model.apply(init_weights),
        transition_model=transition_model.apply(init_weights),
        distribution_cfg=cfg.distribution,
        discrete=world_model_cfg.discrete_size,
    )
    observation_model = MultiDecoder.from_config(cfg.algo.outputs, encoder, latent_size=latent_state_size)
    reward_model = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[world_model_cfg.reward_model.dense_units] * world_model_cfg.reward_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.reward_model.dense_act),
        flatten_dim=None,
        norm_layer=(
            [nn.LayerNorm for _ in range(world_model_cfg.reward_model.mlp_layers)]
            if world_model_cfg.reward_model.layer_norm
            else None
        ),
        norm_args=(
            [
                {"normalized_shape": world_model_cfg.reward_model.dense_units, "eps": LAYER_NORM_EPS}
                for _ in range(world_model_cfg.reward_model.mlp_layers)
            ]
            if world_model_cfg.reward_model.layer_norm
            else None
        ),
    )
    if world_model_cfg.use_continues:
        continue_model = MLP(
            input_dims=latent_state_size,
            output_dim=1,
            hidden_sizes=[world_model_cfg.discount_model.dense_units] * world_model_cfg.discount_model.mlp_layers,
            activation=hydra.utils.get_class(world_model_cfg.discount_model.dense_act),
            flatten_dim=None,
            norm_layer=(
                [nn.LayerNorm for _ in range(world_model_cfg.discount_model.mlp_layers)]
                if world_model_cfg.discount_model.layer_norm
                else None
            ),
            norm_args=(
                [
                    {"normalized_shape": world_model_cfg.discount_model.dense_units, "eps": LAYER_NORM_EPS}
                    for _ in range(world_model_cfg.discount_model.mlp_layers)
                ]
                if world_model_cfg.discount_model.layer_norm
                else None
            ),
        )
    world_model = WorldModel(
        encoder.apply(init_weights),
        rssm,
        observation_model.apply(init_weights),
        reward_model.apply(init_weights),
        continue_model.apply(init_weights) if world_model_cfg.use_continues else None,
    )
    actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)
    actor: nn.Module = actor_cls(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        mlp_layers=actor_cfg.mlp_layers,
        dense_units=actor_cfg.dense_units,
        activation=hydra.utils.get_class(actor_cfg.dense_act),
        distribution_cfg=cfg.distribution,
        layer_norm=actor_cfg.layer_norm,
    )
    critic = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
        activation=hydra.utils.get_class(critic_cfg.dense_act),
        flatten_dim=None,
        norm_layer=[nn.LayerNorm for _ in range(critic_cfg.mlp_layers)] if critic_cfg.layer_norm else None,
        norm_args=(
            [{"normalized_shape": critic_cfg.dense_units, "eps": LAYER_NORM_EPS} for _ in range(critic_cfg.mlp_layers)]
            if critic_cfg.layer_norm
            else None
        ),
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

    # Create the policy agent
    policy_fabric = get_single_device_fabric(fabric)
    policy = DreamerV2Policy(
        copy.deepcopy(world_model.encoder),
        copy.deepcopy(world_model.rssm.recurrent_model),
        copy.deepcopy(world_model.rssm.representation_model),
        copy.deepcopy(actor),
        codec,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        policy_fabric.device,
        discrete_size=cfg.algo.world_model.discrete_size,
        keys=encoder.keys,
        mask_keys=cfg.algo.extra_keys,
    )

    # Setup models with Fabric
    world_model.encoder = setup_module(fabric, world_model.encoder)
    world_model.observation_model = setup_module(fabric, world_model.observation_model)
    world_model.reward_model = setup_module(fabric, world_model.reward_model)
    world_model.rssm.recurrent_model = setup_module(fabric, world_model.rssm.recurrent_model)
    world_model.rssm.representation_model = setup_module(fabric, world_model.rssm.representation_model)
    world_model.rssm.transition_model = setup_module(fabric, world_model.rssm.transition_model)
    if world_model.continue_model:
        world_model.continue_model = setup_module(fabric, world_model.continue_model)
    actor = setup_module(fabric, actor)
    critic = setup_module(fabric, critic)

    # Setup target critic with a SingleDeviceStrategy
    target_critic = copy.deepcopy(critic.module)
    if target_critic_state:
        target_critic.load_state_dict(target_critic_state)
    target_critic = policy_fabric.setup_module(target_critic)

    # Setup the policy agent with a single-device Fabric
    policy.encoder = policy_fabric.setup_module(policy.encoder)
    policy.recurrent_model = policy_fabric.setup_module(policy.recurrent_model)
    policy.representation_model = policy_fabric.setup_module(policy.representation_model)
    policy.actor = policy_fabric.setup_module(policy.actor)

    # Tie weights between the agent and the policy
    for agent_p, p in zip(world_model.encoder.parameters(), policy.encoder.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(world_model.rssm.recurrent_model.parameters(), policy.recurrent_model.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(world_model.rssm.representation_model.parameters(), policy.representation_model.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(actor.parameters(), policy.actor.parameters()):
        p.data = agent_p.data
    # The step of the policy, compiled with `algo.compile` (`compiled_policy`): without CUDA graphs, since it keeps its
    # states in its attributes
    policy.get_actions = compiled_policy(policy.get_actions, fabric, cfg, cuda_graphs=False)
    return world_model, actor, critic, target_critic, policy
