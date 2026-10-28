from __future__ import annotations

import copy
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    OneHotCategoricalStraightThrough,
    TransformedDistribution,
)
from torch.distributions.utils import probs_to_logits

from sheeprl.algos.dreamer_policy import DreamerPolicy
from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
from sheeprl.core.actions import ActionCodec
from sheeprl.models.io import Encoder, build_decoder, build_encoder
from sheeprl.models.models import CNN, MLP, DeCNN, LayerNorm, LayerNormGRUCell
from sheeprl.utils.compile import compiled_policy
from sheeprl.utils.distribution import SafeTanhTransform
from sheeprl.utils.fabric import get_single_device_fabric, setup_module
from sheeprl.utils.model import ModuleType, cnn_forward
from sheeprl.utils.utils import dotdict


def _layer_norm(layer_norm: Dict[str, Any]) -> Tuple[Callable[..., nn.Module], Dict[str, Any]]:
    """The class and the arguments of a layer norm configured as `{cls: ..., kw: {...}}`."""
    cls = layer_norm["cls"]
    return (hydra.utils.get_class(cls) if isinstance(cls, str) else cls), dict(layer_norm.get("kw") or {})


def _activation(activation: Union[str, ModuleType]) -> ModuleType:
    return hydra.utils.get_class(activation) if isinstance(activation, str) else activation


class CNNEncoder(nn.Module):
    """The image encoder of DreamerV3, on images of shape `input_shape` (channels, height, width): `stages` convolutions
    with kernel 4, stride 2 and padding 1, of [1, 2, 4, ...] * `channels_multiplier` channels, each normalized
    (`layer_norm`, `{cls, kw}`; no bias when it is not `nn.Identity`) and activated; the feature maps flattened. The
    sides of the images must be divisible by 2 ** `stages` (64 x 64 images to 4 x 4 feature maps with 4 stages)."""

    def __init__(
        self,
        input_shape: Sequence[int],
        channels_multiplier: int,
        stages: int = 4,
        layer_norm: Dict[str, Any] = {"cls": "sheeprl.models.models.LayerNormChannelLast", "kw": {"eps": 1e-3}},
        activation: Union[str, ModuleType] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(d) for d in input_shape)
        if len(self.input_shape) != 3 or any(side % (2**stages) for side in self.input_shape[1:]):
            raise ValueError(
                f"The image encoder of DreamerV3 takes images (channels, height, width) whose sides are divisible by "
                f"2 ** stages = {2 ** stages}, got {self.input_shape}"
            )
        layer_norm_cls, layer_norm_kw = _layer_norm(layer_norm)
        self.model = nn.Sequential(
            CNN(
                input_channels=self.input_shape[0],
                hidden_channels=(torch.tensor([2**i for i in range(stages)]) * channels_multiplier).tolist(),
                cnn_layer=nn.Conv2d,
                layer_args={"kernel_size": 4, "stride": 2, "padding": 1, "bias": layer_norm_cls == nn.Identity},
                activation=_activation(activation),
                norm_layer=[layer_norm_cls] * stages,
                norm_args=[{**layer_norm_kw, "normalized_shape": (2**i) * channels_multiplier} for i in range(stages)],
            ),
            nn.Flatten(-3, -1),
        )
        with torch.no_grad():
            self.output_dim = self.model(torch.zeros(1, *self.input_shape)).shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        return cnn_forward(self.model, x, x.shape[-3:], (-1,))


class MLPEncoder(nn.Module):
    """The vector encoder of DreamerV3, on vectors of shape `input_shape` (the symlog of the observations is a transform
    of the input, `sheeprl.transforms.Symlog`): `mlp_layers` linear layers of `dense_units` units, each normalized
    (`layer_norm`, `{cls, kw}`; no bias when it is not `nn.Identity`) and activated."""

    def __init__(
        self,
        input_shape: Sequence[int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        layer_norm: Dict[str, Any] = {"cls": "sheeprl.models.models.LayerNorm", "kw": {"eps": 1e-3}},
        activation: Union[str, ModuleType] = nn.SiLU,
    ) -> None:
        super().__init__()
        input_shape = tuple(int(d) for d in input_shape)
        if len(input_shape) != 1:
            raise ValueError(f"The vector encoder of DreamerV3 takes vectors, got the shape {input_shape}")
        layer_norm_cls, layer_norm_kw = _layer_norm(layer_norm)
        self.model = MLP(
            input_shape[0],
            None,
            [dense_units] * mlp_layers,
            activation=_activation(activation),
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.output_dim = dense_units

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class CNNDecoder(nn.Module):
    """The inverse of `CNNEncoder`: a linear layer from the latent state to the feature maps of the last stage of the
    encoder, then `stages` transposed convolutions back to images of shape `output_shape`, the last one without
    normalization and activation."""

    def __init__(
        self,
        latent_size: int,
        output_shape: Sequence[int],
        channels_multiplier: int,
        stages: int = 4,
        layer_norm: Dict[str, Any] = {"cls": "sheeprl.models.models.LayerNormChannelLast", "kw": {"eps": 1e-3}},
        activation: Union[str, ModuleType] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.output_shape = tuple(int(d) for d in output_shape)
        channels, height, width = self.output_shape
        if height % (2**stages) or width % (2**stages):
            raise ValueError(
                f"The image decoder of DreamerV3 reconstructs images whose sides are divisible by 2 ** stages = "
                f"{2 ** stages}, got {self.output_shape}"
            )
        first_maps = (2 ** (stages - 1) * channels_multiplier, height // 2**stages, width // 2**stages)
        layer_norm_cls, layer_norm_kw = _layer_norm(layer_norm)
        activation = _activation(activation)
        self.model = nn.Sequential(
            nn.Linear(latent_size, int(np.prod(first_maps))),
            nn.Unflatten(1, first_maps),
            DeCNN(
                input_channels=first_maps[0],
                hidden_channels=(
                    torch.tensor([2**i for i in reversed(range(stages - 1))]) * channels_multiplier
                ).tolist()
                + [channels],
                cnn_layer=nn.ConvTranspose2d,
                layer_args=[
                    {"kernel_size": 4, "stride": 2, "padding": 1, "bias": layer_norm_cls == nn.Identity}
                    for _ in range(stages - 1)
                ]
                + [{"kernel_size": 4, "stride": 2, "padding": 1}],
                activation=[activation for _ in range(stages - 1)] + [None],
                norm_layer=[layer_norm_cls for _ in range(stages - 1)] + [None],
                norm_args=[
                    {**layer_norm_kw, "normalized_shape": (2 ** (stages - i - 2)) * channels_multiplier}
                    for i in range(stages - 1)
                ]
                + [None],
            ),
        )

    def output_layers(self) -> List[nn.Module]:
        """The layers the official implementation initializes uniformly (`hafner_initialization`): the last transposed
        convolution, then the projection of the latent state to the first feature maps."""
        return [self.model[-1].model[-1], self.model[0]]

    def forward(self, latent_states: Tensor) -> Tensor:
        return cnn_forward(self.model, latent_states, (latent_states.shape[-1],), self.output_shape)


class MLPDecoder(nn.Module):
    """The inverse of `MLPEncoder`: `mlp_layers` normalized linear layers of `dense_units` units from the latent state,
    then a linear head to `output_shape` (its values flat, then reshaped)."""

    def __init__(
        self,
        latent_size: int,
        output_shape: Sequence[int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        layer_norm: Dict[str, Any] = {"cls": "sheeprl.models.models.LayerNorm", "kw": {"eps": 1e-3}},
        activation: Union[str, ModuleType] = nn.SiLU,
    ) -> None:
        super().__init__()
        layer_norm_cls, layer_norm_kw = _layer_norm(layer_norm)
        self.model = MLP(
            latent_size,
            None,
            [dense_units] * mlp_layers,
            activation=_activation(activation),
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.output_shape = tuple(int(d) for d in output_shape)
        self.head = nn.Linear(dense_units, math.prod(self.output_shape))

    def output_layers(self) -> List[nn.Module]:
        """The layers the official implementation initializes uniformly (`hafner_initialization`): the head."""
        return [self.head]

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
        dense_units (int): the number of dense units.
        recurrent_state_size (int): the size of the recurrent state.
        activation_fn (nn.Module): the activation function.
            Default to SiLU.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
    """

    def __init__(
        self,
        input_size: int,
        recurrent_state_size: int,
        dense_units: int,
        activation_fn: nn.Module = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=None,
            hidden_sizes=[dense_units],
            activation=activation_fn,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=[layer_norm_cls],
            norm_args=[{**layer_norm_kw, "normalized_shape": dense_units}],
        )
        self.rnn = LayerNormGRUCell(
            dense_units,
            recurrent_state_size,
            bias=False,
            batch_first=False,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
        )
        self.recurrent_state_size = recurrent_state_size

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


class RepresentationModel(MLP):
    """The representation model of the RSSM: an MLP of the recurrent state and of the embedded observation,
    concatenated (in this order).

    The part of its first layer that depends on the observations can be computed for a whole sequence at once, with
    one matrix product (`forward(observations=...)`), and then added to the part of the recurrent state at every step
    of the unroll (`forward(recurrent_state=..., observation_projection=...)`), instead of one product of the whole
    input per step. The weights are the ones of the `MLP`.

    Args:
        recurrent_state_size (int): the size of the recurrent state, the first part of the input.
        **kwargs: the arguments of the `MLP`.
    """

    def __init__(self, recurrent_state_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recurrent_state_size = recurrent_state_size

    def forward(
        self,
        x: Optional[Tensor] = None,
        *,
        observations: Optional[Tensor] = None,
        recurrent_state: Optional[Tensor] = None,
        observation_projection: Optional[Tensor] = None,
    ) -> Tensor:
        """With `x` (the concatenated input), the output of the MLP. With `observations`, the part of the first layer
        that depends on them. With `recurrent_state` and `observation_projection` (the part of the observations), the
        output of the MLP."""
        if x is not None:
            return super().forward(x)
        first = self.model[0]
        if observations is not None:
            return F.linear(observations, first.weight[:, self.recurrent_state_size :])
        x = F.linear(recurrent_state, first.weight[:, : self.recurrent_state_size], first.bias) + observation_projection
        for layer in list(self.model)[1:]:
            x = layer(x)
        return x


class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in
            [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a
            multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multi-layer perceptron to predict the stochastic part of the latent state.
        distribution_cfg (dotdict): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over states, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`, where `unif = `1 / self.discrete`.
            Defaults to 0.01.
    """

    def __init__(
        self,
        recurrent_model: RecurrentModel | _FabricModule,
        representation_model: nn.Module | _FabricModule,
        transition_model: nn.Module | _FabricModule,
        distribution_cfg: dotdict,
        discrete: int = 32,
        unimix: float = 0.01,
        learnable_initial_recurrent_state: bool = True,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.distribution_cfg = distribution_cfg
        self.discrete = discrete
        self.unimix = unimix
        if learnable_initial_recurrent_state:
            self.initial_recurrent_state = nn.Parameter(
                torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_recurrent_state", torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )

    def get_initial_states(self, batch_shape: Sequence[int] | torch.Size) -> Tuple[Tensor, Tensor]:
        initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]
        return initial_recurrent_state, initial_posterior

    def dynamic(
        self,
        posterior: Tensor,
        recurrent_state: Tensor,
        action: Tensor,
        embedded_obs: Tensor,
        is_first: Tensor,
        initial_states: Optional[Tuple[Tensor, Tensor]] = None,
        projected: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
            initial_states (Tuple[Tensor, Tensor], optional): the initial recurrent state and posterior of the batch
                (`get_initial_states`), to compute them once for a whole sequence. Default: computed here.
            projected (bool): whether `embedded_obs` is the projection of the embedded observations
                (`project_observations`), computed once for a whole sequence.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model.
            The posterior stochastic state (Tensor): computed by the representation model
            The logits of the posterior state (Tensor): computed by the representation model from the recurrent state
                and the embedded observation.
            The logits of the prior are not computed: the recurrence doesn't use them, and `prior_logits` computes them
            for the recurrent states of a whole sequence at once.
        """
        action = (1 - is_first) * action

        if initial_states is None:
            initial_states = self.get_initial_states(recurrent_state.shape[:2])
        initial_recurrent_state, initial_posterior = initial_states
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        posterior_logits, posterior = self._representation(recurrent_state, embedded_obs, projected)
        return recurrent_state, posterior, posterior_logits

    def prior_logits(self, recurrent_states: Tensor) -> Tensor:
        """The logits of the priors of the transition model, for any number of recurrent states at once (e.g. the ones
        of a whole sequence, after the unroll of `dynamic`)."""
        return self._uniform_mix(self.transition_model(recurrent_states))

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def project_observations(self, embedded_obs: Tensor) -> Tensor:
        """The part of the first layer of the representation model that depends on the embedded observations, for a
        whole sequence at once (`RepresentationModel`): `dynamic(..., projected=True)` takes it in place of the
        embedded observations."""
        return self.representation_model(observations=embedded_obs)

    def _representation(
        self, recurrent_state: Tensor, embedded_obs: Tensor, projected: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_state (Tensor): the recurrent state of the recurrent model, i.e.,
                what is called h or deterministic state in
                [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            embedded_obs (Tensor): the embedded real observations provided by the environment, or their projection
                (`project_observations`) if `projected`.
            projected (bool): whether `embedded_obs` is the projection of the embedded observations.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        if projected:
            logits: Tensor = self.representation_model(
                recurrent_state=recurrent_state, observation_projection=embedded_obs
            )
        else:
            logits: Tensor = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

    def _transition(self, recurrent_out: Tensor, sample_state=True) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.
            sampler_state (bool): whether or not to sample the stochastic state.
                Default to True

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits: Tensor = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete, sample=sample_state)

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


class DecoupledRSSM(RSSM):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in
            [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a
            multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multi-layer perceptron to predict the stochastic part of the latent state.
        distribution_cfg (dotdict): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over states, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`, where `unif = `1 / self.discrete`.
            Defaults to 0.01.
    """

    def __init__(
        self,
        recurrent_model: nn.Module | _FabricModule,
        representation_model: nn.Module | _FabricModule,
        transition_model: nn.Module | _FabricModule,
        distribution_cfg: dotdict,
        discrete: int = 32,
        unimix: float = 0.01,
        learnable_initial_recurrent_state: bool = True,
    ) -> None:
        super().__init__(
            recurrent_model,
            representation_model,
            transition_model,
            distribution_cfg,
            discrete,
            unimix,
            learnable_initial_recurrent_state,
        )

    def dynamic(
        self,
        posterior: Tensor,
        recurrent_state: Tensor,
        action: Tensor,
        is_first: Tensor,
        initial_states: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
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
            is_first (Tensor): if this is the first step in the episode.
            initial_states (Tuple[Tensor, Tensor], optional): the initial recurrent state and posterior of the batch
                (`get_initial_states`), to compute them once for a whole sequence. Default: computed here.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model. The logits of the prior are not
            computed: `prior_logits` computes them for the recurrent states of a whole sequence at once.
        """
        action = (1 - is_first) * action

        if initial_states is None:
            initial_states = self.get_initial_states(recurrent_state.shape[:2])
        initial_recurrent_state, initial_posterior = initial_states
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        return self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)

    def _representation(self, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits: Tensor = self.representation_model(embedded_obs)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)


class DreamerV3Policy(nn.Module, DreamerPolicy):
    """
    The model of the Dreamer_v3 policy.

    Args:
        encoder (Encoder): the encoder.
        rssm (RSSM | DecoupledRSSM): the RSSM model.
        actor (_FabricModule): the actor.
        codec (ActionCodec): the actions of the agent and of the environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
            Defaults to 32.
        actor_type (str, optional): which actor the policy is using ('task' or 'exploration').
            Default to None.
        decoupled_rssm (bool, optional): whether to use the DecoupledRSSM model.
    """

    def __init__(
        self,
        encoder: Encoder | _FabricModule,
        rssm: RSSM | DecoupledRSSM,
        actor: Actor | MinedojoActor | _FabricModule,
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
        self.rssm = rssm
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
        self.decoupled_rssm = isinstance(rssm, DecoupledRSSM)

    @torch.no_grad()
    def init_states(self, num_envs: int) -> None:
        """The initial latent states (`RSSM.get_initial_states`) and zero actions of `num_envs` environments."""
        self.num_envs = num_envs
        self.actions = torch.zeros(1, num_envs, np.sum(self.actions_dim), device=self.device)
        recurrent_state, stochastic_state = self.rssm.get_initial_states((1, num_envs))
        # The initial recurrent state is one, expanded to the environments (their rows share the memory): a copy,
        # since the states of the environments are then reset one by one
        self.recurrent_state = recurrent_state.clone()
        self.stochastic_state = stochastic_state.reshape(1, num_envs, -1)

    @torch.no_grad()
    def reset_state(self, env_idxes: Sequence[int]) -> None:
        self.actions[:, env_idxes] = torch.zeros_like(self.actions[:, env_idxes])
        self.recurrent_state[:, env_idxes], stochastic_state = self.rssm.get_initial_states((1, len(env_idxes)))
        self.stochastic_state[:, env_idxes] = stochastic_state.reshape(1, len(env_idxes), -1)

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

        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.encoder(obs, phase="act")
        self.recurrent_state = self.rssm.recurrent_model(
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        if self.decoupled_rssm:
            _, self.stochastic_state = self.rssm._representation(embedded_obs)
        else:
            _, self.stochastic_state = self.rssm._representation(self.recurrent_state, embedded_obs)
        self.stochastic_state = self.stochastic_state.view(
            *self.stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), greedy, mask)
        self.actions = torch.cat(actions, -1)
        return actions


def clip_head_actions(actions: Sequence[Tensor], continuous: Sequence[bool], action_clip: float) -> Tensor:
    """The actions of the heads of the actor concatenated, the continuous ones (`continuous`) clipped to `action_clip`
    (`clip_actions`), the categorical ones as they are."""
    return torch.cat([clip_actions(a, action_clip) if c else a for a, c in zip(actions, continuous)], dim=-1)


def clip_actions(actions: Tensor, action_clip: float) -> Tensor:
    """The continuous `actions` divided by their magnitude where it exceeds `action_clip` (when greater than 0), with
    the gradient of the division stopped, as DreamerV3 clips the actions that its recurrent model takes."""
    if action_clip > 0.0:
        action_clip = torch.full_like(actions, action_clip)
        actions = actions * (action_clip / torch.maximum(action_clip, torch.abs(actions))).detach()
    return actions


class Actor(nn.Module):
    """
    The wrapper class of the Dreamer_v2 Actor model.

    Args:
        latent_state_size (int): the dimension of the latent state (stochastic size + recurrent_state_size).
        actions_dim (Sequence[int]): the size of every action (head): its dimensions if continuous, its number of
            choices if discrete.
        continuous (Sequence[bool]): whether every action is continuous: the actions may mix continuous and discrete
            ones, each with its own distribution.
        distribution_cfg (dotdict): The configs of the distributions (the one of the continuous actions).
        init_std (float): the amount to sum to the standard deviation.
            Default to 0.0.
        min_std (float): the minimum standard deviation for the actions.
            Default to 1.0.
        max_std (float): the maximum standard deviation for the actions.
            Default to 1.0.
        dense_units (int): the dimension of the hidden dense layers.
            Default to 1024.
        activation (int): the activation function to apply after the dense layers.
            Default to nn.SiLU.
        mlp_layers (int): the number of dense layers.
            Default to 5.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over actions, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`,
            where `unif = `1 / self.discrete`.
            Defaults to 0.01.
        action_clip (float): the action clip parameter.
            Default to 1.0.
    """

    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        continuous: Sequence[bool],
        distribution_cfg: dotdict,
        init_std: float = 0.0,
        min_std: float = 1.0,
        max_std: float = 1.0,
        dense_units: int = 1024,
        activation: nn.Module = nn.SiLU,
        mlp_layers: int = 5,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        unimix: float = 0.01,
        action_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.distribution_cfg = distribution_cfg
        self.distribution = distribution_cfg.get("type", "auto").lower()
        if self.distribution not in ("auto", "normal", "tanh_normal", "discrete", "scaled_normal"):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, `normal`, `tanh_normal` and `scaled_normal`. "
                f"Found: {self.distribution}"
            )
        self.continuous = tuple(bool(c) for c in continuous)
        if len(self.continuous) != len(actions_dim):
            raise ValueError(f"One kind per action: got {len(self.continuous)} for the actions {list(actions_dim)}")
        if self.distribution == "discrete" and any(self.continuous):
            raise ValueError("The `discrete` distribution plays categorical actions, but some actions are continuous")
        if self.distribution == "auto":
            self.distribution = "scaled_normal" if any(self.continuous) else "discrete"
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            flatten_dim=None,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        # One layer per action: the means and the stds of a continuous one, the logits of a categorical one
        self.mlp_heads = nn.ModuleList(
            [nn.Linear(dense_units, dim * 2 if c else dim) for dim, c in zip(actions_dim, self.continuous)]
        )
        self.actions_dim = actions_dim
        # Whether all the actions are continuous: the default of the objective (`reinforce_weight`)
        self.is_continuous = all(self.continuous)
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self._unimix = unimix
        self._action_clip = action_clip

    def forward(
        self, state: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None, clip: bool = True
    ) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Dict[str, Tensor], optional): the mask to use on the actions.
                Default to None.
            clip (bool): whether to clip the continuous actions to `action_clip` (`clip_actions`); without, the
                samples of the distribution, whose log-probabilities REINFORCE takes.
                Default to True.

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        pre_dist: List[Tensor] = [head(out) for head in self.mlp_heads]
        actions: List[Tensor] = []
        actions_dist: List[Distribution] = []
        for continuous, head_out in zip(self.continuous, pre_dist):
            if continuous:
                dist = self._continuous_distribution(head_out)
                if not greedy:
                    action = dist.rsample()
                else:
                    # The most likely of 100 samples, for every state
                    sample = dist.sample((100,))
                    best = dist.log_prob(sample).argmax(0, keepdim=True)
                    action = sample.gather(0, best.unsqueeze(-1).expand(1, *sample.shape[1:])).squeeze(0)
                if clip:
                    action = clip_actions(action, self._action_clip)
            else:
                dist = OneHotCategoricalStraightThrough(logits=self._uniform_mix(head_out))
                action = dist.rsample() if not greedy else dist.mode
            actions.append(action)
            actions_dist.append(dist)
        return tuple(actions), tuple(actions_dist)

    def _continuous_distribution(self, head_out: Tensor) -> Distribution:
        """The distribution of a continuous action (`distribution`), from the output of its head."""
        mean, std = torch.chunk(head_out, 2, -1)
        if self.distribution == "tanh_normal":
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + self.init_std) + self.min_std
            return Independent(TransformedDistribution(Normal(mean, std), SafeTanhTransform()), 1)
        if self.distribution == "normal":
            # The std is the output of the network: made positive as the one of `tanh_normal`
            std = F.softplus(std + self.init_std) + self.min_std
            return Independent(Normal(mean, std), 1)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + self.init_std) + self.min_std
        return Independent(Normal(torch.tanh(mean), std), 1)

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        if self._unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - self._unimix) * probs + self._unimix * uniform
            logits = probs_to_logits(probs)
        return logits


class MinedojoActor(Actor):
    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        continuous: Sequence[bool],
        distribution_cfg: dotdict,
        init_std: float = 0,
        min_std: float = 0.1,
        max_std: float = 1.0,
        dense_units: int = 1024,
        activation: nn.Module = nn.SiLU,
        mlp_layers: int = 5,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        unimix: float = 0.01,
        action_clip: float = 1.0,
    ) -> None:
        super().__init__(
            latent_state_size=latent_state_size,
            actions_dim=actions_dim,
            continuous=continuous,
            distribution_cfg=distribution_cfg,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            dense_units=dense_units,
            activation=activation,
            mlp_layers=mlp_layers,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
            unimix=unimix,
            action_clip=action_clip,
        )

    def forward(
        self, state: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None, clip: bool = True
    ) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Dict[str, Tensor], optional): the mask to apply to the actions.
                Default to None.
            clip (bool): unused, the actions are discrete.
                Default to True.

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        actions_logits: List[Tensor] = [self._uniform_mix(head(out)) for head in self.mlp_heads]
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


def build_agent(
    fabric: Fabric,
    codec: ActionCodec,
    cfg: dotdict,
    obs_space: gymnasium.spaces.Dict,
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_state: Optional[Dict[str, Tensor]] = None,
    critic_state: Optional[Dict[str, Tensor]] = None,
    target_critic_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, _FabricModule, DreamerV3Policy]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        codec (ActionCodec): the actions of the agent and of the environments.
        cfg (dotdict): the configs of DreamerV3: the encoder of `algo.inputs`, the decoder of `algo.outputs`.
        obs_space (Dict[str, Any]): the observation space of the store.
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
        The target critic (nn.Module).
    """
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic
    actions_dim = codec.dims

    # Sizes
    recurrent_state_size = world_model_cfg.recurrent_model.recurrent_state_size
    stochastic_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    latent_state_size = stochastic_size + recurrent_state_size

    # Define models
    encoder = build_encoder(cfg, obs_space)

    recurrent_model = RecurrentModel(
        input_size=int(sum(actions_dim) + stochastic_size),
        recurrent_state_size=world_model_cfg.recurrent_model.recurrent_state_size,
        dense_units=world_model_cfg.recurrent_model.dense_units,
        layer_norm_cls=hydra.utils.get_class(world_model_cfg.recurrent_model.layer_norm.cls),
        layer_norm_kw=world_model_cfg.recurrent_model.layer_norm.kw,
    )
    represention_model_input_size = encoder.output_dim
    if not cfg.algo.world_model.decoupled_rssm:
        represention_model_input_size += recurrent_state_size
    representation_ln_cls = hydra.utils.get_class(world_model_cfg.representation_model.layer_norm.cls)
    representation_cls = (
        MLP if cfg.algo.world_model.decoupled_rssm else partial(RepresentationModel, recurrent_state_size)
    )
    representation_model = representation_cls(
        input_dims=represention_model_input_size,
        output_dim=stochastic_size,
        hidden_sizes=[world_model_cfg.representation_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.representation_model.dense_act),
        layer_args={"bias": representation_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=[representation_ln_cls],
        norm_args=[
            {
                **world_model_cfg.representation_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.representation_model.hidden_size,
            }
        ],
    )
    transition_ln_cls = hydra.utils.get_class(world_model_cfg.transition_model.layer_norm.cls)
    transition_model = MLP(
        input_dims=recurrent_state_size,
        output_dim=stochastic_size,
        hidden_sizes=[world_model_cfg.transition_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.transition_model.dense_act),
        layer_args={"bias": transition_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=[transition_ln_cls],
        norm_args=[
            {
                **world_model_cfg.transition_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.transition_model.hidden_size,
            }
        ],
    )

    if cfg.algo.world_model.decoupled_rssm:
        rssm_cls = DecoupledRSSM
    else:
        rssm_cls = RSSM
    rssm = rssm_cls(
        recurrent_model=recurrent_model.apply(init_weights),
        representation_model=representation_model.apply(init_weights),
        transition_model=transition_model.apply(init_weights),
        distribution_cfg=cfg.distribution,
        discrete=world_model_cfg.discrete_size,
        unimix=cfg.algo.unimix,
        learnable_initial_recurrent_state=cfg.algo.world_model.learnable_initial_recurrent_state,
    ).to(fabric.device)

    observation_model = build_decoder(cfg, encoder, latent_size=latent_state_size)

    reward_ln_cls = hydra.utils.get_class(world_model_cfg.reward_model.layer_norm.cls)
    reward_model = MLP(
        input_dims=latent_state_size,
        output_dim=world_model_cfg.reward_model.bins,
        hidden_sizes=[world_model_cfg.reward_model.dense_units] * world_model_cfg.reward_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.reward_model.dense_act),
        layer_args={"bias": reward_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=reward_ln_cls,
        norm_args={
            **world_model_cfg.reward_model.layer_norm.kw,
            "normalized_shape": world_model_cfg.reward_model.dense_units,
        },
    )

    discount_ln_cls = hydra.utils.get_class(world_model_cfg.discount_model.layer_norm.cls)
    continue_model = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[world_model_cfg.discount_model.dense_units] * world_model_cfg.discount_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.discount_model.dense_act),
        layer_args={"bias": discount_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=discount_ln_cls,
        norm_args={
            **world_model_cfg.discount_model.layer_norm.kw,
            "normalized_shape": world_model_cfg.discount_model.dense_units,
        },
    )
    world_model = WorldModel(
        encoder.apply(init_weights),
        rssm,
        observation_model.apply(init_weights),
        reward_model.apply(init_weights),
        continue_model.apply(init_weights),
    )

    actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)
    actor: Actor | MinedojoActor = actor_cls(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        continuous=codec.continuous_heads,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        max_std=actor_cfg.max_std,
        dense_units=actor_cfg.dense_units,
        activation=hydra.utils.get_class(actor_cfg.dense_act),
        mlp_layers=actor_cfg.mlp_layers,
        distribution_cfg=cfg.distribution,
        layer_norm_cls=hydra.utils.get_class(actor_cfg.layer_norm.cls),
        layer_norm_kw=actor_cfg.layer_norm.kw,
        unimix=actor_cfg.unimix,
        action_clip=actor_cfg.action_clip,
    )

    critic_ln_cls = hydra.utils.get_class(critic_cfg.layer_norm.cls)
    critic = MLP(
        input_dims=latent_state_size,
        output_dim=critic_cfg.bins,
        hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
        activation=hydra.utils.get_class(critic_cfg.dense_act),
        layer_args={"bias": critic_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=critic_ln_cls,
        norm_args={
            **critic_cfg.layer_norm.kw,
            "normalized_shape": critic_cfg.dense_units,
        },
    )
    actor.apply(init_weights)
    critic.apply(init_weights)

    if cfg.algo.hafner_initialization:
        actor.mlp_heads.apply(uniform_init_weights(1.0))
        critic.model[-1].apply(uniform_init_weights(0.0))
        rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
        rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
        world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
        world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
        # The output layers of the decoders: the ones of the vectors, then the ones of the images (whose projection of
        # the latent state to the first feature maps gets the default initializer of the linear layers of the official
        # implementation, which its image decoder doesn't override)
        decoder_nets = list(observation_model.named_nets().values())
        for net in sorted(decoder_nets, key=lambda net: isinstance(net, CNNDecoder)):
            for layer in net.output_layers() if hasattr(net, "output_layers") else ():
                layer.apply(uniform_init_weights(1.0))

    # Load models from checkpoint
    if world_model_state:
        world_model.load_state_dict(world_model_state)
    if actor_state:
        actor.load_state_dict(actor_state)
    if critic_state:
        critic.load_state_dict(critic_state)

    if fabric.device.type == "cuda":
        # The convolutions of cuDNN run in the channels-last layout: weights in it spare the conversions of the
        # activations from and to it (the values of the weights don't change). Before the copy of the policy and the
        # setup, which keep the layout: changing it later would replace the weights the policy shares
        world_model.encoder.to(memory_format=torch.channels_last)
        world_model.observation_model.to(memory_format=torch.channels_last)

    # Create the policy agent
    policy_fabric = get_single_device_fabric(fabric)
    policy = DreamerV3Policy(
        copy.deepcopy(world_model.encoder),
        copy.deepcopy(world_model.rssm),
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

    # The target critic: a copy of the critic, or the one of the checkpoint
    target_critic = copy.deepcopy(critic.module)
    if target_critic_state:
        target_critic.load_state_dict(target_critic_state)
    target_critic = setup_module(fabric, target_critic)

    # Setup the policy agent with a single-device Fabric
    policy.encoder = policy_fabric.setup_module(policy.encoder)
    policy.rssm.recurrent_model = policy_fabric.setup_module(policy.rssm.recurrent_model)
    policy.rssm.transition_model = policy_fabric.setup_module(policy.rssm.transition_model)
    policy.rssm.representation_model = policy_fabric.setup_module(policy.rssm.representation_model)
    policy.actor = policy_fabric.setup_module(policy.actor)

    # Tie weights between the agent and the policy
    for agent_p, p in zip(world_model.encoder.parameters(), policy.encoder.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(world_model.rssm.parameters(), policy.rssm.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(actor.parameters(), policy.actor.parameters()):
        p.data = agent_p.data
    # The step of the policy, compiled with `algo.compile` (`compiled_policy`): without CUDA graphs, since it keeps its
    # states in its attributes
    policy.get_actions = compiled_policy(policy.get_actions, fabric, cfg, cuda_graphs=False)
    return world_model, actor, critic, target_critic, policy
