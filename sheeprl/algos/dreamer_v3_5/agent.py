"""The agent of DreamerV3 as published in Nature (Hafner et al., 2025, "Mastering diverse control tasks through world
models"), from https://github.com/danijar/dreamerv3 (`dreamerv3/rssm.py`, `dreamerv3/agent.py`, `embodied/jax/nets.py`
and `embodied/jax/heads.py`).

Compared with DreamerV3 of the 2023 paper (`sheeprl.algos.dreamer_v3`):
- RMSNorm instead of LayerNorm, and every linear layer has a bias;
- the recurrent model is a GRU with block-diagonal weights (`blocks` groups) over a much larger recurrent state, fed
  by separate projections of the recurrent state, of the stochastic state and of the actions;
- the initial state of every episode is zero (it is not learned);
- the encoder convolves without strides and halves the resolution with a max pooling; the decoder doubles it with a
  nearest-neighbour upsampling followed by a convolution, projects the recurrent state to the first feature maps with
  block-diagonal weights, and predicts the pixels in [0, 1] (sigmoid);
- the weights are initialized from a truncated normal scaled by the fan-in, the last layer of every head scaled by its
  `outscale`;
- the continuous actions come from a normal distribution with a bounded mean and standard deviation, and every
  action is learned with REINFORCE.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, Normal, OneHotCategorical

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.models.models import MultiDecoder, MultiEncoder
from sheeprl.utils.compile import compiled_player
from sheeprl.utils.fabric import setup_module
from sheeprl.utils.model import ModuleType, cnn_forward
from sheeprl.utils.utils import symlog

# The standard deviation of the standard normal truncated at -2 and 2
TRUNCATED_NORMAL_STD = 0.87962566103423978


def init_trunc_normal_(weight: Tensor, fan_in: int, scale: float = 1.0) -> Tensor:
    """The `trunc_normal_in` initialization of DreamerV3: a normal truncated at 2 standard deviations, with a variance
    of `scale**2 / fan_in` (zeros if `scale` is 0)."""
    with torch.no_grad():
        if scale == 0:
            return weight.zero_()
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
        return weight.mul_(scale / (TRUNCATED_NORMAL_STD * math.sqrt(fan_in)))


class Linear(nn.Linear):
    """A linear layer initialized as the ones of DreamerV3: the weights with `init_trunc_normal_` (scaled by
    `outscale`), the bias with zeros."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, outscale: float = 1.0) -> None:
        self.outscale = outscale
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self) -> None:
        init_trunc_normal_(self.weight, self.in_features, self.outscale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Conv2d(nn.Conv2d):
    """A convolution with stride 1 and "same" padding, initialized as the ones of DreamerV3 (`Linear`)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, outscale: float = 1.0) -> None:
        self.outscale = outscale
        super().__init__(in_channels, out_channels, kernel_size, padding="same")

    def reset_parameters(self) -> None:
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        init_trunc_normal_(self.weight, fan_in, self.outscale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class BlockLinear(nn.Module):
    """A linear layer with block-diagonal weights: the inputs and the outputs are split in `blocks` contiguous groups,
    and every group of outputs depends only on its group of inputs. The weights are initialized with the fan-in of the
    whole input, as DreamerV3 does.

    Args:
        in_features: the size of the input, a multiple of `blocks`.
        out_features: the size of the output, a multiple of `blocks`.
        blocks: the number of groups.
        bias: whether to add a bias. Default: True.
        outscale: the scale of the initialization of the weights. Default: 1.
    """

    def __init__(
        self, in_features: int, out_features: int, blocks: int, bias: bool = True, outscale: float = 1.0
    ) -> None:
        super().__init__()
        if in_features % blocks != 0 or out_features % blocks != 0:
            raise ValueError(
                f"The sizes of the input ({in_features}) and of the output ({out_features}) must be multiples of the "
                f"blocks ({blocks})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.weight = nn.Parameter(torch.empty(blocks, in_features // blocks, out_features // blocks))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        init_trunc_normal_(self.weight, in_features, outscale)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unflatten(-1, (self.blocks, -1))
        x = torch.einsum("...gi,gio->...go", x, self.weight).flatten(-2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, blocks={self.blocks}"


class RMSNorm(nn.Module):
    """RMSNorm (https://arxiv.org/abs/1910.07467) with a learnable scale, in the precision of the input (the fused
    kernel of PyTorch accumulates in float32), with the scale in that precision too.

    Args:
        dim: the size of the normalized dimension.
        eps: added to the mean square. Default: 1e-4.
        channels_first: whether to normalize the second dimension (the channels of `[N, C, H, W]` images) instead of
            the last one. Default: False.
    """

    def __init__(self, dim: int, eps: float = 1e-4, channels_first: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.channels_first = channels_first
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if self.channels_first:
            # The channels last: a view without copies of the images in the channels-last layout
            x = x.permute(0, 2, 3, 1)
        x = F.rms_norm(x, (self.dim,), self.weight.to(x.dtype), self.eps)
        return x.permute(0, 3, 1, 2) if self.channels_first else x

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, channels_first={self.channels_first}"


class MaxPool(nn.Module):
    """The maximum of every 2x2 window of `[N, C, H, W]` images. When compiled, as a reshape and a max (as DreamerV3
    does): `torch.compile` fuses its backward pass, a mask of the maxima, which is much faster than the one of the
    max pooling; otherwise, with the max pooling of cuDNN, faster than the reshape."""

    def forward(self, x: Tensor) -> Tensor:
        if torch.compiler.is_compiling():
            return x.unflatten(-1, (-1, 2)).unflatten(-3, (-1, 2)).amax((-1, -3))
        return F.max_pool2d(x, 2)


class Upsample(nn.Module):
    """The nearest-neighbour upsampling of `[N, C, H, W]` images by 2, as an expansion: its backward pass is a sum,
    much faster than the one of the interpolation."""

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        return x[:, :, :, None, :, None].expand(n, c, h, 2, w, 2).reshape(n, c, 2 * h, 2 * w)


def mlp(input_size: int, units: int, layers: int, activation: ModuleType = nn.SiLU, eps: float = 1e-4) -> nn.Sequential:
    """`layers` blocks of a linear layer of `units` units, an RMSNorm and the activation (`nets.MLP` of DreamerV3);
    with 0 layers, the identity."""
    modules = []
    for i in range(layers):
        modules += [Linear(input_size if i == 0 else units, units), RMSNorm(units, eps), activation()]
    return nn.Sequential(*modules)


class CNNEncoder(nn.Module):
    """The image encoder: the images, concatenated on the channels, go through `len(stages)` blocks of a convolution
    (stride 1), a max pooling that halves the resolution, an RMSNorm and the activation, and are flattened.

    Args:
        keys: the keys of the images.
        input_channels: the channels of every image.
        image_size: the size (height, width) of the images, divisible by `2 ** len(stages)`.
        channels_multiplier: the channels of the blocks are `channels_multiplier` times `stages`.
        stages: the multipliers of the channels of the blocks. Default: (2, 3, 4, 4).
        kernel_size: the size of the kernels. Default: 5.
        activation: the activation. Default: SiLU.
        eps: the epsilon of the RMSNorms. Default: 1e-4.
    """

    def __init__(
        self,
        keys: Sequence[str],
        input_channels: Sequence[int],
        image_size: Tuple[int, int],
        channels_multiplier: int,
        stages: Sequence[int] = (2, 3, 4, 4),
        kernel_size: int = 5,
        activation: ModuleType = nn.SiLU,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        factor = 2 ** len(stages)
        if image_size[0] % factor != 0 or image_size[1] % factor != 0:
            raise ValueError(f"The size of the images ({image_size}) must be divisible by 2^{len(stages)}")
        self.keys = keys
        self.input_dim = (sum(input_channels), *image_size)
        depths = [channels_multiplier * mult for mult in stages]
        layers = []
        in_channels = self.input_dim[0]
        for depth in depths:
            layers += [
                Conv2d(in_channels, depth, kernel_size),
                MaxPool(),
                RMSNorm(depth, eps, channels_first=True),
                activation(),
            ]
            in_channels = depth
        layers.append(nn.Flatten(-3, -1))
        self.model = nn.Sequential(*layers)
        self.output_dim = depths[-1] * (image_size[0] // factor) * (image_size[1] // factor)

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-3)
        return cnn_forward(self.model, x, x.shape[-3:], (-1,))


class MLPEncoder(nn.Module):
    """The vector encoder: the symlog of the vectors, concatenated, goes through `mlp_layers` blocks of `mlp`.

    Args:
        keys: the keys of the vectors.
        input_dims: the size of every vector.
        mlp_layers: the number of blocks. Default: 3.
        dense_units: the units of every block. Default: 1024.
        activation: the activation. Default: SiLU.
        eps: the epsilon of the RMSNorms. Default: 1e-4.
        symlog_inputs: whether to squash the vectors with the symlog. Default: True.
    """

    def __init__(
        self,
        keys: Sequence[str],
        input_dims: Sequence[int],
        mlp_layers: int = 3,
        dense_units: int = 1024,
        activation: ModuleType = nn.SiLU,
        eps: float = 1e-4,
        symlog_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = sum(input_dims)
        self.model = mlp(self.input_dim, dense_units, mlp_layers, activation, eps)
        self.output_dim = dense_units if mlp_layers > 0 else self.input_dim
        self.symlog_inputs = symlog_inputs

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([symlog(obs[k]) if self.symlog_inputs else obs[k] for k in self.keys], -1)
        return self.model(x)


class CNNDecoder(nn.Module):
    """The image decoder. The latent state is projected to feature maps of `2 ** len(stages)` times smaller than the
    images: with `block_space`, the recurrent state with block-diagonal weights (every block gives a group of channels)
    plus the stochastic state with a two-layer MLP; otherwise the whole latent state with a linear layer. They go
    through an RMSNorm and the activation, then `len(stages)` blocks of a nearest-neighbour upsampling that doubles the
    resolution and a convolution (followed by an RMSNorm and the activation but in the last block). The images are the
    sigmoid of the output, split on the channels.

    Args:
        keys: the keys of the images.
        output_channels: the channels of every image.
        channels_multiplier: the channels of the blocks (in reverse) are `channels_multiplier` times `stages`.
        stochastic_size: the size of the (flattened) stochastic state, the first part of the latent state.
        recurrent_state_size: the size of the recurrent state, the last part of the latent state.
        image_size: the size (height, width) of the images, divisible by `2 ** len(stages)`.
        stages: the multipliers of the channels. Default: (2, 3, 4, 4).
        kernel_size: the size of the kernels. Default: 5.
        block_space: the blocks of the projection of the recurrent state; 0 to project the whole latent state with
            a linear layer. Default: 8.
        dense_units: the units of the projection of the stochastic state are twice them. Default: 1024.
        activation: the activation. Default: SiLU.
        eps: the epsilon of the RMSNorms. Default: 1e-4.
        outscale: the scale of the initialization of the last convolution. Default: 1.
    """

    def __init__(
        self,
        keys: Sequence[str],
        output_channels: Sequence[int],
        channels_multiplier: int,
        stochastic_size: int,
        recurrent_state_size: int,
        image_size: Tuple[int, int],
        stages: Sequence[int] = (2, 3, 4, 4),
        kernel_size: int = 5,
        block_space: int = 8,
        dense_units: int = 1024,
        activation: ModuleType = nn.SiLU,
        eps: float = 1e-4,
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        factor = 2 ** len(stages)
        if image_size[0] % factor != 0 or image_size[1] % factor != 0:
            raise ValueError(f"The size of the images ({image_size}) must be divisible by 2^{len(stages)}")
        self.keys = keys
        self.output_channels = output_channels
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
        self.output_dim = (sum(output_channels), *image_size)
        depths = [channels_multiplier * mult for mult in stages]
        self.initial_shape = (depths[-1], image_size[0] // factor, image_size[1] // factor)
        units = math.prod(self.initial_shape)
        self.block_space = block_space
        if block_space:
            if depths[-1] % block_space != 0:
                raise ValueError(f"The channels of the projection ({depths[-1]}) must be a multiple of `block_space`")
            self.recurrent_projection = BlockLinear(recurrent_state_size, units, block_space)
            self.stochastic_projection = nn.Sequential(
                Linear(stochastic_size, 2 * dense_units),
                RMSNorm(2 * dense_units, eps),
                activation(),
                Linear(2 * dense_units, units),
            )
        else:
            self.latent_projection = Linear(stochastic_size + recurrent_state_size, units)
        self.projection_norm = nn.Sequential(RMSNorm(depths[-1], eps, channels_first=True), activation())
        layers = []
        for in_channels, out_channels in zip(reversed(depths[1:]), reversed(depths[:-1])):
            layers += [
                Upsample(),
                Conv2d(in_channels, out_channels, kernel_size),
                RMSNorm(out_channels, eps, channels_first=True),
                activation(),
            ]
        layers += [
            Upsample(),
            Conv2d(depths[0], self.output_dim[0], kernel_size, outscale=outscale),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, latent_states: Tensor) -> Dict[str, Tensor]:
        batch_shape = latent_states.shape[:-1]
        x = latent_states.reshape(-1, latent_states.shape[-1])
        if self.block_space:
            stochastic_state, recurrent_state = torch.split(x, [self.stochastic_size, self.recurrent_state_size], -1)
            # The block `g` of the projection of the recurrent state gives the `g`-th group of channels
            x = self.recurrent_projection(recurrent_state) + self.stochastic_projection(stochastic_state)
        else:
            x = self.latent_projection(x)
        x = self.projection_norm(x.view(-1, *self.initial_shape))
        x = torch.sigmoid(self.model(x))
        x = x.view(*batch_shape, *x.shape[1:])
        return {k: rec_obs for k, rec_obs in zip(self.keys, torch.split(x, self.output_channels, -3))}


class MLPDecoder(nn.Module):
    """The vector decoder: `mlp_layers` blocks of `mlp` and a linear head per vector, which predicts its symlog.

    Args:
        keys: the keys of the vectors.
        output_dims: the size of every vector.
        latent_state_size: the size of the latent state.
        mlp_layers: the number of blocks. Default: 3.
        dense_units: the units of every block. Default: 1024.
        activation: the activation. Default: SiLU.
        eps: the epsilon of the RMSNorms. Default: 1e-4.
        outscale: the scale of the initialization of the heads. Default: 1.
    """

    def __init__(
        self,
        keys: Sequence[str],
        output_dims: Sequence[int],
        latent_state_size: int,
        mlp_layers: int = 3,
        dense_units: int = 1024,
        activation: ModuleType = nn.SiLU,
        eps: float = 1e-4,
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.output_dims = output_dims
        self.model = mlp(latent_state_size, dense_units, mlp_layers, activation, eps)
        units = dense_units if mlp_layers > 0 else latent_state_size
        self.heads = nn.ModuleList([Linear(units, dim, outscale=outscale) for dim in output_dims])

    def forward(self, latent_states: Tensor) -> Dict[str, Tensor]:
        x = self.model(latent_states)
        return {k: head(x) for k, head in zip(self.keys, self.heads)}


class RecurrentModel(nn.Module):
    """The recurrent model: a GRU with block-diagonal weights (`RSSM._core` of DreamerV3).

    The recurrent state, the stochastic state and the actions are projected separately (a linear layer, an RMSNorm and
    the activation each) and concatenated; every one of the `blocks` groups of the recurrent state gets them all, goes
    through `dynamics_layers` block-diagonal layers and gives its reset, candidate and update gates. The actions are
    first divided by their magnitude where it exceeds one.

    Called with `actions` only, it returns the projection of the actions, which the step takes as
    `action_embedding`: the one of a whole sequence of known actions can be computed at once.

    Args:
        recurrent_state_size: the size of the recurrent state, a multiple of `blocks`.
        stochastic_size: the size of the (flattened) stochastic state.
        actions_size: the size of the actions.
        hidden_size: the size of each projection.
        blocks: the number of groups. Default: 8.
        dynamics_layers: the number of block-diagonal layers before the gates. Default: 1.
        activation: the activation. Default: SiLU.
        eps: the epsilon of the RMSNorms. Default: 1e-4.
    """

    def __init__(
        self,
        recurrent_state_size: int,
        stochastic_size: int,
        actions_size: int,
        hidden_size: int,
        blocks: int = 8,
        dynamics_layers: int = 1,
        activation: ModuleType = nn.SiLU,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if recurrent_state_size % blocks != 0:
            raise ValueError(
                f"The recurrent state size ({recurrent_state_size}) must be a multiple of the blocks ({blocks})"
            )
        self.recurrent_state_size = recurrent_state_size
        self.blocks = blocks
        self.recurrent_projection = mlp(recurrent_state_size, hidden_size, 1, activation, eps)
        self.stochastic_projection = mlp(stochastic_size, hidden_size, 1, activation, eps)
        self.action_projection = mlp(actions_size, hidden_size, 1, activation, eps)
        layers = []
        in_features = recurrent_state_size + blocks * 3 * hidden_size
        for _ in range(dynamics_layers):
            layers += [
                BlockLinear(in_features, recurrent_state_size, blocks),
                RMSNorm(recurrent_state_size, eps),
                activation(),
            ]
            in_features = recurrent_state_size
        self.hidden = nn.Sequential(*layers)
        self.gates = BlockLinear(in_features, 3 * recurrent_state_size, blocks)

    def forward(
        self,
        recurrent_state: Optional[Tensor] = None,
        stochastic_state: Optional[Tensor] = None,
        action_embedding: Optional[Tensor] = None,
        *,
        actions: Optional[Tensor] = None,
    ) -> Tensor:
        if actions is not None:
            return self.action_projection(actions / torch.clamp(actions.abs(), min=1).detach())
        x = torch.cat(
            (
                self.recurrent_projection(recurrent_state),
                self.stochastic_projection(stochastic_state),
                action_embedding,
            ),
            -1,
        )
        x = x.unsqueeze(-2).expand(*x.shape[:-1], self.blocks, x.shape[-1])
        x = torch.cat((recurrent_state.unflatten(-1, (self.blocks, -1)), x), -1).flatten(-2)
        # Every block gives its reset, candidate and update gates
        x = self.gates(self.hidden(x)).unflatten(-1, (self.blocks, 3, -1))
        reset, cand, update = (x[..., i, :].flatten(-2) for i in range(3))
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * recurrent_state


class RepresentationModel(nn.Module):
    """The representation model (posterior): `layers` blocks of `mlp` on the recurrent state and the embedded
    observations, concatenated, and a linear layer to the logits of the stochastic state.

    The part of its first layer that depends on the observations can be computed for a whole sequence at once
    (`forward(observations=...)`) and added to the part of the recurrent state at every step
    (`forward(recurrent_state=..., observation_projection=...)`).

    Args:
        recurrent_state_size: the size of the recurrent state, the first part of the input.
        embedding_size: the size of the embedded observations, the last part of the input.
        hidden_size: the units of the blocks.
        output_size: the size of the logits.
        layers: the number of blocks. Default: 1.
        activation: the activation. Default: SiLU.
        eps: the epsilon of the RMSNorms. Default: 1e-4.
    """

    def __init__(
        self,
        recurrent_state_size: int,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        layers: int = 1,
        activation: ModuleType = nn.SiLU,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.recurrent_state_size = recurrent_state_size
        self.input_layer = Linear(recurrent_state_size + embedding_size, hidden_size if layers > 0 else output_size)
        modules = []
        if layers > 0:
            modules += [RMSNorm(hidden_size, eps), activation()]
            modules += list(mlp(hidden_size, hidden_size, layers - 1, activation, eps))
            modules.append(Linear(hidden_size, output_size))
        self.model = nn.Sequential(*modules)

    def forward(
        self,
        recurrent_state: Optional[Tensor] = None,
        observation_projection: Optional[Tensor] = None,
        *,
        observations: Optional[Tensor] = None,
    ) -> Tensor:
        weight = self.input_layer.weight
        if observations is not None:
            return F.linear(observations, weight[:, self.recurrent_state_size :])
        x = F.linear(recurrent_state, weight[:, : self.recurrent_state_size], self.input_layer.bias)
        return self.model(x + observation_projection)


class RSSM(nn.Module):
    """The RSSM of DreamerV3 (Nature version). The states of the episodes start from zeros: at the first step of an
    episode (`is_first`) the previous recurrent state, stochastic state and actions are zeroed.

    Args:
        recurrent_model: the recurrent model.
        representation_model: the representation model (posterior).
        transition_model: the transition model (prior).
        discrete: the classes of every categorical variable of the stochastic state. Default: 64.
        unimix: the share of the uniform distribution mixed into the categoricals. Default: 0.01.
    """

    def __init__(
        self,
        recurrent_model: RecurrentModel | nn.Module,
        representation_model: RepresentationModel | nn.Module,
        transition_model: nn.Module,
        discrete: int = 64,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.discrete = discrete
        self.unimix = unimix

    def uniform_mix(self, logits: Tensor) -> Tensor:
        """The log-probabilities of the categoricals of `logits` (flattened), mixed with the uniform distribution,
        of shape `[..., stochastic_size, discrete]`."""
        logits = logits.unflatten(-1, (-1, self.discrete)).float()
        if self.unimix > 0:
            probs = logits.softmax(-1)
            probs = (1 - self.unimix) * probs + self.unimix / self.discrete
            return probs.log()
        return logits.log_softmax(-1)

    @staticmethod
    def sample(logits: Tensor) -> Tensor:
        """A one-hot sample of the categoricals of `logits` (`[..., stochastic_size, discrete]`), flattened, with the
        straight-through gradients of their probabilities."""
        probs = logits.softmax(-1)
        index = (probs / torch.empty_like(probs).exponential_()).argmax(-1)
        sample = F.one_hot(index, probs.shape[-1]).to(probs.dtype)
        return (sample + probs - probs.detach()).flatten(-2)

    def embed_actions(self, actions: Tensor) -> Tensor:
        """The projection of the actions taken by the recurrent model: the one of a whole sequence can be computed at
        once (the actions of the first steps of the episodes must be zeroed)."""
        return self.recurrent_model(actions=actions)

    def project_observations(self, embedded_obs: Tensor) -> Tensor:
        """The part of the first layer of the representation model that depends on the embedded observations: the one
        of a whole sequence can be computed at once."""
        return self.representation_model(observations=embedded_obs)

    def dynamic(
        self,
        posterior: Tensor,
        recurrent_state: Tensor,
        action_embedding: Tensor,
        observation_projection: Tensor,
        is_first: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """One step of the representation: the next recurrent state from the previous latent state and actions, and
        the posterior from it and the observations.

        Args:
            posterior: the previous stochastic state (flattened).
            recurrent_state: the previous recurrent state.
            action_embedding: the projection of the previous actions (`embed_actions`), zeroed where `is_first`.
            observation_projection: the projection of the embedded observations (`project_observations`).
            is_first: whether the step starts an episode, where the previous states are zeroed. Default: no reset.

        Returns:
            The recurrent state, the posterior (flattened) and its log-probabilities.
        """
        if is_first is not None:
            keep = 1 - is_first
            recurrent_state = recurrent_state * keep
            posterior = posterior * keep
        recurrent_state = self.recurrent_model(recurrent_state, posterior, action_embedding)
        posterior_logits = self.uniform_mix(self.representation_model(recurrent_state, observation_projection))
        return recurrent_state, self.sample(posterior_logits), posterior_logits

    def prior_logits(self, recurrent_states: Tensor) -> Tensor:
        """The log-probabilities of the priors of any number of recurrent states at once."""
        return self.uniform_mix(self.transition_model(recurrent_states))

    def imagination(self, prior: Tensor, recurrent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """One step of imagination: the next recurrent state and a sample of its prior (flattened)."""
        recurrent_state = self.recurrent_model(recurrent_state, prior, self.embed_actions(actions))
        return self.sample(self.prior_logits(recurrent_state)), recurrent_state


class Actor(nn.Module):
    """The actor: `mlp_layers` blocks of `mlp` and a linear head (initialized with `outscale`).

    The discrete actions come from categorical distributions (one per action, `unimix` of uniform mixed in). The
    continuous actions come from a normal distribution with mean `tanh(mean)` and standard deviation
    `(max_std - min_std) * sigmoid(std + 2) + min_std` (`bounded_normal` of DreamerV3); the samples are not clipped.

    Args:
        latent_state_size: the size of the latent state.
        actions_dim: the size of every discrete action, or of the continuous actions.
        is_continuous: whether the actions are continuous.
        dense_units: the units of the blocks. Default: 1024.
        mlp_layers: the number of blocks. Default: 3.
        min_std: the smallest standard deviation. Default: 0.1.
        max_std: the largest standard deviation. Default: 1.
        unimix: the share of the uniform distribution mixed into the categoricals. Default: 0.
        outscale: the scale of the initialization of the head. Default: 0.01.
        activation: the activation. Default: SiLU.
        eps: the epsilon of the RMSNorms. Default: 1e-4.
    """

    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        is_continuous: bool,
        dense_units: int = 1024,
        mlp_layers: int = 3,
        min_std: float = 0.1,
        max_std: float = 1.0,
        unimix: float = 0.0,
        outscale: float = 0.01,
        activation: ModuleType = nn.SiLU,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.model = mlp(latent_state_size, dense_units, mlp_layers, activation, eps)
        units = dense_units if mlp_layers > 0 else latent_state_size
        if is_continuous:
            self.mlp_heads = nn.ModuleList([Linear(units, 2 * int(np.sum(actions_dim)), outscale=outscale)])
        else:
            self.mlp_heads = nn.ModuleList([Linear(units, int(dim), outscale=outscale) for dim in actions_dim])
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.min_std = min_std
        self.max_std = max_std
        self.unimix = unimix

    def distributions(self, state: Tensor) -> Tuple[Distribution, ...]:
        """The distributions of the actions in `state`: one normal over all the continuous actions, or one categorical
        per discrete action. Called from `forward` (`with_actions=False`), they are computed in the precision of the
        run."""
        x = self.model(state)
        if self.is_continuous:
            mean, std = torch.chunk(self.mlp_heads[0](x).float(), 2, -1)
            std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std
            return (Independent(Normal(torch.tanh(mean), std), 1),)
        dists = []
        for head in self.mlp_heads:
            logits = head(x).float()
            if self.unimix > 0:
                probs = logits.softmax(-1)
                probs = (1 - self.unimix) * probs + self.unimix / probs.shape[-1]
                logits = probs.log()
            dists.append(OneHotCategorical(logits=logits))
        return tuple(dists)

    def forward(
        self,
        state: Tensor,
        greedy: bool = False,
        mask: Optional[Dict[str, Tensor]] = None,
        with_actions: bool = True,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Distribution, ...]]:
        """The actions in `state` (a sample, or the most likely ones if `greedy`; none without `with_actions`) and
        their distributions."""
        dists = self.distributions(state)
        if not with_actions:
            return (), dists
        if greedy:
            actions = tuple(d.mean if self.is_continuous else d.mode for d in dists)
        else:
            actions = tuple(d.sample() for d in dists)
        return actions, dists


class PlayerDV3_5(nn.Module):
    """The player of DreamerV3 (Nature version): it keeps the latent states of the environments and chooses their
    actions. It shares the modules of the agent.

    The continuous actions it returns (the ones played and stored) are clipped to [-1, 1]: the recurrent model divides
    the actions by their magnitude where it exceeds one, so it sees the same actions as from the samples.

    Args:
        encoder: the encoder.
        rssm: the RSSM.
        actor: the actor.
        actions_dim: the size of every discrete action, or of the continuous actions.
        num_envs: the number of environments.
        stochastic_size: the number of categorical variables of the stochastic state.
        recurrent_state_size: the size of the recurrent state.
        device: the device of the states.
        discrete_size: the classes of every categorical variable. Default: 64.
    """

    def __init__(
        self,
        encoder: nn.Module,
        rssm: RSSM,
        actor: nn.Module,
        actions_dim: Sequence[int],
        num_envs: int,
        stochastic_size: int,
        recurrent_state_size: int,
        device: str | torch.device,
        discrete_size: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.actor = actor
        self.actions_dim = actions_dim
        self.num_envs = num_envs
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
        self.device = device
        self.discrete_size = discrete_size

    @torch.no_grad()
    def init_states(self, reset_envs: Optional[Sequence[int]] = None) -> None:
        """Zero the states and the actions of the environments `reset_envs` (default: all of them)."""
        if reset_envs is None or len(reset_envs) == 0:
            self.actions = torch.zeros(1, self.num_envs, int(np.sum(self.actions_dim)), device=self.device)
            self.recurrent_state = torch.zeros(1, self.num_envs, self.recurrent_state_size, device=self.device)
            self.stochastic_state = torch.zeros(
                1, self.num_envs, self.stochastic_size * self.discrete_size, device=self.device
            )
        else:
            self.actions[:, reset_envs] = 0
            self.recurrent_state[:, reset_envs] = 0
            self.stochastic_state[:, reset_envs] = 0

    def get_actions(
        self, obs: Dict[str, Tensor], greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, ...]:
        """The actions of the environments given their observations, updating their latent states.

        Args:
            obs: the observations, of shape `[1, num_envs, ...]`.
            greedy: whether to take the most likely actions instead of sampling them. Default: False.
            mask: unused.

        Returns:
            The actions: one tensor with the continuous actions (clipped to [-1, 1]), or the one-hots of every
            discrete action.
        """
        embedded_obs = self.encoder(obs)
        self.recurrent_state, self.stochastic_state, _ = self.rssm.dynamic(
            self.stochastic_state,
            self.recurrent_state,
            self.rssm.embed_actions(self.actions),
            self.rssm.project_observations(embedded_obs),
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), greedy, mask)
        if self.actor.is_continuous:
            actions = tuple(torch.clamp(a, -1, 1) for a in actions)
        self.actions = torch.cat(actions, -1)
        return actions


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_state: Optional[Dict[str, Tensor]] = None,
    critic_state: Optional[Dict[str, Tensor]] = None,
    target_critic_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, nn.Module, nn.Module, nn.Module, PlayerDV3_5]:
    """Build the models and set them up with Fabric.

    Args:
        fabric: the fabric of the run.
        actions_dim: the size of every discrete action, or of the continuous actions.
        is_continuous: whether the actions are continuous.
        cfg: the configuration of the run.
        obs_space: the observation space.
        world_model_state: the weights of the world model. Default: initialized.
        actor_state: the weights of the actor. Default: initialized.
        critic_state: the weights of the critic. Default: initialized.
        target_critic_state: the weights of the slow critic. Default: the ones of the critic.

    Returns:
        The world model (encoder, RSSM, observation, reward and continue models), the actor, the critic, the slow critic
        (an exponential moving average of the critic) and the player, which shares the modules of the agent.
    """
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic
    activation = hydra.utils.get_class(cfg.algo.dense_act)
    eps = cfg.algo.norm_eps

    # Sizes
    recurrent_state_size = world_model_cfg.recurrent_model.recurrent_state_size
    stochastic_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    latent_state_size = stochastic_size + recurrent_state_size

    cnn_keys = cfg.algo.cnn_keys.encoder
    mlp_keys = cfg.algo.mlp_keys.encoder
    encoder_cfg = world_model_cfg.encoder
    cnn_encoder = (
        CNNEncoder(
            keys=cnn_keys,
            input_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cnn_keys],
            image_size=obs_space[cnn_keys[0]].shape[-2:],
            channels_multiplier=encoder_cfg.cnn_channels_multiplier,
            stages=encoder_cfg.cnn_stages,
            kernel_size=encoder_cfg.cnn_kernel_size,
            activation=activation,
            eps=eps,
        )
        if cnn_keys is not None and len(cnn_keys) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(
            keys=mlp_keys,
            input_dims=[obs_space[k].shape[0] for k in mlp_keys],
            mlp_layers=encoder_cfg.mlp_layers,
            dense_units=encoder_cfg.dense_units,
            activation=activation,
            eps=eps,
            symlog_inputs=encoder_cfg.symlog_inputs,
        )
        if mlp_keys is not None and len(mlp_keys) > 0
        else None
    )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder)

    recurrent_cfg = world_model_cfg.recurrent_model
    recurrent_model = RecurrentModel(
        recurrent_state_size=recurrent_state_size,
        stochastic_size=stochastic_size,
        actions_size=int(np.sum(actions_dim)),
        hidden_size=recurrent_cfg.hidden_size,
        blocks=recurrent_cfg.blocks,
        dynamics_layers=recurrent_cfg.dynamics_layers,
        activation=activation,
        eps=eps,
    )
    representation_cfg = world_model_cfg.representation_model
    representation_model = RepresentationModel(
        recurrent_state_size=recurrent_state_size,
        embedding_size=encoder.output_dim,
        hidden_size=representation_cfg.hidden_size,
        output_size=stochastic_size,
        layers=representation_cfg.layers,
        activation=activation,
        eps=eps,
    )
    transition_cfg = world_model_cfg.transition_model
    transition_model = nn.Sequential(
        *mlp(recurrent_state_size, transition_cfg.hidden_size, transition_cfg.layers, activation, eps),
        Linear(transition_cfg.hidden_size if transition_cfg.layers > 0 else recurrent_state_size, stochastic_size),
    )
    rssm = RSSM(
        recurrent_model=recurrent_model,
        representation_model=representation_model,
        transition_model=transition_model,
        discrete=world_model_cfg.discrete_size,
        unimix=world_model_cfg.unimix,
    )

    decoder_cfg = world_model_cfg.observation_model
    cnn_decoder_keys = cfg.algo.cnn_keys.decoder
    mlp_decoder_keys = cfg.algo.mlp_keys.decoder
    cnn_decoder = (
        CNNDecoder(
            keys=cnn_decoder_keys,
            output_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cnn_decoder_keys],
            channels_multiplier=decoder_cfg.cnn_channels_multiplier,
            stochastic_size=stochastic_size,
            recurrent_state_size=recurrent_state_size,
            image_size=obs_space[cnn_decoder_keys[0]].shape[-2:],
            stages=decoder_cfg.cnn_stages,
            kernel_size=decoder_cfg.cnn_kernel_size,
            block_space=decoder_cfg.block_space,
            dense_units=decoder_cfg.dense_units,
            activation=activation,
            eps=eps,
        )
        if cnn_decoder_keys is not None and len(cnn_decoder_keys) > 0
        else None
    )
    mlp_decoder = (
        MLPDecoder(
            keys=mlp_decoder_keys,
            output_dims=[obs_space[k].shape[0] for k in mlp_decoder_keys],
            latent_state_size=latent_state_size,
            mlp_layers=decoder_cfg.mlp_layers,
            dense_units=decoder_cfg.dense_units,
            activation=activation,
            eps=eps,
        )
        if mlp_decoder_keys is not None and len(mlp_decoder_keys) > 0
        else None
    )
    observation_model = MultiDecoder(cnn_decoder, mlp_decoder)

    def head(cfg_: Dict[str, Any], output_size: int, outscale: float) -> nn.Sequential:
        units = cfg_.dense_units if cfg_.mlp_layers > 0 else latent_state_size
        return nn.Sequential(
            *mlp(latent_state_size, cfg_.dense_units, cfg_.mlp_layers, activation, eps),
            Linear(units, output_size, outscale=outscale),
        )

    reward_model = head(world_model_cfg.reward_model, world_model_cfg.reward_model.bins, 0.0)
    continue_model = head(world_model_cfg.discount_model, 1, 1.0)
    world_model = WorldModel(encoder, rssm, observation_model, reward_model, continue_model)

    actor = Actor(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        dense_units=actor_cfg.dense_units,
        mlp_layers=actor_cfg.mlp_layers,
        min_std=actor_cfg.min_std,
        max_std=actor_cfg.max_std,
        unimix=actor_cfg.unimix,
        outscale=actor_cfg.outscale,
        activation=activation,
        eps=eps,
    )
    critic = head(critic_cfg, critic_cfg.bins, critic_cfg.outscale)

    # Load models from checkpoint
    if world_model_state:
        world_model.load_state_dict(world_model_state)
    if actor_state:
        actor.load_state_dict(actor_state)
    if critic_state:
        critic.load_state_dict(critic_state)

    if fabric.device.type == "cuda":
        # The convolutions of cuDNN run in the channels-last layout: weights in it spare the conversions of the
        # activations from and to it (the values of the weights don't change)
        world_model.encoder.to(memory_format=torch.channels_last)
        world_model.observation_model.to(memory_format=torch.channels_last)

    # Setup models with Fabric: every module whose methods are called (not only `forward`) is set up through its
    # submodules, which run in the precision of the run
    world_model.encoder = setup_module(fabric, world_model.encoder)
    world_model.observation_model = setup_module(fabric, world_model.observation_model)
    world_model.reward_model = setup_module(fabric, world_model.reward_model)
    world_model.continue_model = setup_module(fabric, world_model.continue_model)
    world_model.rssm.recurrent_model = setup_module(fabric, world_model.rssm.recurrent_model)
    world_model.rssm.representation_model = setup_module(fabric, world_model.rssm.representation_model)
    world_model.rssm.transition_model = setup_module(fabric, world_model.rssm.transition_model)
    actor = setup_module(fabric, actor)
    critic = setup_module(fabric, critic)

    # The slow critic: a copy of the critic, or the one of the checkpoint
    target_critic = copy.deepcopy(critic.module)
    if target_critic_state:
        target_critic.load_state_dict(target_critic_state)
    target_critic = setup_module(fabric, target_critic)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    # The player shares the modules of the agent, which are on the device of the process
    player = PlayerDV3_5(
        world_model.encoder,
        world_model.rssm,
        actor,
        actions_dim,
        cfg.env.num_envs,
        world_model_cfg.stochastic_size,
        recurrent_state_size,
        fabric.device,
        discrete_size=world_model_cfg.discrete_size,
    )
    # The step of the player, compiled with `algo.compile` (`compiled_player`): without CUDA graphs, since it keeps its
    # states in its attributes
    player.get_actions = compiled_player(player.get_actions, fabric, cfg, cuda_graphs=False)
    return world_model, actor, critic, target_critic, player
