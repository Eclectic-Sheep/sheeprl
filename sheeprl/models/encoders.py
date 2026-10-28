"""Encoder networks with the contract of the inputs (`sheeprl.models.io.Input`): built with `input_shape`, the shape of
one sample of the concatenated keys after their `batch` transforms, they expose `output_dim` and accept any leading
dimensions."""

from __future__ import annotations

from math import prod
from typing import Optional, Sequence, Union

import hydra
import torch
from torch import Tensor, nn

from sheeprl.models.models import CNN, MLP
from sheeprl.utils.model import cnn_forward


def _activation(activation: Union[str, type]) -> type:
    return hydra.utils.get_class(activation) if isinstance(activation, str) else activation


class MLPEncoder(nn.Module):
    """An MLP on a vector of shape `input_shape`: `mlp_layers` hidden layers of `dense_units` units, then a linear layer
    to `features_dim` features (the last hidden layer without it); the vector itself without layers."""

    def __init__(
        self,
        input_shape: Sequence[int],
        features_dim: Optional[int] = None,
        dense_units: int = 64,
        mlp_layers: int = 2,
        dense_act: Union[str, type] = "torch.nn.ReLU",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        input_shape = tuple(int(d) for d in input_shape)
        if len(input_shape) != 1:
            raise ValueError(
                f"MLPEncoder takes vectors, got the shape {input_shape}: flatten the keys "
                "(`sheeprl.transforms.Flatten`)"
            )
        input_dim = input_shape[0]
        self.output_dim = features_dim if features_dim else dense_units
        if mlp_layers == 0:
            self.model = nn.Identity()
            self.output_dim = input_dim
        else:
            self.model = MLP(
                input_dim,
                features_dim,
                [dense_units] * mlp_layers,
                activation=_activation(dense_act),
                norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
                norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ConvEncoder(nn.Module):
    """Convolutions on grids or small images of shape `input_shape` (channels, height, width): one layer of
    `channels[i]` channels each, with `kernel_size`, `stride`, `padding` and `activation`; the feature maps flattened,
    then a linear layer to `features_dim` features with the activation (none without it)."""

    def __init__(
        self,
        input_shape: Sequence[int],
        channels: Sequence[int] = (32, 64),
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: Union[str, type] = "torch.nn.ReLU",
        features_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        input_shape = tuple(int(d) for d in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"ConvEncoder takes grids of shape (channels, height, width), got {input_shape}")
        self.input_shape = input_shape
        layer_args = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
        self.model = CNN(input_shape[0], list(channels), layer_args=layer_args, activation=_activation(activation))
        with torch.no_grad():
            out_dim = self.model(torch.zeros(1, *input_shape)).flatten(1).shape[1]
        self.fc = None
        self.output_dim = out_dim
        if features_dim is not None:
            self.fc = nn.Sequential(nn.Linear(out_dim, features_dim), _activation(activation)())
            self.output_dim = features_dim

    def forward(self, x: Tensor) -> Tensor:
        x = cnn_forward(self.model, x, input_dim=self.input_shape, output_dim=(-1,))
        return x if self.fc is None else self.fc(x)


class IdentityEncoder(nn.Module):
    """No network: the keys of the input flattened and concatenated, as the features (e.g. the observations SAC's
    actor and critics read)."""

    def __init__(self, input_shape: Sequence[int]) -> None:
        super().__init__()
        self.input_shape = tuple(int(d) for d in input_shape)
        self.output_dim = prod(self.input_shape)

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[: x.dim() - len(self.input_shape)], self.output_dim)
