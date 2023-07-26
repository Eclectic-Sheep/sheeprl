"""
Adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py
"""
import warnings
from math import prod
from typing import Any, Dict, Optional, Sequence, Tuple, Union, no_type_check

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sheeprl.utils.model import ArgsType, LayerNormChannelLast, ModuleType, cnn_forward, create_layers, miniblock


class MLP(nn.Module):
    """Simple MLP backbone.

    Args:
        input_dims (Union[int, Sequence[int]]): dimensions of the input vector.
        output_dim (int, optional): dimension of the output vector. If set to None, there
            is no final linear layer. Else, a final linear layer is added.
            Defaults to None.
        hidden_sizes (Sequence[int], optional): shape of MLP passed in as a list, not including
            input_dims and output_dim.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
        flatten_dim (int, optional): whether to flatten input data. The flatten dimension starts from 1.
            Defaults to True.
    """

    def __init__(
        self,
        input_dims: Union[int, Sequence[int]],
        output_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = (),
        layer_args: Optional[ArgsType] = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        flatten_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_sizes)
        if num_layers < 1 and output_dim is None:
            raise ValueError("The number of layers should be at least 1.")

        if isinstance(input_dims, Sequence) and flatten_dim is None:
            warnings.warn(
                "input_dims is a sequence, but flatten_dim is not specified. "
                "Be careful to flatten the input data correctly before the forward."
            )

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        if isinstance(input_dims, int):
            input_dims = [input_dims]
        hidden_sizes = [prod(input_dims)] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, nn.Linear, l_args, drop, drop_args, norm, norm_args, activ, act_args)
        if output_dim is not None:
            model += [nn.Linear(hidden_sizes[-1], output_dim)]

        self._output_dim = output_dim or hidden_sizes[-1]
        self._model = nn.Sequential(*model)
        self._flatten_dim = flatten_dim

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def flatten_dim(self) -> Optional[int]:
        return self._flatten_dim

    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        if self.flatten_dim is not None:
            obs = obs.flatten(self.flatten_dim)
        return self.model(obs)


class CNN(nn.Module):
    """Simple CNN backbone.

    Args:
        input_channels (int): dimensions of the input channels.
        hidden_channels (Sequence[int], optional): intermediate number of channels for the CNN, including the output channels.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        layer_args: ArgsType = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_channels)
        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, nn.Conv2d, l_args, drop, drop_args, norm, norm_args, activ, act_args)

        self._output_dim = hidden_sizes[-1]
        self._model = nn.Sequential(*model)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        return self.model(obs)


class DeCNN(nn.Module):
    """Simple DeCNN backbone.

    Args:
        input_channels (int): dimensions of the input channels.
        hidden_channels (Sequence[int], optional): intermediate number of channels for the CNN, including the output channels.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int] = (),
        layer_args: ArgsType = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_channels)
        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(
                in_dim, out_dim, nn.ConvTranspose2d, l_args, drop, drop_args, norm, norm_args, activ, act_args
            )

        self._output_dim = hidden_sizes[-1]
        self._model = nn.Sequential(*model)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        return self.model(obs)


class NatureCNN(CNN):
    """CNN from DQN Nature paper: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.

    Args:
        in_channels (int): the input channels to the first convolutional layer
        features_dim (int): the features dimension in output from the last convolutional layer
        screen_size (int, optional): the dimension of the input image as a single integer.
            Needed to extract the features and compute the output dimension after all the
            convolutional layers.
            Defaults to 64.
    """

    def __init__(self, in_channels: int, features_dim: int, screen_size: int = 64):
        super().__init__(
            in_channels,
            [32, 64, 64],
            layer_args=[
                {"kernel_size": 8, "stride": 4},
                {"kernel_size": 4, "stride": 2},
                {"kernel_size": 3, "stride": 1},
            ],
        )

        with torch.no_grad():
            x = self.model(torch.rand(1, in_channels, screen_size, screen_size, device=self.model[0].weight.device))
            out_dim = x.flatten(1).shape[1]
        self._output_dim = out_dim
        self.fc = None
        if features_dim is not None:
            self._output_dim = features_dim
            self.fc = nn.Linear(out_dim, features_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = F.relu(self.fc(x.flatten(1)))
        return x


class LayerNormGRUCell(nn.Module):
    """A GRU cell with a LayerNorm, taken
    from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py#L317.

    This particular GRU cell accepts 3-D inputs, with a sequence of length 1, and applies
    a LayerNorm after the projection of the inputs.

    Args:
        input_size (int): the input size.
        hidden_size (int): the hidden state size
        bias (bool, optional): whether to apply a bias to the input projection.
            Defaults to True.
        batch_first (bool, optional): whether the first dimension represent the batch dimension or not.
            Defaults to False.
        layer_norm (bool, optional): whether to apply a LayerNorm after the input projection.
            Defaults to False.
    """

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True, batch_first: bool = False, layer_norm: bool = False
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(3 * hidden_size)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (
            1,
            2,
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        input = torch.cat((hx, input), -1)
        x = self.linear(input)
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx

        if not is_batched:
            hx = hx.squeeze(0)
        elif is_3d:
            hx = hx.unsqueeze(0)

        return hx


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
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.cnn_keys = cnn_keys
        self.mlp_keys = mlp_keys
        if self.cnn_keys != []:
            cnn_input_channels = sum([np.prod(obs_space[k].shape[:-2]) for k in cnn_keys])
            self.cnn_input_dim = (cnn_input_channels, *obs_space[cnn_keys[0]].shape[-2:])
            self.cnn_encoder = nn.Sequential(
                CNN(
                    input_channels=cnn_input_channels,
                    hidden_channels=(torch.tensor([1, 2, 4, 8]) * cnn_channels_multiplier).tolist(),
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
            self.mlp_input_dim = sum([obs_space[k].shape[0] for k in mlp_keys])
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
        cnn_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ELU,
        mlp_act: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ELU,
        device: Union[str, torch.device] = "cpu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.mlp_splits = [obs_space[k].shape[0] for k in mlp_keys]
        self.cnn_splits = [np.prod(obs_space[k].shape[:-2]) for k in cnn_keys]
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
                    norm_layer=[LayerNormChannelLast for _ in range(3)] + [None] if layer_norm else None,
                    norm_args=[
                        {"normalized_shape": (2 ** (4 - i - 2)) * cnn_channels_multiplier}
                        for i in range(self.cnn_decoder_output_dim[0])
                    ]
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
