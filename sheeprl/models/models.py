"""
Adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py
"""
import warnings
from math import prod
from typing import Dict, Optional, Sequence, Union, no_type_check

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sheeprl.utils.model import ArgsType, ModuleType, cnn_forward, create_layers, miniblock


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
        hidden_channels (Sequence[int], optional): intermediate number of channels for the CNN,
            including the output channels.
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
        cnn_layer: ModuleType = nn.Conv2d,
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
            model += miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

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
        hidden_channels (Sequence[int], optional): intermediate number of channels for the CNN,
            including the output channels.
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
        cnn_layer: ModuleType = nn.ConvTranspose2d,
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
            model += miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

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
        x = cnn_forward(self.model, x, input_dim=x.shape[-3:], output_dim=(-1,))
        x = F.relu(self.fc(x))
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
        cnn_encoder: ModuleType,
        mlp_encoder: ModuleType,
    ) -> None:
        super().__init__()
        if cnn_encoder is None and mlp_encoder is None:
            raise ValueError("There must be at least one encoder, both cnn and mlp encoders are None")
        if cnn_encoder is not None:
            if getattr(cnn_encoder, "input_dim", None) is None:
                raise AttributeError(
                    "`cnn_encoder` must contain the `input_dim` attribute representing "
                    "the dimension of the input tensor"
                )
            if getattr(cnn_encoder, "output_dim", None) is None:
                raise AttributeError(
                    "`cnn_encoder` must contain the `output_dim` attribute representing "
                    "the dimension of the output tensor"
                )
        if mlp_encoder is not None:
            if getattr(mlp_encoder, "input_dim", None) is None:
                raise AttributeError(
                    "`mlp_encoder` must contain the `input_dim` attribute representing "
                    "the dimension of the input tensor"
                )
            if getattr(mlp_encoder, "output_dim", None) is None:
                raise AttributeError(
                    "`mlp_encoder` must contain the `output_dim` attribute representing "
                    "the dimension of the output tensor"
                )
        self.cnn_encoder = cnn_encoder
        self.mlp_encoder = mlp_encoder
        self.cnn_input_dim = self.cnn_encoder.input_dim if self.cnn_encoder is not None else None
        self.mlp_input_dim = self.mlp_encoder.input_dim if self.mlp_encoder is not None else None
        self.cnn_output_dim = self.cnn_encoder.output_dim if self.cnn_encoder is not None else 0
        self.mlp_output_dim = self.mlp_encoder.output_dim if self.mlp_encoder is not None else 0
        self.output_dim = self.cnn_output_dim + self.mlp_output_dim

    @property
    def cnn_keys(self) -> Sequence[str]:
        return self.cnn_encoder.keys if self.cnn_encoder is not None else []

    @property
    def mlp_keys(self) -> Sequence[str]:
        return self.mlp_encoder.keys if self.mlp_encoder is not None else []

    def forward(self, obs: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        device = obs[list(obs.keys())[0]].device
        cnn_out = torch.tensor((), device=device)
        mlp_out = torch.tensor((), device=device)
        if self.cnn_encoder is not None:
            cnn_out = self.cnn_encoder(obs, *args, **kwargs)
        if self.mlp_encoder is not None:
            mlp_out = self.mlp_encoder(obs, *args, **kwargs)
        return torch.cat((cnn_out, mlp_out), -1)


class MultiDecoder(nn.Module):
    def __init__(
        self,
        cnn_decoder: ModuleType,
        mlp_decoder: ModuleType,
    ) -> None:
        super().__init__()
        if cnn_decoder is None and mlp_decoder is None:
            raise ValueError("There must be an decoder, both cnn and mlp decoders are None")
        self.cnn_decoder = cnn_decoder
        self.mlp_decoder = mlp_decoder

    @property
    def cnn_keys(self) -> Sequence[str]:
        return self.cnn_decoder.keys if self.cnn_decoder is not None else []

    @property
    def mlp_keys(self) -> Sequence[str]:
        return self.mlp_decoder.keys if self.mlp_decoder is not None else []

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        reconstructed_obs = {}
        if self.cnn_decoder is not None:
            reconstructed_obs.update(self.cnn_decoder(x))
        if self.mlp_decoder is not None:
            reconstructed_obs.update(self.mlp_decoder(x))
        return reconstructed_obs
