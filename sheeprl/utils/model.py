"""
Adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py
"""
import collections
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ModuleType = Optional[Type[nn.Module]]
ArgType = Union[Tuple[Any, ...], Dict[Any, Any], None]
ArgsType = Union[ArgType, List[ArgType]]


def _ntuple(n):
    """Copied from PyTorch since it's not importable as an internal function

    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/utils.py#L6
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


def create_layer_with_args(layer_type: ModuleType, layer_args: Optional[ArgType]) -> nn.Module:
    """Create a single layer with given layer type and arguments.

    Args:
        layer_type (ModuleType): the type of the layer to be created.
        layer_args (ArgType, optional): the arguments to be passed to the layer.
    """
    if layer_type is None:
        raise ValueError("`layer_type` must be not None")
    if isinstance(layer_args, tuple):
        return layer_type(*layer_args)
    elif isinstance(layer_args, dict):
        return layer_type(**layer_args)
    elif layer_args is None:
        return layer_type()
    else:
        raise ValueError(f"`layer_args` must be None, tuple or dict, got {type(layer_args)}")


def miniblock(
    input_size: int,
    output_size: int,
    layer_type: Type[nn.Module] = nn.Linear,
    layer_args: ArgType = None,
    dropout_layer: ModuleType = None,
    dropout_args: ArgType = None,
    norm_layer: ModuleType = None,
    norm_args: ArgType = None,
    activation: ModuleType = None,
    act_args: ArgType = None,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, dropout layer, norm layer and activation function.

    Based on Tianshou's miniblock function (https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py).

    Args:
        input_size (int): the input size of the miniblock (in_features for Linear and in_channels for Conv2d).
        output_size (int): the output size of the miniblock.
        layer_type (Type[nn.Linear], optional): the type of the layer to be created. Defaults to nn.Linear.
        layer_args (ArgType, optional): the arguments to be passed to the layer.
            Defaults to None.
        dropout_layer (ModuleType, optional): the type of the dropout layer to be created. Defaults to None.
        dropout_args (ArgType, optional): the arguments to be passed to the dropout
            layer. Defaults to None.
        norm_layer (ModuleType, optional): the type of the norm layer to be created. Defaults to None.
        norm_args (ArgType, optional): the arguments to be passed to the norm layer.
            Defaults to None.
        activation (ModuleType, optional): the type of the activation function to be created.
            Defaults to None.
        act_args (Tuple[Any, ...] | Dict[Any, Any] | None, optional): the arguments to be passed to the activation
            function. Defaults to None.

    Returns:
        List[nn.Module]: the miniblock as a list of layers.
    """
    if layer_args is None:
        layers: List[nn.Module] = [layer_type(input_size, output_size)]
    elif isinstance(layer_args, tuple):
        layers = [layer_type(input_size, output_size, *layer_args)]
    elif isinstance(layer_args, dict):
        layers = [layer_type(input_size, output_size, **layer_args)]
    else:
        raise ValueError(f"layer_args must be None, tuple or dict, got {type(layer_args)}")

    if dropout_layer is not None:
        layers += [create_layer_with_args(dropout_layer, dropout_args)]

    if norm_layer is not None:
        layers += [create_layer_with_args(norm_layer, norm_args)]

    if activation is not None:
        layers += [create_layer_with_args(activation, act_args)]
    return layers


def create_layers(
    layer_type: Union[ModuleType, List[ModuleType]], layer_args: Optional[ArgsType], num_layers: int
) -> Tuple[List[ModuleType], ArgsType]:
    """Create a list of layers with given layer type and arguments.

    If a layer_type is not specified, then the lists will be filled with None. If the layer type or the layer arguments
    are specified only once, they will be cast to a sequence of length num_layers.

    Args:
        layer_type (Union[ModuleType, Sequence[ModuleType]]): the type of the layer to be created.
        layer_args (ArgsType, optional): the arguments to be passed to the layer.
        num_layers (int): the number of layers to be created.

    Returns:
        Tuple[Sequence[ModuleType], ArgsType]: a list of layers and a list of args.

    Examples:
        >>> create_layers(nn.Linear, None, 3)
        ([nn.Linear, nn.Linear, nn.Linear], [None, None, None])

        >>> create_layers(nn.Linear, {"arg1":3, "arg2": "foo"}, 3)
        ([nn.Linear, nn.Linear, nn.Linear], [{'arg1': 3, 'arg2': 'foo'}, {'arg1': 3, 'arg2': 'foo'}, {'arg1': 3, 'arg2': 'foo'}])

        >>> create_layers([nn.Linear, nn.Conv2d], [{"bias":False}, {"kernel_size": 5, "bias": True}], 2)
        ([nn.Linear, nn.Conv2d], [{'bias': False}, {'kernel_size':5, 'bias': True}])

        >>> create_layers([nn.Linear, nn.Linear], (64, 10), 2)
        ([nn.Linear, nn.Linear], [(64, 10), (64, 10)])
    """
    if layer_type is None:
        layers_list = [None] * num_layers
        args_list = [None] * num_layers
        return layers_list, args_list

    if isinstance(layer_type, list):
        assert len(layer_type) == num_layers
        layers_list = layer_type
        if isinstance(layer_args, list):
            assert len(layer_args) == num_layers
            args_list = layer_args
        else:
            args_list = [layer_args for _ in range(num_layers)]
    else:
        layers_list = [layer_type for _ in range(num_layers)]
        args_list = [layer_args for _ in range(num_layers)]
    return layers_list, args_list


def per_layer_ortho_init_weights(module: nn.Module, gain: float = 1.0, bias: float = 0.0):
    """Initialize the weights of a module with orthogonal weights.

    Args:
        module (nn.Module): module to initialize
        gain (float, optional): gain of the orthogonal initialization. Defaults to 1.0.
        bias (float, optional): bias of the orthogonal initialization. Defaults to 0.0.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, val=bias)
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=gain)
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):
        for i in range(len(module)):
            per_layer_ortho_init_weights(module[i], gain=gain, bias=bias)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argiument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class ConvTranspose2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argiument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        c = self.conv(padded)
        return F.pad(c, self._reversed_padding_repeated_twice)


class LayerNormChannelLast(nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class LayerNormGRUCell(nn.Module):
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
            self.layer_norm = torch.nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        is_3d = False
        if input.dim() == 3:
            is_3d = True
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (1, 2), (
            f"LayerNormGRUCell: Expected input to be 1-D or 2-D " "but received {input.dim()}-D tensor"
        )

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
        if is_3d:
            hx = hx.unsqueeze(int(self.batch_first))

        return hx, hx
