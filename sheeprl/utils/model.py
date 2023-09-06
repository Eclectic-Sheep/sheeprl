"""
Adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor, nn

ModuleType = Optional[Type[nn.Module]]
ArgType = Union[Tuple[Any, ...], Dict[Any, Any], None]
ArgsType = Union[ArgType, List[ArgType]]


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

    Based on Tianshou's miniblock function
    (https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py).

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
        (
            [nn.Linear, nn.Linear, nn.Linear],
            [{'arg1': 3, 'arg2': 'foo'}, {'arg1': 3, 'arg2': 'foo'}, {'arg1': 3, 'arg2': 'foo'}]
        )

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


def cnn_forward(
    model: nn.Module,
    input: Tensor,
    input_dim: Union[torch.Size, Tuple[int, ...]],
    output_dim: Union[torch.Size, Tuple[int, ...]],
) -> Tensor:
    """
    Compute the forward of a Convolutional neural network.
    It flattens all the dimensions before the model input_size, i.e.,
    the dimensions before the (C_in, H, W) dimensions for the encoder
    and the dimensions before the (feature_size,) dimension for the decoder.

    Args:
        model (nn.Module): the model.
        input (Tensor): the input tensor of dimension (*, C_in, H, W) or (*, feature_size),
            where * means any number of dimensions including None.
        input_dim (Union[torch.Size, Tuple[int, ...]]): the input dimensions,
            i.e., either (C_in, H, W) or (feature_size,).
        output_dim (Union[torch.Size, Tuple[int, ...]]): the desired dimensions in output.

    Returns:
        The output of dimensions (*, *output_dim).

    Examples:
        >>> encoder
        CNN(
            (network): Sequential(
                (0): Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (3): ReLU()
                (4): Flatten(start_dim=1, end_dim=-1)
                (5): Linear(in_features=128, out_features=25, bias=True)
            )
        )
        >>> input = torch.rand(10, 20, 3, 4, 4)
        >>> cnn_forward(encoder, input, (3, 4, 4), -1).shape
        torch.Size([10, 20, 25])

        >>> decoder
        Sequential(
            (0): Linear(in_features=230, out_features=1024, bias=True)
            (1): Unflatten(dim=-1, unflattened_size=(1024, 1, 1))
            (2): ConvTranspose2d(1024, 128, kernel_size=(5, 5), stride=(2, 2))
            (3): ReLU()
            (4): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2))
            (5): ReLU()
            (6): ConvTranspose2d(64, 32, kernel_size=(6, 6), stride=(2, 2))
            (7): ReLU()
            (8): ConvTranspose2d(32, 3, kernel_size=(6, 6), stride=(2, 2))
        )
        >>> input = torch.rand(10, 20, 230)
        >>> cnn_forward(decoder, input, (230,), (3, 64, 64)).shape
        torch.Size([10, 20, 3, 64, 64])
    """
    batch_shapes = input.shape[: -len(input_dim)]
    flatten_input = input.reshape(-1, *input_dim)
    model_out = model(flatten_input)
    return model_out.reshape(*batch_shapes, *output_dim)


class LayerNormChannelLast(nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (NCHW), received {len(x.shape)}D instead: {x.shape}")
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x
