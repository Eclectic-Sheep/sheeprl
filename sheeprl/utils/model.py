"""
Adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from torch import nn

ModuleType = Type[nn.Module]
ArgType = Union[Tuple[Any, ...], Dict[Any, Any]]
ArgsType = Union[ArgType, Sequence[ArgType]]


def create_layer_with_args(layer_type: ModuleType, layer_args: Optional[ArgType]) -> nn.Module:
    if isinstance(layer_args, tuple):
        return layer_type(*layer_args)
    elif isinstance(layer_args, dict):
        return layer_type(**layer_args)
    elif layer_args is None:
        return layer_type()
    else:
        raise ValueError(f"layer_args must be None, tuple or dict, got {type(layer_args)}")


def miniblock(
    input_size: int,
    output_size: int = 0,
    layer_type: Type[nn.Linear] = nn.Linear,
    layer_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    dropout_layer: Optional[ModuleType] = None,
    dropout_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    norm_layer: Optional[ModuleType] = None,
    norm_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    activation: Optional[ModuleType] = None,
    act_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation function."""
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
