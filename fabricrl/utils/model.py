from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from torch import nn

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]], Sequence[Dict[Any, Any]]]


def miniblock(
    input_size: int,
    output_size: int = 0,
    dropout_layer: Optional[ModuleType] = None,
    dropout_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    norm_layer: Optional[ModuleType] = None,
    norm_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    activation: Optional[ModuleType] = None,
    act_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation function."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if dropout_layer is not None:
        if isinstance(dropout_args, tuple):
            layers += [dropout_layer(*dropout_args)]
        elif isinstance(dropout_args, dict):
            layers += [dropout_layer(**dropout_args)]
        else:
            layers += [dropout_layer()]
    if norm_layer is not None:
        if isinstance(norm_args, tuple):
            layers += [norm_layer(output_size, *norm_args)]
        elif isinstance(norm_args, dict):
            layers += [norm_layer(output_size, **norm_args)]
        else:
            layers += [norm_layer(output_size)]
    if activation is not None:
        if isinstance(act_args, tuple):
            layers += [activation(*act_args)]
        elif isinstance(act_args, dict):
            layers += [activation(**act_args)]
        else:
            layers += [activation()]
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
