from typing import Sequence, Tuple, Union

import torch
from torch import nn


def create_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    activation_fn: Sequence[nn.Module],
) -> torch.nn.Module:
    """Create an MLP backbone consisting of MultiLayer-Perceptrons followed by an activation function

    Args:
        input_dim (int): the input dimension.
        hidden_sizes (Sequence[int]): a sequence of integers (possibly empty), which specifies the hidden dimensions
            of the MLP.
        activation_fn (Sequence[nn.Module]): the activation function between linear layers.

    Returns:
        torch.nn.Module: the MLP model as a `torch.nn.Sequential`
    """
    if len(hidden_sizes) > 0:
        layers = [nn.Linear(input_dim, hidden_sizes[0]), activation_fn[0] or nn.Identity()]
        for dim_i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[dim_i - 1], hidden_sizes[dim_i]))
            layers.append(activation_fn[dim_i] or nn.Identity())
        mlp = nn.Sequential(*layers)
    else:
        mlp = nn.Identity()
    return mlp


def per_layer_ortho_init_weights(module: torch.nn.Module, gain: float = 1.0, bias: float = 0.0):
    """Initialize the weights of a module with orthogonal weights.

    Args:
        module (torch.nn.Module): module to initialize
        gain (float, optional): gain of the orthogonal initialization. Defaults to 1.0.
        bias (float, optional): bias of the orthogonal initialization. Defaults to 0.0.
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(bias)
    elif isinstance(module, torch.nn.LSTM):
        for name, param in module.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, val=bias)
            elif "weight" in name:
                torch.nn.init.orthogonal_(param, gain=gain)
    elif isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
        for i in range(len(module)):
            per_layer_ortho_init_weights(module[i], gain=gain, bias=bias)


def repackage_hidden(h: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
    """Wraps hidden states in new Tensors, to detach them from their history.
    Args:
        h (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): the hidden state to repackage.
    """

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
