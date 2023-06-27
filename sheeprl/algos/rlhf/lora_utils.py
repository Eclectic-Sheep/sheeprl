"""
References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
Taken and adapted from https://github.com/cccntu/minlora
License: MIT
"""
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, no_type_check

import torch
import torch.nn.utils.parametrize as parametrize
import transformers
from torch import nn

from sheeprl.algos.rlhf.args import ModelArgs


class LoRAParametrization(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        fan_in_fan_out: bool = False,
        rank: int = 4,
        lora_dropout_p: float = 0.0,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
        self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype))
        self.forward_fn = self.lora_forward
        self.lora_As: List[torch.nn.Parameter] = []
        self.lora_Bs: List[torch.nn.Parameter] = []

    def _dropout(self, A: torch.Tensor):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)  # type: ignore[operator]

    def lora_forward(self, X: torch.Tensor):
        return X + torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))) * self.scaling

    def forward(self, X: torch.Tensor):
        return self.forward_fn(X)

    def disable_lora(self):
        self.forward_fn = lambda x: x

    def enable_lora(self):
        self.forward_fn = self.lora_forward

    @classmethod
    def from_linear(cls, layer: torch.nn.Linear, rank: int = 4, lora_dropout_p: float = 0.0, lora_alpha: float = 1.0):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_conv2d(cls, layer: torch.nn.Conv2d, rank: int = 4, lora_dropout_p: float = 0.0, lora_alpha: float = 1.0):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_embedding(
        cls, layer: torch.nn.Embedding, rank: int = 4, lora_dropout_p: float = 0.0, lora_alpha: float = 1.0
    ):
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_conv1d(cls, layer: torch.nn.Conv1d, rank: int = 4, lora_dropout_p: float = 0.0, lora_alpha: float = 1.0):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )


def apply_lora(layer, register: bool = True, merge: bool = False, lora_config: Optional[Dict[Any, Any]] = None):
    """add lora parametrization to a layer, designed to be used with model.apply"""
    if register:
        if lora_config is None:
            raise ValueError("lora_config must be specified when register=True")
        if type(layer) in lora_config:
            for attr_name, parametrization in lora_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


def merge_lora(model: torch.nn.Module):
    """merge lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=True))


def remove_lora(model: torch.nn.Module):
    """remove lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=False))


def apply_to_lora(fn: Callable[[LoRAParametrization], None]):
    """apply a function to LoRAParametrization layers, designed to be used with model.apply"""

    def apply_fn(layer):
        if isinstance(layer, LoRAParametrization):
            fn(layer)

    return apply_fn


enable_lora = lambda model: model.apply(apply_to_lora(lambda x: x.enable_lora()))
disable_lora = lambda model: model.apply(apply_to_lora(lambda x: x.disable_lora()))


# ------------------- helper function for collecting parameters for training/saving -------------------


def name_is_lora(name: str):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["lora_A", "lora_B"]
    )


def name_is_bias(name: str):
    return name.split(".")[-1] == "bias"


def get_params_by_name(
    model: torch.nn.Module, print_shapes: bool = False, name_filter: Optional[Callable[[str], bool]] = None
):
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            if print_shapes:
                print(n, p.shape)
            yield p


def get_lora_params(model: torch.nn.Module, print_shapes: bool = False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_lora)


def get_bias_params(model: torch.nn.Module, print_shapes: bool = False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_bias)


def get_lora_state_dict(model: torch.nn.Module):
    return {k: v for k, v in model.state_dict().items() if name_is_lora(k)}


# ------------------- helper function for inferencing with multiple lora -------------------


def _prepare_for_multiple_lora(lora_layer: LoRAParametrization):
    lora_layer.lora_As = []
    lora_layer.lora_Bs = []


def _append_lora(lora_layer: LoRAParametrization):
    lora_layer.lora_As.append(nn.Parameter(lora_layer.lora_A.clone()))
    lora_layer.lora_Bs.append(nn.Parameter(lora_layer.lora_B.clone()))


def load_multiple_lora(model: torch.nn.Module, lora_state_dicts: List[Dict[str, torch.Tensor]]):
    model.apply(apply_to_lora(_prepare_for_multiple_lora))
    for state_dict in lora_state_dicts:
        _ = model.load_state_dict(state_dict, strict=False)
        model.apply(apply_to_lora(_append_lora))
    return model


def _select_lora(lora_layer: LoRAParametrization, index: int):
    lora_layer.lora_A = lora_layer.lora_As[index]
    lora_layer.lora_B = lora_layer.lora_Bs[index]


def select_lora(model: torch.nn.Module, index: int):
    model.apply(apply_to_lora(lambda x: _select_lora(x, index)))
    return model


# ------------------- helper function for tying and untieing weights -------------------


@no_type_check
def tie_weights(linear: nn.Linear, embedding: nn.Embedding):
    """tie the weights of the linear layer and the embedding layer both with the same lora"""
    # this line below is optional if the original is already tied
    embedding.parametrizations.weight.original = linear.parametrizations.weight.original
    embedding.parametrizations.weight[0].lora_A = linear.parametrizations.weight[0].lora_B
    embedding.parametrizations.weight[0].lora_B = linear.parametrizations.weight[0].lora_A


@no_type_check
def untie_weights(linear: nn.Linear, embedding: nn.Embedding):
    """untie the weights of the linear layer and the embedding layer"""
    embedding.parametrizations.weight.original = nn.Parameter(embedding.weight.original.clone())
    embedding.parametrizations.weight[0].lora_A = nn.Parameter(embedding.parametrizations.weight[0].lora_A.clone())
    embedding.parametrizations.weight[0].lora_B = nn.Parameter(embedding.parametrizations.weight[0].lora_B.clone())


def add_lora(model: torch.nn.Module, model_args: ModelArgs):
    target_modules = model_args.lora_targets
    rank = model_args.lora_rank
    alpha = model_args.lora_alpha
    dropout = model_args.lora_dropout

    lora_config = {
        torch.nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=rank, lora_alpha=alpha, lora_dropout_p=dropout),
        },
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=alpha, lora_dropout_p=dropout),
        },
        torch.nn.Conv1d: {
            "weight": partial(LoRAParametrization.from_conv1d, rank=rank, lora_alpha=alpha, lora_dropout_p=dropout),
        },
        transformers.pytorch_utils.Conv1D: {
            "weight": partial(LoRAParametrization.from_conv1d, rank=rank, lora_alpha=alpha, lora_dropout_p=dropout),
        },
    }

    def _target_module(name: str):
        return any(t in name for t in target_modules)

    named_modules = list(model.named_modules())
    for n, m in named_modules:
        type_m = type(m)
        if _target_module(n) and type_m in lora_config.keys():
            for attr_name, parametrization in lora_config[type_m].items():
                p = getattr(m, attr_name)
                if p.data.dtype != torch.get_default_dtype():
                    # fix it otherwise we have to set unsafe=true
                    p.data = p.data.to(torch.get_default_dtype())
                    setattr(m, attr_name, p)
                parametrize.register_parametrization(m, attr_name, parametrization(m))
