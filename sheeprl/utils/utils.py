from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rich.syntax
import rich.tree
import torch
import torch.nn as nn
from lightning.fabric.wrappers import _FabricModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

NUMPY_TO_TORCH_DTYPE_DICT = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}
TORCH_TO_NUMPY_DTYPE_DICT = {value: key for key, value in NUMPY_TO_TORCH_DTYPE_DICT.items()}


class dotdict(dict):
    """
    A dictionary supporting dot notation.
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = dotdict(v)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def as_dict(self) -> Dict[str, Any]:
        _copy = dict(self)
        for k, v in _copy.items():
            if isinstance(v, dotdict):
                _copy[k] = v.as_dict()
        return _copy


@torch.no_grad()
def gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    num_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """Compute returns and advantages following https://arxiv.org/abs/1506.02438

    Args:
        rewards (Tensor): all rewards collected from the last rollout
        values (Tensor): all values collected from the last rollout
        dones (Tensor): all dones collected from the last rollout
        next_values (Tensor): estimated values for the next observations
        num_steps (int): the number of steps played
        gamma (float): discout factor
        gae_lambda (float): lambda for GAE estimation

    Returns:
        estimated returns
        estimated advantages
    """
    lastgaelam = 0
    nextvalues = next_value
    not_dones = torch.logical_not(dones)
    nextnonterminal = not_dones[-1]
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(num_steps)):
        if t < num_steps - 1:
            nextnonterminal = not_dones[t]
            nextvalues = values[t + 1]
        delta = rewards[t] + nextvalues * nextnonterminal * gamma - values[t]
        advantages[t] = lastgaelam = delta + nextnonterminal * lastgaelam * gamma * gae_lambda
    returns = advantages + values
    return returns, advantages


def init_weights(m: nn.Module):
    """
    Initialize the parameters of the m module acording to the method described in
    [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852) using a uniform distribution.

    Args:
        m (nn.Module): the module to be initialized.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def normalize_tensor(tensor: Tensor, eps: float = 1e-8, mask: Optional[Tensor] = None) -> Tensor:
    unmasked = mask is None
    if unmasked:
        mask = torch.ones_like(tensor, dtype=torch.bool)
    masked_tensor = tensor[mask]
    normalized = (masked_tensor - masked_tensor.mean()) / (masked_tensor.std() + eps)
    if unmasked:
        return normalized.reshape_as(mask)
    else:
        return normalized


def polynomial_decay(
    current_step: int,
    *,
    initial: float = 1.0,
    final: float = 0.0,
    max_decay_steps: int = 100,
    power: float = 1.0,
) -> float:
    if current_step > max_decay_steps or initial == final:
        return final
    else:
        return (initial - final) * ((1 - current_step / max_decay_steps) ** power) + final


# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot_encoder(tensor: Tensor, support_range: int = 300, num_buckets: Optional[int] = None) -> Tensor:
    """Encode a tensor representing a floating point number `x` as a tensor with all zeros except for two entries in the
    indexes of the two buckets closer to `x` in the support of the distribution.
    Check https://arxiv.org/pdf/2301.04104v1.pdf equation 9 for more details.

    Args:
        tensor (Tensor): tensor to encode of shape (..., batch_size, 1)
        support_range (int): range of the support of the distribution, going from -support_range to support_range
        num_buckets (int): number of buckets in the support of the distribution

    Returns:
        Tensor: tensor of shape (..., batch_size, support_size)
    """
    if tensor.shape == torch.Size([]):
        tensor = tensor.unsqueeze(0)
    if num_buckets is None:
        num_buckets = support_range * 2 + 1
    if num_buckets % 2 == 0:
        raise ValueError("support_size must be odd")
    tensor = tensor.clip(-support_range, support_range)
    buckets = torch.linspace(-support_range, support_range, num_buckets, device=tensor.device)
    bucket_size = buckets[1] - buckets[0] if len(buckets) > 1 else 1.0

    right_idxs = torch.bucketize(tensor, buckets)
    left_idxs = (right_idxs - 1).clip(min=0)

    two_hot = torch.zeros(tensor.shape[:-1] + (num_buckets,), device=tensor.device)
    left_value = torch.abs(buckets[right_idxs] - tensor) / bucket_size
    right_value = 1 - left_value
    two_hot.scatter_add_(-1, left_idxs, left_value)
    two_hot.scatter_add_(-1, right_idxs, right_value)

    return two_hot


def two_hot_decoder(tensor: torch.Tensor, support_range: int) -> torch.Tensor:
    """Decode a tensor representing a two-hot vector as a tensor of floating point numbers.

    Args:
        tensor (Tensor): tensor to decode of shape (..., batch_size, support_size)
        support_range (int): range of the support of the values, going from -support_range to support_range

    Returns:
        Tensor: tensor of shape (..., batch_size, 1)
    """
    num_buckets = tensor.shape[-1]
    if num_buckets % 2 == 0:
        raise ValueError("support_size must be odd")
    support = torch.linspace(-support_range, support_range, num_buckets).to(tensor.device)
    return torch.sum(tensor * support, dim=-1, keepdim=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = ("algo", "buffer", "checkpoint", "env", "fabric", "metric"),
    resolve: bool = True,
    cfg_save_path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config: Configuration composed by Hydra.
        fields: Determines which main fields from config will
            be printed and in what order.
        resolve: Whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    if cfg_save_path is not None:
        with open(os.path.join(os.getcwd(), "config_tree.txt"), "w") as fp:
            rich.print(tree, file=fp)


def unwrap_fabric(model: _FabricModule | nn.Module) -> nn.Module:
    """Recursively unwrap the model from _FabricModule. This method returns a deep copy of the model.

    Args:
        model (_FabricModule | nn.Module): the model to unwrap.

    Returns:
        nn.Module: the unwrapped model.
    """
    model = copy.deepcopy(model)
    if isinstance(model, _FabricModule):
        model = model.module
    for name, child in model.named_children():
        setattr(model, name, unwrap_fabric(child))
    return model


def save_configs(cfg: dotdict, log_dir: str):
    OmegaConf.save(cfg.as_dict(), os.path.join(log_dir, "config.yaml"), resolve=True)


class Ratio:
    """Directly taken from Hafner et al. (2023) implementation:
    https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/when.py#L26
    """

    def __init__(self, ratio: float, pretrain_steps: int = 0):
        if pretrain_steps < 0:
            raise ValueError(f"'pretrain_steps' must be non-negative, got {pretrain_steps}")
        if ratio < 0:
            raise ValueError(f"'ratio' must be non-negative, got {ratio}")
        self._pretrain_steps = pretrain_steps
        self._ratio = ratio
        self._prev = None

    def __call__(self, step: int) -> int:
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            repeats = 1
            if self._pretrain_steps > 0:
                if step < self._pretrain_steps:
                    warnings.warn(
                        "The number of pretrain steps is greater than the number of current steps. This could lead to "
                        f"a higher ratio than the one specified ({self._ratio}). Setting the 'pretrain_steps' equal to "
                        "the number of current steps."
                    )
                    self._pretrain_steps = step
                repeats = int(self._pretrain_steps * self._ratio)
            return repeats
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats

    def state_dict(self) -> Dict[str, Any]:
        return {"_ratio": self._ratio, "_prev": self._prev, "_pretrain_steps": self._pretrain_steps}

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        self._ratio = state_dict["_ratio"]
        self._prev = state_dict["_prev"]
        self._pretrain_steps = state_dict["_pretrain_steps"]
        return self
