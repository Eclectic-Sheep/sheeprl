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
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.wrappers import _FabricModule
from omegaconf import DictConfig, OmegaConf
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
    """Zero mean and unit standard deviation (of the elements selected by `mask`: the tensor of the same shape is
    returned). A single element has no standard deviation: it is returned as it is (as Stable-Baselines3 does with the
    advantages of a one-element minibatch)."""
    if mask is None:
        # No selection: the shape doesn't depend on the data (`torch.compile` doesn't break the graph)
        if tensor.numel() < 2:
            return tensor
        return (tensor - tensor.mean()) / (tensor.std() + eps)
    # The statistics of the selected elements from masked sums, without selecting them: the shape doesn't depend on the
    # data. The elements not selected are normalized too, with the same statistics
    count = mask.sum()
    mean = torch.where(mask, tensor, 0).sum() / count.clamp(min=1)
    var = torch.where(mask, (tensor - mean) ** 2, 0).sum() / (count - 1).clamp(min=1)
    return torch.where(count > 1, (tensor - mean) / (var.sqrt() + eps), tensor)


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
    model = copy.deepcopy(getattr(model, "module", model))
    for name, child in model.named_children():
        setattr(model, name, unwrap_fabric(child))
    return model


def save_configs(cfg: dotdict, log_dir: str):
    OmegaConf.save(cfg.as_dict(), os.path.join(log_dir, "config.yaml"), resolve=True)


class Ratio:
    """Directly taken from Hafner et al. (2023) implementation:
    https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/when.py#L26
    """

    def __init__(self, ratio: float):
        if ratio < 0:
            raise ValueError(f"'ratio' must be non-negative, got {ratio}")
        self._ratio = ratio
        self._prev = None

    def __call__(self, step: int) -> int:
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            return int(step * self._ratio)
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats

    def realign(self, step: float) -> None:
        """Continue from `step`, the step of the last call before the state of the ratio was saved.

        A ratio continues from the step of its last call minus less than one gradient step (`1 / ratio` steps). The
        state saved by a run that counted its steps from another start (a run resumed by an older version, which counted
        them from where it resumed) is behind or ahead of `step` by more: the first call would do all the gradient steps
        in between at once, or none for a while. Such a ratio continues from `step`, with a warning.
        """
        if self._ratio == 0 or self._prev is None:
            return
        behind = step - self._prev
        if -1e-6 <= behind < 1 / self._ratio + 1e-6:
            return
        warnings.warn(
            f"The replay ratio of the checkpoint counted {self._prev} steps, but the run resumes after {step}: it "
            f"continues from {step} instead of doing {int(max(behind, 0) * self._ratio)} gradient steps at once"
        )
        self._prev = step

    def state_dict(self) -> Dict[str, Any]:
        return {"_ratio": self._ratio, "_prev": self._prev}

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        # The checkpoints saved when the ratio did the pretraining also have `_pretrain_steps`
        self._ratio = state_dict["_ratio"]
        self._prev = state_dict["_prev"]
        return self


def off_policy_schedule(
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    start_iter: int,
    total_iters: int,
    policy_steps_per_iter: int,
    world_size: int,
) -> Tuple[int, int, int, int, Ratio]:
    """When an off-policy run plays random actions and trains, and its replay ratio.

    The run plays random actions in its first `algo.learning_starts` policy steps (rounded down to whole iterations),
    filling its replay buffer, and trains from the end of the last of them (from the first iteration without them),
    `algo.replay_ratio` gradient steps per policy step played from the start of that iteration. Its first training
    also does `algo.per_rank_pretrain_steps` more gradient steps on the filled buffer (the `pretrain` of DreamerV1 and
    DreamerV2), but in a dry run. The algorithms that train on sequences of `algo.per_rank_sequence_length` steps of
    every environment (Dreamer) start training only when every environment has played that many steps (an iteration
    plays one step of every environment), also when `algo.learning_starts` comes earlier; a dry run lasts until then.

    A resumed run continues as the run it resumes: it doesn't play random actions again, and with the replay buffer
    of the checkpoint it trains from its first iteration, with the ratio of the checkpoint. Without it
    (`buffer.checkpoint=False`) it fills a new one first, playing its policy for `algo.learning_starts` policy steps,
    then trains as a new run.

    Args:
        cfg (Dict[str, Any]): the configuration of the run.
        state (Dict[str, Any]): the checkpoint the run resumes from (unused when it doesn't).
        start_iter (int): the first iteration of the run (1, or the one after the checkpoint).
        total_iters (int): the last iteration of the run.
        policy_steps_per_iter (int): the policy steps of one iteration, played by all the processes.
        world_size (int): the number of processes.

    Returns:
        The last iteration that plays random actions (the ones from 1 to it do), the first iteration that trains, the
        gradient steps its training adds to the ones of the ratio, the last iteration of the run (a dry run lasts until
        its first training), and the replay ratio, which gives the gradient steps of every process from the policy
        steps played from the start of the first iteration that trains, divided by the number of processes.
    """
    learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
    pretrain_steps = cfg.algo.per_rank_pretrain_steps if not cfg.dry_run else 0
    if pretrain_steps < 0:
        raise ValueError(f"`algo.per_rank_pretrain_steps` must be non-negative, got {pretrain_steps}")
    # The algorithms that sample sequences of steps of every environment wait for the first ones to be played
    sequence_length = cfg.algo.get("per_rank_sequence_length") or 0
    fill_iters = max(learning_starts, sequence_length, 1)
    if fill_iters > max(learning_starts, 1) and not cfg.dry_run:
        warnings.warn(
            f"The training starts after {fill_iters * policy_steps_per_iter} policy steps, not after "
            f"`algo.learning_starts={cfg.algo.learning_starts}`: it samples sequences of "
            f"`algo.per_rank_sequence_length={sequence_length}` steps of every environment"
        )
    refill = bool(cfg.checkpoint.resume_from) and not cfg.buffer.checkpoint
    train_starts = (start_iter - 1 if refill else 0) + fill_iters
    if cfg.dry_run:
        # A dry run lasts until its first training
        total_iters = max(total_iters, train_starts)
    ratio = Ratio(cfg.algo.replay_ratio)
    if cfg.checkpoint.resume_from and not refill:
        ratio.load_state_dict(state["ratio"])
        # The policy steps of every process counted by the ratio at the end of the iteration of the checkpoint
        ratio.realign((start_iter - train_starts) * policy_steps_per_iter / world_size)
    return learning_starts, train_starts, pretrain_steps, total_iters, ratio


# https://github.com/pytorch/rl/blob/824f6d192e88c115790cf046e4df416ce2d7aaf6/torchrl/modules/distributions/utils.py#L156
def safetanh(x, eps):
    lim = 1.0 - eps
    y = x.tanh()
    return y.clamp(-lim, lim)


# https://github.com/pytorch/rl/blob/824f6d192e88c115790cf046e4df416ce2d7aaf6/torchrl/modules/distributions/utils.py#L161
def safeatanh(y, eps):
    lim = 1.0 - eps
    return y.clamp(-lim, lim).atanh()
