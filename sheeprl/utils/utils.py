import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import rich.syntax
import rich.tree
import torch
import torch.nn as nn
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


def nstep_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    nstep_horizon: int,
    gamma: float,
) -> np.ndarray:
    """
    Compute the nstep return of the trajectory. If the trajectory has less then nsteps, uses all the steps.

    Args:
        rewards (np.ndarray): all rewards collected in the trajectory.
        values (np.ndarray): all values collected in the trajectory.
        dones (np.ndarray): all dones collected in the trajectory.
        nstep_horizon (int): the number of steps to use for computing the estimate.
        gamma (float): discout factor.

    Returns:
        Estimated returns.
    """
    if nstep_horizon > rewards.shape[0]:
        nstep_horizon = rewards.shape[0]

    # create the gammas vector of len num_steps
    extra_shapes = rewards.shape[1:]
    gammas = np.array([gamma**i for i in range(nstep_horizon)]).reshape(-1, *extra_shapes)

    # for each time step, use the num_steps rewards and values to compute the nstep return
    returns = np.zeros(rewards.shape)
    for t in range(rewards.shape[0]):
        n_to_consider = min(nstep_horizon, rewards.shape[0] - t)
        returns[t] = (rewards[t : t + n_to_consider] * gammas[:n_to_consider]).sum() + (
            gamma**n_to_consider
        ) * values[t + n_to_consider - 1 : t + n_to_consider] * ~dones[t + n_to_consider - 1 : t + n_to_consider]

    return returns


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    done_mask: Tensor,
    last_values: Tensor,
    horizon: int = 15,
    lmbda: float = 0.95,
) -> Tensor:
    """
    Compute the lambda values by keeping the gradients of the variables.

    Args:
        rewards (Tensor): the estimated rewards in the latent space.
        values (Tensor): the estimated values in the latent space.
        done_mask (Tensor): 1s for the entries that are relative to a terminal step, 0s otherwise.
        last_values (Tensor): the next values for the last state in the horzon.
        horizon: (int, optional): the horizon of imagination.
            Default to 15.
        lmbda (float, optional): the discout lmbda factor for the lambda values computation.
            Default to 0.95.

    Returns:
        The tensor of the computed lambda values.
    """
    last_values = torch.clone(last_values)
    last_lambda_values = 0
    lambda_targets = []
    for step in reversed(range(horizon - 1)):
        if step == horizon - 2:
            next_values = last_values
        else:
            next_values = values[step + 1] * (1 - lmbda)
        delta = rewards[step] + next_values * done_mask[step]
        last_lambda_values = delta + lmbda * done_mask[step] * last_lambda_values
        lambda_targets.append(last_lambda_values)
    return torch.stack(list(reversed(lambda_targets)), dim=0)


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


def two_hot_encoder(array: np.ndarray, support_range: int = 300, num_buckets: Optional[int] = None) -> Tensor:
    """Encode a tensor representing a floating point number `x` as a tensor with all zeros except for two entries in the
    indexes of the two buckets closer to `x` in the support of the distribution.
    Check https://arxiv.org/pdf/2301.04104v1.pdf equation 9 for more details.

    Args:
        array (np.ndarray): array to encode of shape (..., batch_size, 1)
        support_range (int): range of the support of the distribution, going from -support_range to support_range
        num_buckets (int): number of buckets in the support of the distribution

    Returns:
        Tensor: tensor of shape (..., batch_size, support_size)
    """
    if len(array.shape) < 2:
        missing_dims = 2 - len(array.shape)
        array = np.expand_dims(array, tuple(range(missing_dims)))
    if num_buckets is None:
        num_buckets = support_range * 2 + 1
    if num_buckets % 2 == 0:
        raise ValueError("support_size must be odd")
    array = array.clip(-support_range, support_range)
    buckets = np.linspace(-support_range, support_range, num_buckets)
    bucket_size = buckets[1] - buckets[0] if buckets.shape[0] > 1 else 1.0

    right_idxs = np.digitize(array, buckets)
    left_idxs = (right_idxs - 1).clip(min=0)

    two_hot = np.zeros((array.shape[:-1] + (num_buckets,)))
    left_value = np.abs(buckets[right_idxs] - array) / bucket_size
    right_value = 1 - left_value

    batch_idx = np.arange(array.shape[-2])
    # Use advanced indexing to accumulate values
    two_hot[..., batch_idx, left_idxs.squeeze()] = left_value.squeeze()
    two_hot[..., batch_idx, right_idxs.squeeze()] = right_value.squeeze()

    return two_hot


def two_hot_decoder(array: np.ndarray, support_range: int) -> float:
    """Decode a tensor representing a two-hot vector as a tensor of floating point numbers.

    Args:
        array (np.ndarray): array to decode of shape (..., batch_size, support_size)
        support_range (int): range of the support of the values, going from -support_range to support_range

    Returns:
        Tensor: tensor of shape (..., batch_size, 1)
    """
    num_buckets = array.shape[-1]
    if num_buckets % 2 == 0:
        raise ValueError("support_size must be odd")
    support = np.linspace(-support_range, support_range, num_buckets)
    return np.sum(array * support, axis=-1, keepdims=True)


def symsqrt(x: np.ndarray, eps=0.001) -> np.ndarray:
    """Scales the tensor using the formula sign(x) * (sqrt(abs(x) + 1) - 1 + eps * x)."""
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x


def inverse_symsqrt(x, eps=0.001) -> np.ndarray:
    """Inverts symsqrt."""
    return np.sign(x) * (((np.sqrt(1 + 4 * eps * (np.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)


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
