import os
from typing import Optional, Tuple

import gymnasium as gym
import torch
from torch import Tensor

from sheeprl.envs.wrappers import ActionRepeat, MaskVelocityWrapper


@torch.no_grad()
def gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    next_done: Tensor,
    num_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """Compute returns and advantages following https://arxiv.org/abs/1506.02438

    Args:
        rewards (Tensor): all rewards collected from the last rollout
        values (Tensor): all values collected from the last rollout
        dones (Tensor): all dones collected from the last rollout
        next_value (Tensor): next observation
        next_done (Tensor): next done
        num_steps (int): the number of steps played
        gamma (float): discout factor
        gae_lambda (float): lambda for GAE estimation

    Returns:
        estimated returns
        estimated advantages
    """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    not_done = torch.logical_not(dones)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = torch.logical_not(next_done)
            nextvalues = next_value
        else:
            nextnonterminal = not_done[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return returns, advantages


@torch.no_grad()
def normalize_tensor(tensor: Tensor, eps: float = 1e-8, mask: Optional[Tensor] = None):
    if mask is None:
        mask = torch.ones_like(tensor, dtype=torch.bool)
    return (tensor - tensor[mask].mean()) / (tensor[mask].std() + eps)


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


def make_env(
    env_id: str,
    seed: Optional[int],
    idx: int,
    capture_video: bool,
    run_name: Optional[str] = None,
    prefix: str = "",
    mask_velocities: bool = False,
    vector_env_idx: int = 0,
    action_repeat: int = 1,
):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if mask_velocities:
            env = MaskVelocityWrapper(env)
        env = ActionRepeat(env, action_repeat)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if vector_env_idx == 0 and idx == 0 and run_name is not None:
                env = gym.experimental.wrappers.RecordVideoV0(
                    env,
                    os.path.join(run_name, prefix + "_videos" if prefix else "videos"),
                    disable_logger=True,
                )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def two_hot_encoder(tensor: Tensor, support_size: int = 601) -> Tensor:
    """Encode a tensor representing a floating point number `x` as a tensor with all zeros except for two entries in the
    indexes -`support_size` / 2 + `floor(x)` and `support_size` / 2 + `ceil(x)`,
    which are set to `x - floor(x)` and `ceil(x) - x` respectively.

    Args:
        tensor (Tensor): tensor to encode of shape (..., batch_size, 1)
        support_size optional(int): size of the support of the distribution (default: 601)

    Returns:
        Tensor: tensor of shape (..., batch_size, support_size)
    """
    if tensor.shape == torch.Size([]):
        tensor = tensor.unsqueeze(0)
    if support_size % 2 == 0:
        raise ValueError("support_size must be odd")
    tensor = tensor.clip(-(support_size - 1) / 2, (support_size - 1) / 2)
    floor = tensor.floor().long()
    ceil = floor + 1
    floor_prob = ceil - tensor
    ceil_prob = 1.0 - floor_prob
    two_hot = torch.zeros_like(tensor).expand(*tensor.shape[:-1], support_size).clone()
    two_hot.scatter_add_(-1, floor + (support_size - 1) // 2, floor_prob)
    ceil = ceil.clip(-(support_size - 1) // 2, (support_size - 1) // 2)
    two_hot.scatter_add_(-1, ceil + (support_size - 1) // 2, ceil_prob)
    return two_hot


def two_hot_decoder(tensor: torch.Tensor) -> torch.Tensor:
    """Decode a tensor representing a two-hot vector as a tensor of floating point numbers.

    Args:
        tensor (Tensor): tensor to decode of shape (..., batch_size, support_size)

    Returns:
        Tensor: tensor of shape (..., batch_size, 1)
    """
    support_size = tensor.shape[-1]
    if support_size % 2 == 0:
        raise ValueError("support_size must be odd")
    support = torch.arange(-(support_size - 1) / 2, (support_size - 1) / 2 + 1).to(tensor.device)
    return torch.sum(tensor * support, dim=-1, keepdim=True)
