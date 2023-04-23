import math
from typing import Dict, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        if type(observation_space.n) in [tuple, list, np.ndarray]:
            return tuple(observation_space.n)
        else:
            return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def linear_annealing(optimizer: torch.optim.Optimizer, update: int, num_updates: int, initial_lr: float):
    frac = 1.0 - (update - 1.0) / num_updates
    lrnow = frac * initial_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lrnow


def layer_init(
    layer: torch.nn.Module,
    std: float = math.sqrt(2),
    bias_const: float = 0.0,
    ortho_init: bool = True,
):
    if ortho_init:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def conditional_arange(n: int, mask: Tensor) -> Tensor:
    rolled_mask = torch.roll(mask, 1, 0)
    rolled_mask[0] = 0
    cs = (torch.ones(n) * (1 - rolled_mask)).cumsum(dim=0)
    acc = torch.cummax(rolled_mask * cs, 0)[0]
    return cs - torch.where(acc > 0, acc - 1, 0) - 1


@torch.no_grad()
def estimate_returns_and_advantages(
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
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = torch.logical_not(next_done)
            nextvalues = next_value
        else:
            nextnonterminal = torch.logical_not(dones[t + 1])
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return returns, advantages


@torch.no_grad()
def vectorized_estimate_returns_and_advantages(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    next_done: Tensor,
    num_steps: int,
    gamma: float,
    gae_lambda: float,
):
    """Compute returns and advantages following https://arxiv.org/abs/1506.02438

    Args:
        rewards (Tensor): all rewards collected from the last rollout
        values (Tensor): all values collected from the last rollout
        dones (Tensor): all dones collected from the last rollout
        next_value (Tensor): next value
        next_done (Tensor): next done
        num_steps (int): the number of steps played
        gamma (float): discout factor
        gae_lambda (float): lambda for GAE estimation

    Returns:
        estimated returns
        estimated advantages
    """
    if len(rewards.shape) == 3:
        t_steps = torch.cat(
            [conditional_arange(num_steps, dones[:, dim, :].view(-1)).view(-1, 1) for dim in range(rewards.shape[1])],
            dim=1,
        ).unsqueeze(-1)
    elif len(rewards.shape) == 2:
        t_steps = conditional_arange(num_steps, dones.view(-1)).view(-1, 1)
    else:
        raise ValueError(f"Shape must be 2 or 3 dimensional, got {rewards.shape}")
    gt = torch.pow(gamma * gae_lambda, t_steps.view_as(dones))
    next_values = torch.roll(values, -1, dims=0)
    next_values[-1] = next_value
    next_dones = torch.roll(dones, -1, dims=0)
    next_dones[-1] = next_done
    deltas = rewards + gamma * next_values * (1 - next_dones) - values
    cs = torch.flipud(deltas * gt).cumsum(dim=0)
    acc = torch.cummax(torch.flipud(dones) * cs, 0)[0]
    acc[0] = 0
    dones[-1] = 0
    adv = torch.flipud(cs - acc) / gt
    adv = adv + dones * (deltas + gamma * gae_lambda * adv.roll(-1, 0))
    return adv + values, adv
