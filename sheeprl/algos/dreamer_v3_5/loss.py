"""The output distributions and the returns of DreamerV3 (Nature version), from `embodied/jax/outs.py`,
`embodied/jax/heads.py` and `dreamerv3/agent.py` of https://github.com/danijar/dreamerv3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from sheeprl.utils.utils import symexp, symlog


def symexp_bins(bins: int, device: torch.device | str | None = None) -> Tensor:
    """The `bins` bins of the two-hot distributions: `symexp(linspace(-20, 20, bins))`, with the negative half the
    exact opposite of the positive one (an odd number of bins has an exact zero in the middle)."""
    if bins % 2 == 1:
        half = symexp(torch.linspace(-20, 0, (bins - 1) // 2 + 1, device=device))
        return torch.cat([half, -half[:-1].flip(0)], 0)
    half = symexp(torch.linspace(-20, 0, bins // 2, device=device))
    return torch.cat([half, -half.flip(0)], 0)


class TwoHot:
    """The two-hot distribution of the rewards and of the values (`symexp_twohot`): a categorical over bins spaced
    exponentially (`symexp_bins`). Its mean is the average of the bins weighted by their probabilities, in the space of
    the values, and its target puts the weights of the two bins around a value in proportion to their distance from it,
    so that the target has the value as its mean (DreamerV3 of the 2023 paper did both in the symlog space).

    Args:
        logits: the logits, the bins in the last dimension.
        bins: the bins (`symexp_bins`); default: the ones of the number of logits.
    """

    def __init__(self, logits: Tensor, bins: Tensor | None = None) -> None:
        self.logits = logits.float()
        self.bins = symexp_bins(logits.shape[-1], logits.device) if bins is None else bins

    @property
    def mean(self) -> Tensor:
        """The mean, summed symmetrically from the middle: with symmetric bins and uniform probabilities (as at the
        initialization of the heads with zero weights) it is exactly 0. The last dimension (the bins) is removed."""
        probs = self.logits.softmax(-1)
        n = probs.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            sides = (probs[..., :m] * self.bins[:m]).flip(-1) + probs[..., m + 1 :] * self.bins[m + 1 :]
            return probs[..., m] * self.bins[m] + sides.sum(-1)
        m = n // 2
        return ((probs[..., :m] * self.bins[:m]).flip(-1) + probs[..., m:] * self.bins[m:]).sum(-1)

    def loss(self, target: Tensor) -> Tensor:
        """The cross-entropy with the two-hot encoding of `target` (of the shape of the mean)."""
        return -(self.encode(target) * self.logits.log_softmax(-1)).sum(-1)

    def encode(self, target: Tensor) -> Tensor:
        """The two-hot encoding of `target`: the weights of the two bins around every value, whose average is the
        value (the bins at the ends for the values beyond them)."""
        target = target.detach().float()
        n = len(self.bins)
        below = (self.bins <= target[..., None]).sum(-1) - 1
        above = n - (self.bins > target[..., None]).sum(-1)
        below = below.clamp(0, n - 1)
        above = above.clamp(0, n - 1)
        equal = below == above
        dist_to_below = torch.where(equal, 1, (self.bins[below] - target).abs())
        dist_to_above = torch.where(equal, 1, (self.bins[above] - target).abs())
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        return F.one_hot(below, n) * weight_below[..., None] + F.one_hot(above, n) * weight_above[..., None]


def binary_loss(logits: Tensor, target: Tensor) -> Tensor:
    """The negative log-likelihood of `target` (in [0, 1], also soft) under the Bernoulli of `logits`."""
    logits = logits.float()
    return -(target * F.logsigmoid(logits) + (1 - target) * F.logsigmoid(-logits))


def symlog_mse(prediction: Tensor, target: Tensor, dims: int) -> Tensor:
    """The squared error between `prediction` and the symlog of `target`, summed over the last `dims` dimensions."""
    return (prediction.float() - symlog(target.float())).square().sum(tuple(range(-dims, 0)))


def mse(prediction: Tensor, target: Tensor, dims: int) -> Tensor:
    """The squared error, summed over the last `dims` dimensions."""
    return (prediction.float() - target.float()).square().sum(tuple(range(-dims, 0)))


def lambda_return(
    last: Tensor, terminal: Tensor, rewards: Tensor, bootstrap: Tensor, discount: float, lmbda: float
) -> Tensor:
    """The lambda-returns of the trajectories, time first (`lambda_return` of DreamerV3).

    The return of the step `t` is `r[t+1] + live[t+1] * ((1 - cont[t+1]) * boot[t+1] + cont[t+1] * ret[t+1])`, with
    `live = (1 - terminal) * discount` and `cont = (1 - last) * lmbda`, and the return of the last step is its
    bootstrap: the return stops at the end of an episode (`last`), where it bootstraps from the value unless the
    episode terminated.

    Args:
        last: whether the steps end an episode, of shape `[T, ...]`.
        terminal: whether the episodes terminated at the steps (soft values are allowed: one minus the continue
            probabilities), of shape `[T, ...]`.
        rewards: the rewards of the steps, of shape `[T, ...]`.
        bootstrap: the values of the steps, of shape `[T, ...]`.
        discount: the discount factor.
        lmbda: the lambda of the returns.

    Returns:
        The returns of the first `T - 1` steps, of shape `[T - 1, ...]`.
    """
    live = (1 - terminal.to(rewards.dtype))[1:] * discount
    cont = (1 - last.to(rewards.dtype))[1:] * lmbda
    interm = rewards[1:] + (1 - cont) * live * bootstrap[1:]
    returns = [bootstrap[-1]]
    for t in reversed(range(live.shape[0])):
        returns.append(interm[t] + live[t] * cont[t] * returns[-1])
    return torch.stack(list(reversed(returns))[:-1], 0)
