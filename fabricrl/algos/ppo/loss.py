from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution


def policy_loss(
    dist: Distribution,
    actions: Tensor,
    logprobs: Tensor,
    advantages: Tensor,
    clip_coef: float,
    reduction: Optional[str] = "mean",
) -> Tensor:
    """Compute the policy loss for a batch of data, as described in equation (7) of the paper.

        - Compute the logprobs using the updated model for the actions taken.
        - Compute the difference between the new and old logprobs.
        - Exponentiate it to find the ratio.
        - Use the ratio and advantages to compute the loss as per equation (7).

    Args:
        dist (Distribution): the policy distribution.
        actions (Tensor): the actions sampled.
        logprobs (Tensor): the log-probs of the actions.
        advantages (Tensor): the advantages.
        clip_coef (float): the clipping coefficient.

    Returns:
        the policy loss
    """
    new_logprobs = dist.log_prob(actions)
    logratio = new_logprobs - logprobs
    ratio = logratio.exp()

    pg_loss1 = advantages * ratio
    pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.min(pg_loss1, pg_loss2)
    if reduction is None:
        return pg_loss
    elif reduction.lower() == "mean":
        return pg_loss.mean()
    elif reduction.lower() == "sum":
        return pg_loss.sum()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")


def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
    reduction: Optional[str] = "mean",
) -> Tensor:
    if not clip_vloss:
        values_pred = new_values
    else:
        values_pred = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    return F.mse_loss(values_pred, returns, reduction=reduction)
