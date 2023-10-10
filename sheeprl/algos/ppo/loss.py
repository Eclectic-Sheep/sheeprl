import torch
import torch.nn.functional as F
from torch import Tensor


def policy_loss(
    new_logprobs: Tensor,
    logprobs: Tensor,
    advantages: Tensor,
    clip_coef: float,
    reduction: str = "mean",
) -> Tensor:
    """Compute the policy loss for a batch of data, as described in equation (7) of the paper.

        - Compute the difference between the new and old logprobs.
        - Exponentiate it to find the ratio.
        - Use the ratio and advantages to compute the loss as per equation (7).

    Args:
        new_logprobs (Tensor): the log-probs of the new actions.
        logprobs (Tensor): the log-probs of the sampled actions from the environment.
        advantages (Tensor): the advantages.
        clip_coef (float): the clipping coefficient.

    Returns:
        the policy loss
    """
    logratio = new_logprobs - logprobs
    ratio = logratio.exp()

    pg_loss1 = advantages * ratio
    pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = -torch.min(pg_loss1, pg_loss2)
    reduction = reduction.lower()
    if reduction == "none":
        return pg_loss
    elif reduction == "mean":
        return pg_loss.mean()
    elif reduction == "sum":
        return pg_loss.sum()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")


def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
    reduction: str = "mean",
) -> Tensor:
    if not clip_vloss:
        values_pred = new_values
        # return F.mse_loss(values_pred, returns, reduction=reduction)
    else:
        values_pred = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
        # return torch.max((new_values - returns) ** 2, (values_pred - returns) ** 2).mean()
    return F.mse_loss(values_pred, returns, reduction=reduction)


def entropy_loss(entropy: Tensor, reduction: str = "mean") -> Tensor:
    ent_loss = -entropy
    reduction = reduction.lower()
    if reduction == "none":
        return ent_loss
    elif reduction == "mean":
        return ent_loss.mean()
    elif reduction == "sum":
        return ent_loss.sum()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")
