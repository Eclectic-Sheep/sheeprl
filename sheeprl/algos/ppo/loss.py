import torch
import torch.nn.functional as F
from torch import Tensor


def reduce_loss(loss: Tensor, reduction: str = "mean") -> Tensor:
    reduction = reduction.lower()
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")


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
    return reduce_loss(pg_loss, reduction)


def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
    reduction: str = "mean",
) -> Tensor:
    """The squared error of the values from the returns, without a factor ½ (as in Stable-Baselines3: `algo.vf_coef`
    weighs this scale). With `clip_vloss`, the larger of the errors of the new values and of the new values clipped
    to `clip_coef` around the old ones."""
    if not clip_vloss:
        return F.mse_loss(new_values, returns, reduction=reduction)
    v_loss_unclipped = (new_values - returns) ** 2
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    v_loss_clipped = (v_clipped - returns) ** 2
    return reduce_loss(torch.max(v_loss_unclipped, v_loss_clipped), reduction)


def entropy_loss(entropy: Tensor, reduction: str = "mean") -> Tensor:
    return reduce_loss(-entropy, reduction)
