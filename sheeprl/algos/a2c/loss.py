import torch.nn.functional as F
from torch import Tensor


def policy_loss(
    logprobs: Tensor,
    advantages: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute the policy loss for a batch of data, as described in equation (7) of the paper.

        - Compute the difference between the new and old logprobs.
        - Exponentiate it to find the ratio.
        - Use the ratio and advantages to compute the loss as per equation (7).

    Args:
        logprobs (Tensor): the log-probs of the sampled actions from the environment.
        advantages (Tensor): the advantages.

    Returns:
        the policy loss
    """
    pg_loss = -(logprobs * advantages)
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
    values: Tensor,
    returns: Tensor,
    reduction: str = "mean",
) -> Tensor:
    return F.mse_loss(values, returns, reduction=reduction)
