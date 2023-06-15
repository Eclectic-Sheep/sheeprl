import torch.nn.functional as F
from torch import Tensor


def policy_loss(
    logprobs: Tensor,
    advantages: Tensor,
    reduction: str = "mean",
) -> Tensor:
    pg_loss = -logprobs * advantages.detach()
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
