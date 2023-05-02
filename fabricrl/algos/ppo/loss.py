import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from torch import Tensor


def policy_loss(dist: torch.distributions.Distribution, batch: TensorDict, clip_coef: float) -> torch.Tensor:
    """Compute the policy loss for a batch of data, as described in equation (7) of the paper.

    - Compute the logprobs using the updated model for the actions taken.
    - Compute the difference between the new and old logprobs.
    - Exponentiate it to find the ratio.
    - Use the ratio and advantages to compute the loss as per equation (7).

    Args:
        dist (torch.distributions.Distribution): the policy distribution.
        batch (TensorDict): the batch of data.
        clip_coef (float): the clipping coefficient.

    Returns:
        the policy loss.
    """
    new_logprobs = dist.log_prob(batch["actions"])
    logratio = new_logprobs - batch["logprobs"]
    ratio = logratio.exp()
    advantages: Tensor = batch["advantages"]

    pg_loss1 = advantages * ratio
    pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.min(pg_loss1, pg_loss2).mean()
    return pg_loss


def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
) -> Tensor:
    if not clip_vloss:
        values_pred = new_values
    else:
        values_pred = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    return F.mse_loss(values_pred, returns)
