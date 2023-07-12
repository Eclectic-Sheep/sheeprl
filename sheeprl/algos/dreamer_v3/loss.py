from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence


def reconstruction_loss(
    po: Distribution,
    observations: Tensor,
    pr: Distribution,
    rewards: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    kl_balancing_alpha: float = 0.8,
    kl_free_nats: float = 0.0,
    kl_free_avg: bool = True,
    kl_regularizer: float = 1.0,
    pc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 2 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

    Args:
        po (Distribution): the distribution returned by the observation_model (decoder).
        observations (Tensor): the observations provided by the environment.
        pr (Distribution): the reward distribution returned by the reward_model.
        rewards (Tensor): the rewards obtained by the agent during the "Environment interaction" phase.
        priors_logits (Tensor): the logits of the prior.
        posteriors_logits (Tensor): the logits of the posterior.
        kl_balancing_alpha (float): the kl-balancing alpha value.
            Defaults to 0.8.
        kl_free_nats (float): lower bound of the KL divergence.
            Default to 0.0.
        kl_regularizer (float): scale factor of the KL divergence.
            Default to 1.0.
        pc (Bernoulli, optional): the predicted Bernoulli distribution of the terminal steps.
            0s for the entries that are relative to a terminal step, 1s otherwise.
            Default to None.
        continue_targets (Tensor, optional): the targets for the discount predictor. Those are normally computed
            as `(1 - data["dones"]) * args.gamma`.
            Default to None.
        continue_scale_factor (float): the scale factor for the continue loss.
            Default to 10.

    Returns:
        observation_loss (Tensor): the value of the observation loss.
        reward_loss (Tensor): the value of the reward loss.
        state_loss (Tensor): the value of the state loss.
        continue_loss (Tensor): the value of the continue loss (0 if it is not computed).
        reconstruction_loss (Tensor): the value of the overall reconstruction loss.
    """
    device = observations.device
    observation_loss = -po.log_prob(observations).mean()
    reward_loss = -pr.log_prob(rewards).mean()
    # KL balancing
    lhs = kl_divergence(
        OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()),
        OneHotCategoricalStraightThrough(logits=priors_logits),
    )
    rhs = kl_divergence(
        OneHotCategoricalStraightThrough(logits=posteriors_logits),
        OneHotCategoricalStraightThrough(logits=priors_logits.detach()),
    )
    kl_free_nats = torch.tensor([kl_free_nats], device=lhs.device)
    if kl_free_avg:
        loss_lhs = torch.maximum(lhs.mean(), kl_free_nats)
        loss_rhs = torch.maximum(rhs.mean(), kl_free_nats)
    else:
        loss_lhs = torch.maximum(lhs, kl_free_nats).mean()
        loss_rhs = torch.maximum(rhs, kl_free_nats).mean()
    kl_loss = kl_balancing_alpha * loss_lhs + (1 - kl_balancing_alpha) * loss_rhs
    continue_loss = torch.tensor(0, device=device)
    if pc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continue_targets).mean()
    reconstruction_loss = kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss
    return reconstruction_loss, kl_loss, reward_loss, observation_loss, continue_loss
