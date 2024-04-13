from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence


def reconstruction_loss(
    po: Dict[str, Distribution],
    observations: Dict[str, Tensor],
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
    discount_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 2 in
    [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

    Args:
        po (Dict[str, Distribution]): the distribution returned by the observation_model (decoder).
        observations (Dict[str, Tensor]): the observations provided by the environment.
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
        pc (Distribution, optional): the predicted Bernoulli distribution of the terminal steps.
            0s for the entries that are relative to a terminal step, 1s otherwise.
            Default to None.
        continue_targets (Tensor, optional): the targets for the discount predictor. Those are normally computed
            as `(1 - data["dones"]) * args.gamma`.
            Default to None.
        discount_scale_factor (float): the scale factor for the continue loss.
            Default to 1.0.

    Returns:
        observation_loss (Tensor): the value of the observation loss.
        kl divergence (Tensor): the KL between posterior and prior state.
        reward_loss (Tensor): the value of the reward loss.
        state_loss (Tensor): the value of the state loss.
        continue_loss (Tensor): the value of the continue loss (0 if it is not computed).
        reconstruction_loss (Tensor): the value of the overall reconstruction loss.
    """
    observation_loss = -sum([po[k].log_prob(observations[k]).mean() for k in po.keys()])
    reward_loss = -pr.log_prob(rewards).mean()
    # KL balancing
    lhs = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    rhs = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    if kl_free_avg:
        lhs = lhs.mean()
        rhs = rhs.mean()
        free_nats = torch.full_like(lhs, kl_free_nats)
        loss_lhs = torch.maximum(lhs, free_nats)
        loss_rhs = torch.maximum(rhs, free_nats)
    else:
        free_nats = torch.full_like(lhs, kl_free_nats)
        loss_lhs = torch.maximum(lhs, kl_free_nats).mean()
        loss_rhs = torch.maximum(rhs, kl_free_nats).mean()
    kl_loss = kl_balancing_alpha * loss_lhs + (1 - kl_balancing_alpha) * loss_rhs
    if pc is not None and continue_targets is not None:
        continue_loss = discount_scale_factor * -pc.log_prob(continue_targets).mean()
    else:
        continue_loss = torch.zeros_like(reward_loss)
    reconstruction_loss = kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss
    return reconstruction_loss, kl, kl_loss, reward_loss, observation_loss, continue_loss
