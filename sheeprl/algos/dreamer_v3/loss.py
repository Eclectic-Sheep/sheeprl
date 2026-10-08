from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution


def categorical_kl(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """The KL divergence between independent categorical distributions, given by the logits of their classes (the
    last dimension), summed over the variables (the dimension before): the one of
    `kl_divergence(Independent(OneHotCategorical(logits=p_logits), 1), Independent(OneHotCategorical(logits=q_logits),
    1))`, computed as PyTorch does, without its distributions (which `torch.compile` handles less well)."""
    p_log_probs = p_logits - p_logits.logsumexp(dim=-1, keepdim=True)
    q_log_probs = q_logits - q_logits.logsumexp(dim=-1, keepdim=True)
    p_probs, q_probs = p_log_probs.softmax(dim=-1), q_log_probs.softmax(dim=-1)
    t = p_probs * (p_log_probs - q_log_probs)
    t = torch.where(q_probs == 0, torch.inf, t)
    t = torch.where(p_probs == 0, 0.0, t)
    return t.sum(dim=-1).sum(dim=-1)


def reconstruction_loss(
    po: Dict[str, Distribution],
    observations: Tensor,
    pr: Distribution,
    rewards: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    kl_dynamic: float = 0.5,
    kl_representation: float = 0.1,
    kl_free_nats: float = 1.0,
    kl_regularizer: float = 1.0,
    pc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 5 in
    [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).

    Args:
        po (Dict[str, Distribution]): the distribution returned by the observation_model (decoder).
        observations (Tensor): the observations provided by the environment.
        pr (Distribution): the reward distribution returned by the reward_model.
        rewards (Tensor): the rewards obtained by the agent during the "Environment interaction" phase.
        priors_logits (Tensor): the logits of the prior.
        posteriors_logits (Tensor): the logits of the posterior.
        kl_dynamic (float): the kl-balancing dynamic loss regularizer.
            Defaults to 0.5.
        kl_balancing_alpha (float): the kl-balancing representation loss regularizer.
            Defaults to 0.1.
        kl_free_nats (float): lower bound of the KL divergence.
            Default to 1.0.
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
        KL divergence (Tensor): the KL divergence between the posterior and the prior.
        reward_loss (Tensor): the value of the reward loss.
        state_loss (Tensor): the value of the state loss.
        continue_loss (Tensor): the value of the continue loss (0 if it is not computed).
        reconstruction_loss (Tensor): the value of the overall reconstruction loss.
        step_losses (Tensor): the overall reconstruction loss of every step, without gradients (the priorities of
            Curious Replay): the reconstruction loss is their mean.
    """
    rewards.device
    observation_loss = -sum([po[k].log_prob(observations[k]) for k in po.keys()])
    reward_loss = -pr.log_prob(rewards)
    # KL balancing
    dyn_loss = kl = categorical_kl(posteriors_logits.detach(), priors_logits)
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)
    repr_loss = categorical_kl(posteriors_logits, priors_logits.detach())
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    kl_loss = dyn_loss + repr_loss
    if pc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continue_targets)
    else:
        continue_loss = torch.zeros_like(reward_loss)
    step_losses = kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss
    return (
        step_losses.mean(),
        kl.mean(),
        kl_loss.mean(),
        reward_loss.mean(),
        observation_loss.mean(),
        continue_loss.mean(),
        step_losses.detach(),
    )
