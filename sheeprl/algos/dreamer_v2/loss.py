from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategorical
from torch.distributions.kl import kl_divergence


def critic_loss(qv: Distribution, lambda_values: Tensor, discount: Tensor) -> Tensor:
    """
    Compute the critic loss as described in Eq. 8 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    Args:
        qv (Distribution): the predicted distribution of the values.
        lambda_values (Tensor): the lambda values computed from the imagined states.
        discount (Tensor): the discount to apply to the loss.

    Returns:
        The tensor of the critic loss.
    """
    return -torch.mean(discount * qv.log_prob(lambda_values))


def actor_loss(lambda_values: Tensor) -> Tensor:
    """
    Compute the actor loss as described in Eq. 7 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    Args:
        lambda_values (Tensor): the lambda values computed on the predictions in the latent space.

    Returns:
        The tensor of the actor loss.
    """
    return -torch.mean(lambda_values)


def reconstruction_loss(
    qo: Distribution,
    observations: Tensor,
    qr: Distribution,
    rewards: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    kl_balancing_alpha: float = 0.8,
    kl_free_nats: float = 0.0,
    kl_free_avg: bool = True,
    kl_regularizer: float = 1.0,
    qc: Optional[Distribution] = None,
    continues: Optional[Tensor] = None,
    continue_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 10 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    Args:
        qo (Distribution): the distribution returned by the observation_model (decoder).
        observations (Tensor): the observations provided by the environment.
        qr (Distribution): the reward distribution returned by the reward_model.
        rewards (Tensor): the rewards obtained by the agent during the "Environment interaction" phase.
        priors_logits (Tensor): the logits of the prior.
        posteriors_logits (Tensor): the logits of the posterior.
        kl_balancing_alpha (float): the kl-balancing alpha value.
            Defaults to 0.8.
        kl_free_nats (float): lower bound of the KL divergence.
            Default to 0.0.
        kl_regularizer (float): scale factor of the KL divergence.
            Default to 1.0.
        qc (Bernoulli, optional): the predicted Bernoulli distribution of the terminal steps.
            0s for the entries that are relative to a terminal step, 1s otherwise.
            Default to None.
        dones (Tensor, optional): 1s for the entries that are relative to a terminal step, 0s otherwise.
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
    observation_loss = -qo.log_prob(observations).mean()
    reward_loss = -qr.log_prob(rewards).mean()
    # KL balancing
    lhs = kl_divergence(
        Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategorical(logits=priors_logits), 1),
    )
    rhs = kl_divergence(
        Independent(OneHotCategorical(logits=posteriors_logits), 1),
        Independent(OneHotCategorical(logits=priors_logits.detach()), 1),
    )
    kl_free_nats = torch.tensor([kl_free_nats], device=lhs.device)
    if kl_free_avg:
        loss_lhs = torch.max(lhs.mean(), kl_free_nats)
        loss_rhs = torch.max(rhs.mean(), kl_free_nats)
    else:
        loss_lhs = torch.maximum(lhs, kl_free_nats).mean()
        loss_rhs = torch.maximum(rhs, kl_free_nats).mean()
    kl_loss = kl_balancing_alpha * loss_lhs + (1 - kl_balancing_alpha) * loss_rhs
    continue_loss = torch.tensor(0, device=device)
    if qc is not None and continues is not None:
        continue_loss = continue_scale_factor * -qc.log_prob(continues).mean()
    reconstruction_loss = kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss
    return reconstruction_loss, kl_loss, reward_loss, observation_loss, continue_loss
