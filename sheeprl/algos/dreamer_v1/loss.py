from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.kl import kl_divergence


def critic_loss(qv: Distribution, lambda_values: Tensor, discount: Tensor) -> Tensor:
    """
    Compute the critic loss as described in Eq. 8 in
    [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    Args:
        qv (Distribution): the predicted distribution of the values.
        lambda_values (Tensor): the lambda values computed from the imagined states.
            Shape of (sequence_length, batch_size, 1).
        discount (Tensor): the discount to apply to the loss.
            Shape of (sequence_length, batch_size), the log_prob removes the last dimension.

    Returns:
        The tensor of the critic loss.
    """
    return -torch.mean(discount * qv.log_prob(lambda_values))


def actor_loss(lambda_values: Tensor) -> Tensor:
    """
    Compute the actor loss as described in Eq. 7 in
    [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    Args:
        lambda_values (Tensor): the lambda values computed on the predictions in the latent space.

    Returns:
        The tensor of the actor loss.
    """
    return -torch.mean(lambda_values)


def reconstruction_loss(
    qo: Distribution,
    observations: Dict[str, Tensor],
    qr: Distribution,
    rewards: Tensor,
    posteriors_dist: Distribution,
    priors_dist: Distribution,
    kl_free_nats: float = 3.0,
    kl_regularizer: float = 1.0,
    qc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 10.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 10 in
    [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    Args:
        qo (Distribution): the distribution returned by the observation_model (decoder).
        observations (Dict[str, Tensor]): the observations provided by the environment.
        qr (Distribution): the reward distribution returned by the reward_model.
        rewards (Tensor): the rewards obtained by the agent during the "Environment interaction" phase.
        posteriors_dist (Distribution): the distribution of the stochastic state.
        priors_dist (Distribution): the predicted distribution of the stochastic state.
        kl_free_nats (float): lower bound of the KL divergence.
            Default to 3.0.
        kl_regularizer (float): scale factor of the KL divergence.
            Default to 1.0.
        qc (Bernoulli, optional): the predicted Bernoulli distribution of the terminal steps.
            0s for the entries that are relative to a terminal step, 1s otherwise.
            Default to None.
        continue_targets (Tensor, optional): 1s for the entries that are relative to a terminal step, 0s otherwise.
            Default to None.
        continue_scale_factor (float): the scale factor for the continue loss.
            Default to 10.

    Returns:
        observation_loss (Tensor): the value of the observation loss.
        kl (Tensor): the value of the kl between p and q.
        reward_loss (Tensor): the value of the reward loss.
        state_loss (Tensor): the value of the state loss.
        continue_loss (Tensor): the value of the continue loss (0 if it is not computed).
        reconstruction_loss (Tensor): the value of the overall reconstruction loss.
    """
    device = rewards.device
    observation_loss = -sum([qo[k].log_prob(observations[k]).mean() for k in qo.keys()])
    reward_loss = -qr.log_prob(rewards).mean()
    kl = kl_divergence(posteriors_dist, priors_dist).mean()
    state_loss = torch.max(torch.tensor(kl_free_nats, device=device), kl)
    continue_loss = torch.tensor(0, device=device)
    if qc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * qc.log_prob(continue_targets)
    reconstruction_loss = kl_regularizer * state_loss + observation_loss + reward_loss + continue_loss
    return reconstruction_loss, kl, state_loss, reward_loss, observation_loss, continue_loss
