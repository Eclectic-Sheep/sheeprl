"""Based on "Soft Actor-Critic Algorithms and Applications": https://arxiv.org/abs/1812.05905
"""

from numbers import Number

import torch.nn.functional as F
from torch import Tensor


def policy_loss(alpha: Number, logprobs: Tensor, qf_values: Tensor) -> Tensor:
    # Eq. 7
    return ((alpha * logprobs) - qf_values).mean()


def critic_loss(qf_values: Tensor, next_qf_value: Tensor, num_critics: int) -> Tensor:
    # Eq. 5
    qf_loss = sum(
        F.mse_loss(qf_values[..., qf_value_idx].unsqueeze(-1), next_qf_value) for qf_value_idx in range(num_critics)
    )
    return qf_loss


def entropy_loss(log_alpha: Tensor, logprobs: Tensor, target_entropy: Tensor) -> Tensor:
    # Eq. 17
    alpha_loss = (-log_alpha * (logprobs + target_entropy)).mean()
    return alpha_loss
