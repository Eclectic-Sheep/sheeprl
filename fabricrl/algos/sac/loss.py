"""Based on "Soft Actor-Critic Algorithms and Applications": https://arxiv.org/abs/1812.05905
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from fabricrl.algos.sac.agent import SACAgent


def policy_loss(agent: SACAgent, obs: Tensor) -> Tuple[Tensor, Tensor]:
    pi, log_pi = agent.get_action_and_log_prob(obs)
    qf_pi = agent.get_q_values(obs, pi)
    min_qf_pi = torch.min(qf_pi, dim=-1, keepdim=True)[0]

    # Eq. 7
    actor_loss = ((agent.alpha * log_pi) - min_qf_pi).mean()
    return actor_loss, log_pi.detach()


def critic_loss(agent: SACAgent, obs: Tensor, actions: Tensor, next_qf_value: Tensor) -> Tensor:
    # Get q-values for the current observations and actions
    qf_values = agent.get_q_values(obs, actions)

    # Eq. 5
    qf_loss = sum(
        F.mse_loss(qf_values[..., qf_value_idx].unsqueeze(-1), next_qf_value)
        for qf_value_idx in range(agent.num_critics)
    )
    return qf_loss


def entropy_loss(agent: SACAgent, log_pi: Tensor) -> Tensor:
    # Eq. 17
    alpha_loss = (-agent.log_alpha * (log_pi + agent.target_entropy)).mean()
    return alpha_loss
