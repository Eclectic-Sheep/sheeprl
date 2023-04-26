"""Based on "Soft Actor-Critic Algorithms and Applications": https://arxiv.org/abs/1812.05905
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from fabricrl.algos.sac.agent import SACAgent


def policy_loss(agent: SACAgent, obs: Tensor) -> Tuple[Tensor, Tensor]:
    pi, log_pi, _ = agent.get_action(obs)
    qf_pi = agent.get_q_values(obs, pi)
    mean_qf_pi = torch.mean(qf_pi, dim=-1, keepdim=True)

    # Line 10 - Algorithm 2
    actor_loss = ((agent.alpha * log_pi) - mean_qf_pi).mean()

    # Update actor metric
    agent.avg_pg_loss(actor_loss)
    return actor_loss, log_pi.detach()


def critic_loss(agent: SACAgent, obs: Tensor, actions: Tensor, next_qf_value: Tensor, critic_idx: int) -> Tensor:
    # Line 8 - Algorithm 2
    qf_loss = F.mse_loss(agent.get_ith_q_value(obs, actions, critic_idx), next_qf_value)

    # Update critic metric
    agent.avg_value_loss(qf_loss)
    return qf_loss
