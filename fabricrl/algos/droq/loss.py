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
    mean_qf_pi = torch.mean(qf_pi, dim=-1, keepdim=True)

    # Line 10 - Algorithm 2
    actor_loss = ((agent.alpha * log_pi) - mean_qf_pi).mean()
    return actor_loss, log_pi.detach()


def critic_loss(agent: SACAgent, obs: Tensor, actions: Tensor, next_qf_value: Tensor, critic_idx: int) -> Tensor:
    # Line 8 - Algorithm 2
    qf_loss = F.mse_loss(agent.get_ith_q_value(obs, actions, critic_idx), next_qf_value)
    return qf_loss
