"""Based on "Soft Actor-Critic Algorithms and Applications": https://arxiv.org/abs/1812.05905
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from fabricrl.algos.sac.agent import SACAgent


def policy_loss(agent: SACAgent, obs: Tensor) -> Tuple[Tensor, Tensor]:
    pi, log_pi, _ = agent.actor.get_action(obs)
    qf_pi = agent.qf(obs, pi)
    min_qf_pi = torch.min(qf_pi, dim=-1, keepdim=True)[0]

    # Eq. 7
    actor_loss = ((agent.alpha * log_pi) - min_qf_pi).mean()

    # Update actor metric
    agent.avg_pg_loss(actor_loss)
    return actor_loss, log_pi.detach()


def critic_loss(
    agent: SACAgent,
    obs: Tensor,
    next_obs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    dones: Tensor,
    gamma: float,
) -> Tensor:
    # Get q-values for the next observations and actions, estimated by the target q-functions
    with torch.no_grad():
        next_state_actions, next_state_log_pi, _ = agent.actor.get_action(next_obs)
        qf_next_target = agent.qf_target(next_obs, next_state_actions)
        min_qf_next_target = torch.min(qf_next_target, dim=-1, keepdim=True)[0] - agent.alpha * next_state_log_pi
        next_qf_value = rewards + (~dones) * gamma * min_qf_next_target

    # Get q-values for the current observations and actions
    qf_values = agent.qf(obs, actions)

    # Eq. 5
    qf_loss = (
        1
        / agent.num_critics
        * sum(
            F.mse_loss(qf_values[..., qf_value_idx].unsqueeze(-1), next_qf_value)
            for qf_value_idx in range(agent.num_critics)
        )
    )

    # Update critic metric
    agent.avg_value_loss(qf_loss)
    return qf_loss


def entropy_loss(agent: SACAgent, log_pi: Tensor) -> Tensor:
    # Eq. 17
    alpha_loss = (-agent.log_alpha * (log_pi + agent.target_entropy)).mean()

    # Update entropy metric
    agent.avg_ent_loss(alpha_loss)
    return alpha_loss
