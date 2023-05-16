from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sheeprl.models.models import MLP


class PPOContinuousActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
    ):
        """PPO continuous actor.

        Args:
            observation_dim (int): the input dimensions. Can be either an integer
                or a sequence of integers.
            action_dim (int): the action dimension.
        """
        super().__init__()
        self.model = MLP(input_dims=observation_dim, output_dim=0, hidden_sizes=(64, 64), flatten_dim=None)
        self.fc_mean = nn.Linear(self.model.output_dim, action_dim)
        self.fc_logstd = nn.Linear(self.model.output_dim, action_dim)

    def forward(self, obs: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Given an observation, it returns a sampled action (correctly rescaled to the environment action bounds),
        its log-prob and the entropy of the distribution

        Args:
            obs (Tensor): the observation tensor
            action (Tensor, optional): if not None, then the log-probs of this `action`
                are computed, otherwise the actions are sampled from a normal distribution.
                Defaults to None

        Returns:
            action, rescaled to the environment action bounds
            action log-prob
            distribution entropy
        """
        x = self.model(obs)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if action is None:
            action = normal.sample()
        log_prob = normal.log_prob(action)
        return action, log_prob, normal.entropy()

    def get_greedy_actions(self, obs: Tensor) -> Tensor:
        """Get the action given the input observation greedily

        Args:
            obs (Tensor): input observation

        Returns:
            action
        """
        x = self.model(obs)
        mean = self.fc_mean(x)
        return mean
