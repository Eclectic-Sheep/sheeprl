from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from sheeprl.models.models import MLP


class A2CActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 256, continuous: bool = False):
        """A2C actor.

        Args:
            observation_dim (int): the input dimensions. Can be either an integer
                or a sequence of integers.
            action_dim (int): the action dimension.
            hidden_size (int): the hidden sizes for both of the two-layer MLP.
                Defaults to 256.
            continuous (bool, optional): whether to model continuous actions.
                Defaults to False
        """
        super().__init__()
        self._continuous = continuous
        self.model = MLP(
            input_dims=observation_dim,
            hidden_sizes=(hidden_size, hidden_size),
            activation=nn.Tanh,
            flatten_dim=None,
        )
        if continuous:
            self.fc_mean = nn.Linear(self.model.output_dim, action_dim)
            self.fc_logstd = nn.Linear(self.model.output_dim, action_dim)
        else:
            self.fc_logits = nn.Linear(self.model.output_dim, action_dim)

    def forward(self, obs: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Given an observation, it returns a sampled action (correctly rescaled to the environment action bounds),
        its log-prob and the entropy of the distribution

        Args:
            obs (Tensor): the observation tensor
            action (Tensor, optional): if not None, then the log-probs of this `action`
                are computed, otherwise the actions are sampled from a normal distribution.
                Defaults to None

        Returns:
            action
            action log-prob
            distribution entropy
        """
        x = self.model(obs)
        if self._continuous:
            mean = self.fc_mean(x)
            log_std = self.fc_logstd(x)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
        else:
            logits = self.fc_logits(x)
            dist = torch.distributions.Categorical(logits=logits.unsqueeze(-2))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()

    def get_greedy_actions(self, obs: Tensor) -> Tensor:
        """Get the action given the input observation greedily

        Args:
            obs (Tensor): input observation

        Returns:
            action
        """
        x = self.model(obs)
        if self._continuous:
            action = self.fc_mean(x)
        else:
            logits = self.fc_logits(x)
            action = F.softmax(logits, dim=-1).argmax(-1, keepdim=True)
        return action
