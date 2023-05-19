from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sheeprl.models.models import MLP, NatureCNN


# Simple wrapper to let torch.distributed.algorithms.join.Join
# correctly injects fake communication hooks when we are
# working with uneven inputs
class PPOPixelContinuousAgent(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features_dim: int,
        action_dim: int,
        screen_size: int = 64,
    ):
        super().__init__()
        self.feature_extractor = NatureCNN(in_channels=in_channels, features_dim=features_dim, screen_size=screen_size)
        self.critic = MLP(input_dims=features_dim, output_dim=1, hidden_sizes=(), activation=torch.nn.ReLU)
        self.fc_mean = nn.Linear(features_dim, action_dim)
        self.fc_logstd = nn.Linear(features_dim, action_dim)

    def forward(self, obs: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feat = self.feature_extractor(obs)
        mean = self.fc_mean(feat)
        log_std = self.fc_logstd(feat)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if action is None:
            action = normal.sample()
        log_prob = normal.log_prob(action)
        return action, log_prob, normal.entropy(), self.critic(feat)

    def get_value(self, obs: Tensor) -> Tensor:
        feat = self.feature_extractor(obs)
        return self.critic(feat)

    def get_greedy_actions(self, obs: Tensor) -> Tensor:
        feat = self.feature_extractor(obs)
        mean = self.fc_mean(feat)
        return mean
