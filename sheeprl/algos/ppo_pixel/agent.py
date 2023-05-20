from abc import ABC
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal
from abc import ABC, abstractmethod

from sheeprl.models.models import MLP, NatureCNN


class PPOPixelAgent(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def get_value(self, *args, **kwargs) -> Tensor:
        ...

    @abstractmethod
    def get_greedy_actions(self, *args, **kwargs) -> Tensor:
        ...


# Simple wrapper to let torch.distributed.algorithms.join.Join
# correctly injects fake communication hooks when we are
# working with uneven inputs
class PPOPixelContinuousAgent(PPOPixelAgent):
    def __init__(self, in_channels: int, features_dim: int, action_dim: int, screen_size: int = 64):
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
        normal = Normal(mean, std)
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


class PPOAtariAgent(PPOPixelAgent):
    def __init__(self, in_channels: int, features_dim: int, action_dim: int, screen_size: int = 64):
        super().__init__()
        self.feature_extractor = NatureCNN(in_channels=in_channels, features_dim=features_dim, screen_size=screen_size)
        self.critic = MLP(input_dims=features_dim, output_dim=1, hidden_sizes=(), activation=torch.nn.ReLU)
        self.actor = MLP(input_dims=features_dim, output_dim=action_dim, hidden_sizes=(), activation=torch.nn.ReLU)

    def forward(self, obs: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feat = self.feature_extractor(obs)
        actions_logits = self.actor(feat)
        categorical = Categorical(logits=actions_logits.unsqueeze(-2))
        if action is None:
            action = categorical.sample()
        log_prob = categorical.log_prob(action)
        return action, log_prob, categorical.entropy(), self.critic(feat)

    def get_value(self, obs: Tensor) -> Tensor:
        feat = self.feature_extractor(obs)
        return self.critic(feat)

    def get_greedy_actions(self, obs: Tensor) -> Tensor:
        feat = self.feature_extractor(obs)
        action_logits = self.actor(feat)
        return F.softmax(action_logits, dim=-1).argmax(dim=-1, keepdim=True)
