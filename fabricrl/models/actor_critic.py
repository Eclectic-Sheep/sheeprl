import math
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.distributions import Categorical
from tensordict import TensorDict
from torchmetrics import MeanMetric


class ActorCritic(torch.nn.Module):
    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x: TensorDict) -> TensorDict:
        action_logits = self.actor(x)
        critic_values = self.critic(x)