import copy
from math import prod
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(LightningModule):
    def __init__(self, envs: gym.vector.SyncVectorEnv, num_critics: int = 2):
        super().__init__()
        act_space = prod(envs.single_action_space.shape)
        obs_space = prod(envs.single_observation_space.shape)
        self.fc1 = nn.Linear(obs_space + act_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_critics)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([obs, action], 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(LightningModule):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
    ):
        super().__init__()
        act_space = prod(envs.single_action_space.shape)
        obs_space = prod(envs.single_observation_space.shape)
        self.fc1 = nn.Linear(obs_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_space)
        self.fc_logstd = nn.Linear(256, act_space)

        # Action rescaling buffers
        self.register_buffer(
            "action_scale", torch.tensor((envs.action_space.high - envs.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((envs.action_space.high + envs.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()

        # Squash sample
        y_t = torch.tanh(x_t)

        # Action sampled from a Tanh transformed Gaussian distribution
        action = y_t * self.action_scale + self.action_bias

        # Change of variable for probability distributions
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)

        # Log-prob of independent actions is the sum of the log-probs
        log_prob = log_prob.sum(-1, keepdim=True)

        # Squash mean
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SACAgent(LightningModule):
    def __init__(self, envs: gym.vector.SyncVectorEnv, num_critics: int = 2, tau: float = 0.005) -> None:
        super().__init__()

        # Actor and critics
        self._num_critics = num_critics
        self._actor = Actor(envs)
        self._qf = SoftQNetwork(envs, num_critics=num_critics)
        self._qf_target = copy.deepcopy(self.qf)
        for p in self._qf_target.parameters():
            p.requires_grad = False

        # Automatic entropy tuning
        self._target_entropy = -torch.prod(torch.tensor(envs.single_action_space.shape).to(self.device))
        self._log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)
        self._alpha = self._log_alpha.exp().item()

        # EMA tau
        self._tau = tau

    @property
    def num_critics(self) -> int:
        return self._num_critics

    @property
    def qf(self) -> SoftQNetwork:
        return self._qf

    @property
    def actor(self) -> Actor:
        return self._actor

    @property
    def qf_target(self) -> SoftQNetwork:
        return self._qf_target

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def target_entropy(self) -> float:
        return self._target_entropy

    @property
    def log_alpha(self) -> float:
        return self._log_alpha

    def get_action(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.actor.get_action(obs)

    def get_q_values(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.qf.forward(obs, action)

    def qf_target_ema(self) -> None:
        for param, target_param in zip(self.qf.parameters(), self.qf_target.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
