import copy
from math import prod
from typing import Sequence, Tuple, Union

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Critic(nn.Module):
    def __init__(
        self, envs: gym.vector.SyncVectorEnv, num_critics: int = 1, dropout: float = 0.0, layer_norm: bool = False
    ):
        super().__init__()
        act_space = prod(envs.single_action_space.shape)
        obs_space = prod(envs.single_observation_space.shape)
        self.fc1 = nn.Linear(obs_space + act_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_critics)
        if dropout > 0.0:
            self.dr1 = nn.Dropout(dropout)
            self.dr2 = nn.Dropout(dropout)
        else:
            self.dr1 = nn.Identity()
            self.dr2 = nn.Identity()
        if layer_norm:
            self.ln1 = nn.LayerNorm(256)
            self.ln2 = nn.LayerNorm(256)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([obs, action], -1)
        x = F.relu(self.ln1(self.dr1(self.fc1(x))))
        x = F.relu(self.ln2(self.dr2(self.fc2(x))))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        act_space = prod(envs.single_action_space.shape)
        obs_space = prod(envs.single_observation_space.shape)
        self.fc1 = nn.Linear(obs_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_space)
        self.fc_logstd = nn.Linear(256, act_space)

        # Action rescaling buffers
        self.register_buffer(
            "action_scale",
            torch.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward of the Actor: given an observation, it returns the mean and
        the standard deviation of a Normal distribution, to be used with
        `self.get_action_and_log_probs` to sample an action randomly during the
        exploration of the environment.

        Args:
            obs (Tensor): the observation tensor

        Returns:
            mean
            standard deviation
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        return mean, std

    def get_action_and_log_prob(self, mean: Tensor, std: Tensor):
        """Given the mean and the std of a Normal distribution, it returns a tanh-squashed
        sampled action (correctly rescaled to the environment action bounds) and its
        log-prob (as defined in Eq. 26 of https://arxiv.org/abs/1812.05905)

        Args:
            mean (Tensor): the mean of the distribution
            std (Tensor): the standard deviation of the distribution

        Returns:
            tanh-squashed action, rescaled to the environment action bounds
            action log-prob
        """
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()

        # Squash sample
        y_t = torch.tanh(x_t)

        # Action sampled from a Tanh transformed Gaussian distribution
        action = y_t * self.action_scale + self.action_bias

        # Change of variable for probability distributions
        # Eq. 26 of https://arxiv.org/abs/1812.05905
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)

        # Log-prob of independent actions is the sum of the log-probs
        log_prob = log_prob.sum(-1, keepdim=True)

        # Squash mean
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob

    def get_greedy_action(self, obs: Tensor) -> Tensor:
        """Get the action given the input observation greedily

        Args:
            obs (Tensor): input observation

        Returns:
            action
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean


class SACAgent:
    def __init__(
        self,
        actor: Union[Actor, _FabricModule],
        critics: Sequence[Union[Critic, _FabricModule]],
        target_entropy: float,
        alpha: float = 1.0,
        tau: float = 0.005,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        # Actor and critics
        self._num_critics = len(critics)
        self._actor = actor
        self._qfs = nn.ModuleList(critics)
        qfs_target = []
        for critic in critics:
            if isinstance(critic, (DistributedDataParallel, _FabricModule)):
                qfs_target.append(copy.deepcopy(critic.module))
            elif isinstance(critic, nn.Module):
                qfs_target.append(copy.deepcopy(critic))
            else:
                raise ValueError("Every critic must be a subclass of `torch.nn.Module`")
        self._qfs_target = nn.ModuleList(qfs_target)
        for p in self._qfs_target.parameters():
            p.requires_grad = False

        # Automatic entropy tuning
        self._target_entropy = torch.tensor(target_entropy, device=device)
        self._log_alpha = torch.nn.Parameter(torch.log(torch.tensor([alpha], device=device)), requires_grad=True)

        # EMA tau
        self._tau = tau

    @property
    def num_critics(self) -> int:
        return self._num_critics

    @property
    def qfs(self) -> nn.ModuleList:
        return self._qfs

    @property
    def actor(self) -> Union[Actor, _FabricModule]:
        return self._actor

    @property
    def qfs_target(self) -> nn.ModuleList:
        return self._qfs_target

    @property
    def alpha(self) -> float:
        return self._log_alpha.exp().item()

    @property
    def target_entropy(self) -> Tensor:
        return self._target_entropy

    @property
    def log_alpha(self) -> Tensor:
        return self._log_alpha

    def get_action_and_log_prob(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        mean, std = self.actor(obs)
        return self.actor.get_action_and_log_prob(mean, std)

    def get_greedy_action(self, obs: Tensor) -> Tensor:
        return self.actor.get_greedy_action(obs)

    def get_ith_q_value(self, obs: Tensor, action: Tensor, critic_idx: int) -> Tensor:
        return self.qfs[critic_idx](obs, action)

    def get_q_values(self, obs: Tensor, action: Tensor) -> Tensor:
        return torch.cat([self.get_ith_q_value(obs, action, critic_idx=i) for i in range(len(self.qfs))], dim=-1)

    @torch.no_grad()
    def get_ith_target_q_value(self, obs: Tensor, action: Tensor, critic_idx: int) -> Tensor:
        return self.qfs_target[critic_idx](obs, action)

    @torch.no_grad()
    def get_target_q_values(self, obs: Tensor, action: Tensor) -> Tensor:
        return torch.cat([self.get_ith_target_q_value(obs, action, critic_idx=i) for i in range(len(self.qfs))], dim=-1)

    @torch.no_grad()
    def get_next_target_q_value(self, next_obs: Tensor, rewards: Tensor, dones: Tensor, gamma: float):
        # Get q-values for the next observations and actions, estimated by the target q-functions
        next_state_actions, next_state_log_pi = self.get_action_and_log_prob(next_obs)
        qf_next_target = self.get_target_q_values(next_obs, next_state_actions)
        min_qf_next_target = torch.min(qf_next_target, dim=-1, keepdim=True)[0] - self.alpha * next_state_log_pi
        next_qf_value = rewards + (1 - dones) * gamma * min_qf_next_target
        return next_qf_value

    @torch.no_grad()
    def qfs_target_ema(self) -> None:
        for param, target_param in zip(self.qfs.parameters(), self.qfs_target.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
