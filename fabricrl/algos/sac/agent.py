import copy
from typing import Sequence, SupportsFloat, Tuple, Union

import torch
import torch.nn as nn
from lightning.fabric.wrappers import _FabricModule
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from fabricrl.models.models import MLP

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACCritic(nn.Module):
    def __init__(self, observation_dim: int, num_critics: int = 1):
        """The SAC critic. The architecture is the one specified in https://arxiv.org/abs/1812.05905

        Args:
            observation_dim (int): the input dimensions. Can be either an integer
                or a sequence of integers.
            num_critics (int, optional): the number of critic values to output.
                This is useful if one wants to have a single shared backbone that outputs
                `num_critics` critic values.
                Defaults to 1.
        """
        super().__init__()
        self.model = MLP(
            input_dims=observation_dim,
            output_dim=num_critics,
            hidden_sizes=(256, 256),
            activation=nn.ReLU,
            flatten_dim=None,
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Return the Q-value conditioned on the observation and the action

        Args:
            obs (Tensor): input observation
            action (Tensor): input action

        Returns:
            q-value
        """
        x = torch.cat([obs, action], -1)
        return self.model(x)


class SACActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_low: Union[SupportsFloat, NDArray] = -1.0,
        action_high: Union[SupportsFloat, NDArray] = 1.0,
    ):
        """The SAC critic. The architecture is the one specified in https://arxiv.org/abs/1812.05905

        Args:
            observation_dim (int): the input dimensions. Can be either an integer
                or a sequence of integers.
            action_dim (int): the action dimension.
            action_low (Union[SupportsFloat, NDArray], optional): the action lower bound.
                Defaults to -1.0.
            action_high (Union[SupportsFloat, NDArray], optional): the action higher bound.
                Defaults to 1.0.
        """
        super().__init__()
        self.model = MLP(input_dims=observation_dim, output_dim=0, hidden_sizes=(256, 256), flatten_dim=None)
        self.fc_mean = nn.Linear(self.model.output_dim, action_dim)
        self.fc_logstd = nn.Linear(self.model.output_dim, action_dim)

        # Action rescaling buffers
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Given an observation, it returns a tanh-squashed
        sampled action (correctly rescaled to the environment action bounds) and its
        log-prob (as defined in Eq. 26 of https://arxiv.org/abs/1812.05905)

        Args:
            obs (Tensor): the observation tensor

        Returns:
            tanh-squashed action, rescaled to the environment action bounds
            action log-prob
        """
        x = self.model(obs)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        return self.get_actions_and_log_probs(mean, std)

    def get_actions_and_log_probs(self, mean: Tensor, std: Tensor):
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

    def get_greedy_actions(self, obs: Tensor) -> Tensor:
        """Get the action given the input observation greedily

        Args:
            obs (Tensor): input observation

        Returns:
            action
        """
        x = self.model(obs)
        mean = self.fc_mean(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean


class SACAgent:
    def __init__(
        self,
        actor: Union[SACActor, _FabricModule],
        critics: Sequence[Union[SACCritic, _FabricModule]],
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
    def actor(self) -> Union[SACActor, _FabricModule]:
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

    def get_actions_and_log_probs(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        return self.actor(obs)

    def get_greedy_actions(self, obs: Tensor) -> Tensor:
        return self.actor.get_greedy_actions(obs)

    def get_q_values(self, obs: Tensor, action: Tensor) -> Tensor:
        return torch.cat([self.qfs[i](obs, action) for i in range(len(self.qfs))], dim=-1)

    @torch.no_grad()
    def get_target_q_values(self, obs: Tensor, action: Tensor) -> Tensor:
        return torch.cat([self.qfs_target[i](obs, action) for i in range(len(self.qfs))], dim=-1)

    @torch.no_grad()
    def get_next_target_q_values(self, next_obs: Tensor, rewards: Tensor, dones: Tensor, gamma: float):
        # Get q-values for the next observations and actions, estimated by the target q-functions
        next_state_actions, next_state_log_pi = self.get_actions_and_log_probs(next_obs)
        qf_next_target = self.get_target_q_values(next_obs, next_state_actions)
        min_qf_next_target = torch.min(qf_next_target, dim=-1, keepdim=True)[0] - self.alpha * next_state_log_pi
        next_qf_value = rewards + (1 - dones) * gamma * min_qf_next_target
        return next_qf_value

    @torch.no_grad()
    def qfs_target_ema(self) -> None:
        for param, target_param in zip(self.qfs.parameters(), self.qfs_target.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
