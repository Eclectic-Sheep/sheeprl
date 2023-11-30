import copy
from math import prod
from typing import Any, Dict, Optional, Sequence, SupportsFloat, Tuple, Union

import gymnasium
import torch
import torch.nn as nn
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from numpy.typing import NDArray
from torch import Tensor

from sheeprl.models.models import MLP

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_size: int = 256, num_critics: int = 1):
        """The SAC critic. The architecture is the one specified in https://arxiv.org/abs/1812.05905

        Args:
            observation_dim (int): the input dimensions. Can be either an integer
                or a sequence of integers.
            hidden_size (int): the hidden sizes for both of the two-layer MLP.
                Defaults to 256.
            num_critics (int, optional): the number of critic values to output.
                This is useful if one wants to have a single shared backbone that outputs
                `num_critics` critic values.
                Defaults to 1.
        """
        super().__init__()
        self.model = MLP(
            input_dims=observation_dim,
            output_dim=num_critics,
            hidden_sizes=(hidden_size, hidden_size),
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
        distribution_cfg: Dict[str, Any],
        hidden_size: int = 256,
        action_low: Union[SupportsFloat, NDArray] = -1.0,
        action_high: Union[SupportsFloat, NDArray] = 1.0,
    ):
        """The SAC critic. The architecture is the one specified in https://arxiv.org/abs/1812.05905

        Args:
            observation_dim (int): the input dimensions. Can be either an integer
                or a sequence of integers.
            action_dim (int): the action dimension.
            distribution_cfg (Dict[str, Any]): the configs of the distributions.
            hidden_size (int): the hidden sizes for both of the two-layer MLP.
                Defaults to 256.
            action_low (Union[SupportsFloat, NDArray], optional): the action lower bound.
                Defaults to -1.0.
            action_high (Union[SupportsFloat, NDArray], optional): the action higher bound.
                Defaults to 1.0.
        """
        super().__init__()
        self.distribution_cfg = distribution_cfg

        self.model = MLP(input_dims=observation_dim, hidden_sizes=(hidden_size, hidden_size), flatten_dim=None)
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
        normal = torch.distributions.Normal(mean, std, validate_args=self.distribution_cfg.validate_args)

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


class SACAgent(nn.Module):
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
        self.actor = actor
        self.critics = critics

        # Automatic entropy tuning
        self._target_entropy = torch.tensor(target_entropy, device=device)
        self._log_alpha = torch.nn.Parameter(torch.log(torch.tensor([alpha], device=device)), requires_grad=True)

        # EMA tau
        self._tau = tau

    def __setattr__(self, name: str, value: Union[Tensor, nn.Module]) -> None:
        # Taken from https://github.com/pytorch/pytorch/pull/92044
        # Check if a property setter exists. If it does, use it.
        class_attr = getattr(self.__class__, name, None)
        if isinstance(class_attr, property) and class_attr.fset is not None:
            return class_attr.fset(self, value)
        super().__setattr__(name, value)

    @property
    def critics(self) -> nn.ModuleList:
        return self.qfs

    @critics.setter
    def critics(self, critics: Sequence[Union[SACCritic, _FabricModule]]) -> None:
        self._qfs = nn.ModuleList(critics)

        # Create target critic unwrapping the DDP module from the critics to prevent
        # `RuntimeError: DDP Pickling/Unpickling are only supported when using DDP with the default process group.
        # That is, when you have called init_process_group and have not passed process_group
        # argument to DDP constructor`.
        # This happens when we're using the decoupled version of SAC for example
        qfs_unwrapped_modules = []
        for critic in critics:
            if hasattr(critic, "module"):
                critic_module = critic.module
            else:
                critic_module = critic
            qfs_unwrapped_modules.append(critic_module)
        self._qfs_unwrapped = nn.ModuleList(qfs_unwrapped_modules)
        self._qfs_target = copy.deepcopy(self._qfs_unwrapped)
        for p in self._qfs_target.parameters():
            p.requires_grad = False
        return

    @property
    def num_critics(self) -> int:
        return self._num_critics

    @property
    def qfs(self) -> nn.ModuleList:
        return self._qfs

    @property
    def qfs_unwrapped(self) -> nn.ModuleList:
        return self._qfs_unwrapped

    @property
    def actor(self) -> Union[SACActor, _FabricModule]:
        return self._actor

    @actor.setter
    def actor(self, actor: Union[SACActor, _FabricModule]) -> None:
        self._actor = actor
        return

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
        for param, target_param in zip(self.qfs_unwrapped.parameters(), self.qfs_target.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)


def build_agent(
    fabric: Fabric,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    action_space: gymnasium.spaces.Box,
    agent_state: Optional[Dict[str, Tensor]] = None,
) -> SACAgent:
    act_dim = prod(action_space.shape)
    obs_dim = sum([prod(obs_space[k].shape) for k in cfg.algo.mlp_keys.encoder])
    actor = SACActor(
        observation_dim=obs_dim,
        action_dim=act_dim,
        distribution_cfg=cfg.distribution,
        hidden_size=cfg.algo.actor.hidden_size,
        action_low=action_space.low,
        action_high=action_space.high,
    )
    critics = [
        SACCritic(observation_dim=obs_dim + act_dim, hidden_size=cfg.algo.critic.hidden_size, num_critics=1)
        for _ in range(cfg.algo.critic.n)
    ]
    target_entropy = -act_dim
    agent = SACAgent(actor, critics, target_entropy, alpha=cfg.algo.alpha.alpha, tau=cfg.algo.tau, device=fabric.device)
    if agent_state:
        agent.load_state_dict(agent_state)
    agent.actor = fabric.setup_module(agent.actor)
    agent.critics = [fabric.setup_module(critic) for critic in agent.critics]

    return agent
