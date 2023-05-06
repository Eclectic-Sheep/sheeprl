import copy
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from fabricrl.algos.sac.agent import SACActor
from fabricrl.models.models import MLP

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class DROQCritic(nn.Module):
    def __init__(self, observation_dim: int, num_critics: int = 1, dropout: float = 0.0):
        """The DroQ critic with Dropout and LayerNorm layers. The architecture is the one specified in
        https://arxiv.org/abs/2110.02034

        Args:
            observation_dim (int): the input dimension.
            num_critics (int, optional): the number of critic values to output.
                This is useful if one wants to have a single shared backbone that outputs
                `num_critics` critic values.
                Defaults to 1.
            dropout (float, optional): the dropout probability for every layer.
                Defaults to 0.0.
        """
        super().__init__()
        self.model = MLP(
            observation_dim=observation_dim,
            output_dim=num_critics,
            hidden_sizes=(256, 256),
            dropout_layer=nn.Dropout if dropout > 0 else None,
            dropout_args={"p": dropout} if dropout > 0 else None,
            norm_layer=nn.LayerNorm,
            activation=nn.ReLU,
            flatten_input=False,
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


class DROQAgent:
    def __init__(
        self,
        actor: Union[SACActor, _FabricModule],
        critics: Sequence[Union[DROQCritic, _FabricModule]],
        target_entropy: float,
        alpha: float = 1.0,
        tau: float = 0.005,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """DroQ agent with some helper functions.

        Args:
            actor (Union[SACActor, _FabricModule]): the actor.
            critics (Sequence[Union[DROQCritic, _FabricModule]]): a sequence of critics.
            target_entropy (float): the target entropy to learn the alpha parameter.
            alpha (float, optional): initial alpha value. The parameter learned is the logarithm
                of the alpha value.
                Defaults to 1.0.
            tau (float, optional): the tau value for the exponential moving average of the critics.
                Defaults to 0.005.
            device (torch.device, optional): defaults to torch.device("cpu").
        """
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

    def get_greedy_action(self, obs: Tensor) -> Tensor:
        return self.actor.get_greedy_actions(obs)

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
