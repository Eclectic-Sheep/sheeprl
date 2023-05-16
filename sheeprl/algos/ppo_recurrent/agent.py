from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sheeprl.models.models import MLP


class RecurrentPPOAgent(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, lstm_hidden_size: int = 64, num_envs: int = 1):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_envs = num_envs

        # Actor: Obs -> Feature -> LSTM -> Logits
        self._actor_fc = MLP(
            input_dims=observation_dim,
            output_dim=0,
            hidden_sizes=(),
            activation=nn.ReLU,
            flatten_dim=2,
        )
        self._actor_rnn = nn.LSTM(input_size=self._actor_fc.output_dim, hidden_size=lstm_hidden_size, batch_first=False)
        self._actor_logits = MLP(
            lstm_hidden_size, action_dim, (lstm_hidden_size * 2, lstm_hidden_size * 2), flatten_dim=None
        )

        # Critic: Obs -> Feature -> LSTM -> Values
        self._critic_fc = MLP(
            input_dims=observation_dim,
            output_dim=0,
            hidden_sizes=(),
            activation=nn.ReLU,
            flatten_dim=2,
        )
        self._critic_rnn = nn.LSTM(
            input_size=self._critic_fc.output_dim, hidden_size=lstm_hidden_size, batch_first=False
        )
        self._critic = MLP(lstm_hidden_size, 1, (lstm_hidden_size * 2, lstm_hidden_size * 2), flatten_dim=None)

        # Initial recurrent states for both the actor and critic rnn
        self._initial_states: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]] = self.reset_hidden_states()

    @property
    def initial_states(self) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        return self._initial_states

    @initial_states.setter
    def initial_states(self, value: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]) -> None:
        self._initial_states = value

    def reset_hidden_states(self) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        actor_state = (
            torch.zeros(1, self.num_envs, self.lstm_hidden_size),
            torch.zeros(1, self.num_envs, self.lstm_hidden_size),
        )
        critic_state = (
            torch.zeros(1, self.num_envs, self.lstm_hidden_size),
            torch.zeros(1, self.num_envs, self.lstm_hidden_size),
        )
        return (actor_state, critic_state)

    def get_greedy_action(self, obs: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x = self._actor_fc(obs)
        x, state = self._actor_rnn(x, state)
        logits = self._actor_logits(x)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1), state

    def get_logits(
        self, obs: Tensor, actor_state: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x_actor = self._actor_fc(obs)
        self._actor_rnn.flatten_parameters()
        if mask is not None:
            lengths = mask.sum(dim=0).view(-1).cpu()
            x_actor = torch.nn.utils.rnn.pack_padded_sequence(
                x_actor, lengths=lengths, batch_first=False, enforce_sorted=False
            )
        actor_hidden, actor_state = self._actor_rnn(x_actor, actor_state)
        if mask is not None:
            actor_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(
                actor_hidden, batch_first=False, total_length=obs.shape[0]
            )
        logits = self._actor_logits(actor_hidden)
        return logits, actor_state

    def get_values(
        self, obs: Tensor, critic_state: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x_critic = self._critic_fc(obs)
        self._critic_rnn.flatten_parameters()
        if mask is not None:
            lengths = mask.sum(dim=0).view(-1).cpu()
            x_critic = torch.nn.utils.rnn.pack_padded_sequence(
                x_critic, lengths=lengths, batch_first=False, enforce_sorted=False
            )
        critic_hidden, critic_state = self._critic_rnn(x_critic, critic_state)
        if mask is not None:
            critic_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(
                critic_hidden, batch_first=False, total_length=obs.shape[0]
            )
        values = self._critic(critic_hidden)
        return values, critic_state

    def forward(
        self,
        obs: Tensor,
        state: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]] = ((None, None), (None, None)),
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
        """Compute actor logits and critic values.

        Args:
            obs (Tensor): observations collected (possibly padded with zeros).
            state (Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]], optional): the recurrent states.
                Defaults to None.
            mask (Tensor, optional): boolean mask with valid indices. This is used to pack the padded
                sequences during training. If None, no packing will be done.
                Defaults to None.

        Returns:
            actor logits
            critic values
            next recurrent state for both the actor and the critic
        """
        device = obs.device
        actor_state, critic_state = state
        logits, actor_state = self.get_logits(obs, tuple([s.to(device) for s in actor_state]), mask)
        values, critic_state = self.get_values(obs, tuple([s.to(device) for s in critic_state]), mask)
        return logits, values, (actor_state, critic_state)
