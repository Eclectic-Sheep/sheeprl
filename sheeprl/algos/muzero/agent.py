from typing import Tuple

import torch

from sheeprl.models.models import MLP, NatureCNN


class MuzeroAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.representation: torch.nn.Module
        self.prediction: torch.nn.Module
        self.dynamics: torch.nn.Module

    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(
        self, action: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reward, next_hidden_state = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value

    def forward(self, action, hidden_state):
        return self.recurrent_inference(action, hidden_state)


class GruMlp(torch.nn.Module):
    def __init__(self, hidden_state_size=256):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=hidden_state_size + 1, hidden_size=hidden_state_size)
        self.mlp = MLP(input_size=hidden_state_size, output_size=1)

    def forward(self, x, h0):
        y, h1 = self.gru(x, h0)
        y = self.mlp(y)
        return y, h1


class RecurrentMuzero(MuzeroAgent):
    def __init__(self, hidden_state_size=256, num_actions=4):
        super().__init__()
        self.representation = NatureCNN(in_channels=3, features_dim=hidden_state_size)
        self.dynamics: torch.nn.Module = GruMlp(hidden_state_size=hidden_state_size)
        self.prediction: torch.nn.Module = MLP(input_size=hidden_state_size, output_size=num_actions + 1)
