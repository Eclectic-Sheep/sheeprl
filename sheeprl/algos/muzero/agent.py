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
        reward, next_hidden_state = self.dynamics(action, hidden_state)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value

    def forward(self, action, hidden_state):
        return self.recurrent_inference(action, hidden_state)


class GruMlpDynamics(torch.nn.Module):
    def __init__(self, hidden_state_size=256):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=hidden_state_size)
        self.mlp = MLP(input_dims=hidden_state_size, output_dim=1)

    def forward(self, x, h0):
        y, h1 = self.gru(x, h0)
        y = self.mlp(y)
        return y, h1


class Predictor(torch.nn.Module):
    def __init__(self, hidden_state_size=256, num_actions=4):
        super().__init__()
        self.mlp1 = MLP(input_dims=hidden_state_size, output_dim=num_actions)
        self.mlp2 = MLP(input_dims=hidden_state_size, output_dim=1)

    def forward(self, x):
        return self.mlp1(x), self.mlp2(x)


class RecurrentMuzero(MuzeroAgent):
    def __init__(self, hidden_state_size=256, num_actions=4):
        super().__init__()
        self.representation = NatureCNN(in_channels=3, features_dim=hidden_state_size)
        self.dynamics: torch.nn.Module = GruMlpDynamics(hidden_state_size=hidden_state_size)
        self.prediction: torch.nn.Module = Predictor(hidden_state_size=hidden_state_size, num_actions=num_actions)


if __name__ == "__main__":
    agent = RecurrentMuzero()
    observation = torch.rand(1, 3, 64, 64)
    hidden_state, policy_logits, value = agent.initial_inference(observation)
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(value.shape)
    action = torch.randint(0, 4, (1, 1)).to(torch.float32)
    next_hidden_state, reward, policy_logits, value = agent.recurrent_inference(action, hidden_state)
    print(next_hidden_state.shape)
    print(reward.shape)
    print(policy_logits.shape)
    print(value.shape)
    action2 = torch.randint(0, 4, (1, 1)).to(torch.float32)
    last_hidden_state, reward, policy_logits, value = agent.recurrent_inference(action2, next_hidden_state)
    print(last_hidden_state.shape)
    print(reward.shape)
    print(policy_logits.shape)
    print(value.shape)
