import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, state_size, actions_dim, out_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            state_size + actions_dim,
            state_size,
        )
        self.fc2 = nn.Linear(
            state_size + actions_dim,
            state_size,
        )
        self.fc3 = nn.Linear(state_size + actions_dim, out_dim)
        self.activation_fn = nn.ReLU()

    def forward(self, state, action):
        x = self.activation_fn(self.fc1(torch.cat((state, action), -1)))
        x = self.activation_fn(self.fc2(torch.cat((x, action), -1)))
        return self.fc3(torch.cat((x, action), -1))
