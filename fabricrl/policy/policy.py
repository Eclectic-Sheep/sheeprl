from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class CategoricalPolicy(nn.Module):
    """Categorical policy for discrete actions."""

    def __init__(self, feature_dim: int, action_dims: Sequence[int]):
        """Create a policy for multi-discrete actions with a custom number of dimensions.

        Args:
            feature_dim (int): length of the input vector.
            action_dims (sequence of int): dimension for each action.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for dim in action_dims:
            self.layers.append(nn.Linear(feature_dim, dim))

        self.actions_prob = None

    def forward(self, x, greedy):
        """Get the actions according to the policy."""
        self.actions_prob = []
        for layer in self.layers:
            probs = nn.Softmax(dim=1)(layer(x))
            self.actions_prob.append(Categorical(probs=probs))
        if greedy.all():
            actions = [action.mode() for action in self.actions_prob]
        else:
            actions = [action.sample() for action in self.actions_prob]
        return torch.stack(actions, dim=-1)

    def get_logprob(self, actions):
        """Get the log probability of the actions given the current policy."""
        if self.actions_prob is None:
            raise ValueError("No action sampled")
        return torch.stack(
            [
                self.actions_prob[i].log_prob(actions[:, i])
                for i in range(len(self.actions_prob))
            ],
            dim=-1,
        )


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous actions."""

    def __init__(self, feature_dim, num_actions):
        """Create a policy for continuous actions.

        Args:
            feature_dim (int): length of the feature vector.
            num_actions (int): number of actions.
        """
        super().__init__()
        self.num_actions = num_actions
        self.layer = nn.Linear(feature_dim, num_actions * 2)
        self.actions_prob = []

    def forward(self, x):
        """Get the actions according to the policy.

        Samples them from a normal distribution with mean and standard deviation computed during
        the forward pass.
        """
        probs = self.layer(x)
        loc = probs[:, : self.num_actions]
        scale = torch.exp(probs[:, self.num_actions :])
        self.actions_prob.append(Normal(loc=loc, scale=scale))
        return self.actions_prob[0].sample()
