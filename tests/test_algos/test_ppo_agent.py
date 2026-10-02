from math import sqrt
from typing import List

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from sheeprl.algos.ppo.agent import PPOAgent
from sheeprl.utils.utils import dotdict


def build_agent(ortho_init: bool, encoder_layers: int = 0) -> PPOAgent:
    torch.manual_seed(0)
    networks = {"dense_units": 64, "mlp_layers": 2, "dense_act": "torch.nn.Tanh", "layer_norm": False}
    return PPOAgent(
        actions_dim=[4],
        obs_space=gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (8,), np.float32)}),
        encoder_cfg=dotdict(
            {
                **networks,
                "mlp_layers": encoder_layers,
                "cnn_features_dim": 512,
                "mlp_features_dim": 64,
                "ortho_init": ortho_init,
            }
        ),
        actor_cfg=dotdict({**networks, "ortho_init": ortho_init}),
        critic_cfg=dotdict({**networks, "ortho_init": ortho_init}),
        cnn_keys=[],
        mlp_keys=["state"],
        screen_size=64,
        distribution_cfg=dotdict({"type": "auto"}),
        is_continuous=False,
    )


def linear_layers(module: nn.Module) -> List[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def assert_orthogonal(layer: nn.Linear, gain: float) -> None:
    # An orthogonal matrix scaled by `gain` has all its singular values equal to `gain`
    singular_values = torch.linalg.svdvals(layer.weight.detach())
    torch.testing.assert_close(singular_values, torch.full_like(singular_values, gain))
    assert torch.all(layer.bias == 0)


def test_ppo_ortho_init_of_actor_and_critic():
    agent = build_agent(ortho_init=True)
    *actor_hidden, actor_head = linear_layers(agent.actor)
    *critic_hidden, critic_output = linear_layers(agent.critic)
    assert len(actor_hidden) == len(critic_hidden) == 2
    for layer in actor_hidden + critic_hidden:
        assert_orthogonal(layer, sqrt(2))
    # An almost uniform initial policy, and the value with the scale of the hidden features
    assert_orthogonal(actor_head, 0.01)
    assert_orthogonal(critic_output, 1.0)


def test_ppo_ortho_init_of_the_encoder():
    agent = build_agent(ortho_init=True, encoder_layers=2)
    for layer in linear_layers(agent.feature_extractor):
        assert_orthogonal(layer, 1.0)


@pytest.mark.parametrize("encoder_layers", [0, 2])
def test_ppo_default_init_is_not_orthogonal(encoder_layers):
    agent = build_agent(ortho_init=False, encoder_layers=encoder_layers)
    # A layer with one output (the critic's) has a single singular value: it says nothing
    for layer in [layer for layer in linear_layers(agent) if min(layer.weight.shape) > 1]:
        singular_values = torch.linalg.svdvals(layer.weight.detach())
        assert not torch.allclose(singular_values, singular_values[0].expand_as(singular_values))
