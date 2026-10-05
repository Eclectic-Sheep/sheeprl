from math import sqrt
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from sheeprl.algos.ppo.agent import PPOAgent, PPOPlayer
from sheeprl.utils.utils import dotdict


def build_agent(
    ortho_init: bool, encoder_layers: int = 0, is_continuous: bool = False, distribution: str = "auto"
) -> PPOAgent:
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
        distribution_cfg=dotdict({"type": distribution}),
        is_continuous=is_continuous,
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


def tanh_normal_player(mean: float, std: float) -> Tuple[PPOPlayer, PPOAgent]:
    """A tanh-normal policy whose normal, before the tanh, has mean `mean` and standard deviation `std` everywhere."""
    agent = build_agent(ortho_init=False, is_continuous=True, distribution="tanh_normal")
    head = agent.actor.actor_heads[0]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([mean] * 4 + [np.log(std)] * 4))
    return PPOPlayer(agent.feature_extractor, agent.actor, agent.critic), agent


def test_ppo_tanh_normal_greedy_actions_are_squashed():
    # The greedy action is the tanh of the mean (it was its atanh, out of the action bounds)
    player, _ = tanh_normal_player(mean=2.0, std=0.5)
    (actions,) = player.get_actions({"state": torch.zeros(3, 8)}, greedy=True)
    torch.testing.assert_close(actions, torch.full((3, 4), np.tanh(2.0), dtype=torch.float32))


@pytest.mark.parametrize("mean", [0.5, 8.0])
def test_ppo_tanh_normal_log_probs_of_the_played_actions(mean):
    torch.manual_seed(1)
    player, agent = tanh_normal_player(mean=mean, std=0.5)
    obs = {"state": torch.zeros(1000, 8)}
    actions, logprobs, _ = player(obs)
    # The environments play the tanh of the stored actions
    torch.testing.assert_close(player.env_actions(actions).squeeze(-1), actions[0].tanh().clamp(-1 + 1e-6, 1 - 1e-6))
    # The agent computes the log-probabilities the player computed, also where the tanh saturates (mean 8: the stored
    # tanh lost the action, and the ratio of PPO started far from 1)
    _, new_logprobs, _, _ = agent(obs, actions)
    torch.testing.assert_close(new_logprobs, logprobs)
    # They are the log-probabilities of the tanh-normal distribution (in float64, where the tanh doesn't saturate)
    if mean < 1:
        u = actions[0].double()
        dist = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(torch.full_like(u, mean), torch.full_like(u, 0.5)),
            [torch.distributions.TanhTransform()],
        )
        torch.testing.assert_close(logprobs.double(), dist.log_prob(u.tanh()).sum(-1, keepdim=True), atol=1e-4, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The player is replayed by CUDA graphs on the GPU")
@pytest.mark.parametrize("is_continuous", [False, True])
def test_the_compiled_player_plays_as_the_eager_one(monkeypatch, is_continuous):
    # The same weights, observations and random numbers: the same actions, log-probabilities and values, with and
    # without `torch.compile` and its CUDA graphs, whose outputs are copies (the next replay overwrites them)
    from lightning import Fabric

    from sheeprl.utils.compile import compiled_player

    from .compiled import same_random_numbers

    same_random_numbers(monkeypatch)
    agent = build_agent(ortho_init=False, is_continuous=is_continuous).cuda()
    cfg = dotdict(
        {"algo": {"compile": {"enabled": True, "mode": "reduce-overhead"}}, "fabric": {"precision": "32-true"}}
    )
    eager = PPOPlayer(agent.feature_extractor, agent.actor, agent.critic)
    compiled = PPOPlayer(agent.feature_extractor, agent.actor, agent.critic)
    compiled.forward = compiled_player(compiled.forward, Fabric(accelerator="cuda", devices=1), cfg)
    obs = {"state": torch.randn(4, 8, device="cuda")}
    played = []
    for module in (eager, compiled):
        torch.manual_seed(1)
        with torch.inference_mode():
            played.append([module(obs) for _ in range(3)])
    for (eager_actions, eager_logprobs, eager_values), (actions, logprobs, values) in zip(*played):
        for a, e in zip(actions, eager_actions):
            torch.testing.assert_close(a, e)
        torch.testing.assert_close(logprobs, eager_logprobs, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(values, eager_values, rtol=1e-4, atol=1e-5)
