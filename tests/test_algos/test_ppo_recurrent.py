"""The agent of PPO-recurrent: orthogonal initialization, continuous actions, the player."""

from math import sqrt
from typing import List

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent, RecurrentPPOPlayer
from sheeprl.utils.utils import dotdict

HIDDEN_SIZE = 16


def build_agent(ortho_init: bool = False, is_continuous: bool = False) -> RecurrentPPOAgent:
    torch.manual_seed(0)
    networks = {"dense_units": 32, "mlp_layers": 2, "dense_act": "torch.nn.Tanh", "layer_norm": False}
    no_mlp = {"apply": False, "bias": True, "activation": "torch.nn.Tanh", "layer_norm": False, "dense_units": 8}
    return RecurrentPPOAgent(
        actions_dim=[2] if is_continuous else [3],
        obs_space=gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (8,), np.float32)}),
        encoder_cfg=dotdict({**networks, "cnn_features_dim": 64, "mlp_features_dim": 32, "ortho_init": ortho_init}),
        rnn_cfg=dotdict({"lstm": {"hidden_size": HIDDEN_SIZE}, "pre_rnn_mlp": no_mlp, "post_rnn_mlp": no_mlp}),
        actor_cfg=dotdict({**networks, "ortho_init": ortho_init}),
        critic_cfg=dotdict({**networks, "ortho_init": ortho_init}),
        cnn_keys=[],
        mlp_keys=["state"],
        is_continuous=is_continuous,
        distribution_cfg=dotdict({"type": "auto"}),
    )


def player_of(agent: RecurrentPPOAgent) -> RecurrentPPOPlayer:
    return RecurrentPPOPlayer(
        agent.feature_extractor, agent.rnn, agent.actor, agent.critic, HIDDEN_SIZE, agent.actions_dim
    )


def linear_layers(module: nn.Module) -> List[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def assert_orthogonal(layer: nn.Linear, gain: float) -> None:
    singular_values = torch.linalg.svdvals(layer.weight.detach())
    torch.testing.assert_close(singular_values, torch.full_like(singular_values, gain))
    assert torch.all(layer.bias == 0)


def test_ortho_init():
    # The flags inherited from PPO's configuration were not read
    agent = build_agent(ortho_init=True)
    for layer in linear_layers(agent.feature_extractor):
        assert_orthogonal(layer, 1.0)
    *actor_hidden, actor_head = linear_layers(agent.actor)
    *critic_hidden, critic_output = linear_layers(agent.critic)
    for layer in actor_hidden + critic_hidden:
        assert_orthogonal(layer, sqrt(2))
    assert_orthogonal(actor_head, 0.01)
    assert_orthogonal(critic_output, 1.0)
    default = build_agent(ortho_init=False)
    for layer in [layer for layer in linear_layers(default) if min(layer.weight.shape) > 1]:
        singular_values = torch.linalg.svdvals(layer.weight.detach())
        assert not torch.allclose(singular_values, singular_values[0].expand_as(singular_values))


def test_continuous_actions():
    # The log-probabilities of the continuous actions were computed from `None` (the player) or from the tuple of
    # actions (the agent): both crashed
    agent = build_agent(is_continuous=True)
    player = player_of(agent)
    obs = {"state": torch.randn(1, 4, 8)}
    prev_actions = torch.zeros(1, 4, 2)
    states = (torch.zeros(1, 4, HIDDEN_SIZE), torch.zeros(1, 4, HIDDEN_SIZE))
    with torch.no_grad():
        actions, logprobs, _, _ = player(obs, prev_actions=prev_actions, prev_states=states)
        _, agent_logprobs, entropies, _, _ = agent(obs, prev_actions=prev_actions, prev_states=states, actions=actions)
    assert actions[0].shape == (1, 4, 2) and logprobs.shape == (1, 4, 1) and entropies.shape == (1, 4, 1)
    torch.testing.assert_close(agent_logprobs, logprobs)


def config(overrides: List[str]) -> dotdict:
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        return dotdict(OmegaConf.to_container(compose(config_name="config", overrides=overrides), resolve=True))


@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"))],
)
def test_the_player_follows_the_updates_of_the_agent(accelerator):
    # The player was a copy of the agent with the weights tied: on CUDA its LSTM moved them into a new buffer at its
    # first forward (`flatten_parameters`), and the player played with the initial weights for the whole training
    from lightning import Fabric

    from sheeprl.algos.ppo_recurrent.agent import build_agent as build_agents

    cfg = config(
        ["exp=ppo_recurrent", "env.num_envs=2", "algo.mlp_keys.encoder=[state]", "algo.rnn.lstm.hidden_size=8"]
    )
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (8,), np.float32)})
    fabric = Fabric(accelerator=accelerator, devices=1)
    agent, player = build_agents(fabric, [3], False, cfg, obs_space)
    device = fabric.device
    with torch.no_grad():
        player(
            {"state": torch.randn(1, 2, 8, device=device)},
            prev_actions=torch.zeros(1, 2, 3, device=device),
            prev_states=(torch.zeros(1, 2, 8, device=device), torch.zeros(1, 2, 8, device=device)),
        )
        for p in agent.parameters():
            p.add_(1.0)
    for module in ("feature_extractor", "rnn", "actor", "critic"):
        agent_params = list(getattr(agent, module).parameters())
        player_params = list(getattr(player, module).parameters())
        assert len(agent_params) == len(player_params) > 0
        for agent_p, player_p in zip(agent_params, player_params):
            assert torch.equal(agent_p, player_p), module
