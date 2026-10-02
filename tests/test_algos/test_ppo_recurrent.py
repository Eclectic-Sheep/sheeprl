"""The agent and the rollouts of PPO-recurrent: orthogonal initialization, continuous actions, bootstrapped rewards."""

from math import sqrt
from types import SimpleNamespace
from typing import List

import gymnasium as gym
import numpy as np
import pytest
import torch
from lightning import Fabric
from torch import nn

from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent, RecurrentPPOPlayer
from sheeprl.algos.ppo_recurrent.ppo_recurrent import RecurrentRollout, RecurrentRolloutPlayer
from sheeprl.core import EnvStep
from sheeprl.data.buffers import ReplayBuffer
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


@pytest.mark.parametrize("clip_rewards", [False, True])
def test_rollout_bootstraps_only_the_truncated_episodes(clip_rewards):
    # Three envs end their episode with reward 3: truncated by the time limit, truncated and terminated in the same
    # step, terminated. Only the first one goes on in the MDP: its reward gets gamma times the value (2) of its final
    # observation, added after the clipping of the reward (`env.clip_rewards` was ignored)
    num_envs = 3
    obs = {"state": np.zeros((num_envs, 8), dtype=np.float32)}
    step = EnvStep(
        obs=obs,
        next_obs={"state": np.ones((num_envs, 8), dtype=np.float32)},
        rewards=np.full(num_envs, 3.0),
        terminated=np.array([False, True, True]),
        truncated=np.array([True, True, False]),
        info={"final_obs": np.array([{"state": np.full(8, 5, dtype=np.float32)}] * num_envs, dtype=object)},
    )
    env = SimpleNamespace(num_envs=num_envs, obs=obs, step=lambda actions: step)
    player = player_of(build_agent())
    player.get_values = lambda obs, actions, states: (torch.full((1, len(obs["state"][0]), 1), 2.0), states)
    cfg = dotdict(
        {
            "algo": {
                "cnn_keys": {"encoder": []},
                "mlp_keys": {"encoder": ["state"]},
                "gamma": 0.9,
                "reset_recurrent_state_on_done": True,
            },
            "env": {"clip_rewards": clip_rewards},
            "buffer": {"memmap": False, "validate_args": False},
        }
    )
    rollout = RecurrentRollout(ReplayBuffer(1, num_envs, memmap=False, obs_keys=["state"]))
    with torch.no_grad():
        RecurrentRolloutPlayer(Fabric(accelerator="cpu", devices=1), cfg, player).step(env, rollout)

    reward = np.tanh(3.0) if clip_rewards else 3.0
    np.testing.assert_allclose(rollout.buffer["rewards"][0, :, 0], [reward + 0.9 * 2.0, reward, reward], rtol=1e-6)
    np.testing.assert_array_equal(rollout.buffer["dones"][0, :, 0], [1, 1, 1])
