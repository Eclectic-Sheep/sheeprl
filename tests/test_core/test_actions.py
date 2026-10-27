"""The translation between the actions of the agents and of the environments (`ActionCodec`)."""

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sheeprl.core.actions import ActionCodec, ActionHead


@pytest.mark.parametrize(
    "space,heads",
    [
        (gym.spaces.Discrete(4), [ActionHead("categorical", 4)]),
        (gym.spaces.MultiDiscrete([2, 3]), [ActionHead("categorical", 2), ActionHead("categorical", 3)]),
        (gym.spaces.MultiBinary(2), [ActionHead("categorical", 2)] * 2),
        (gym.spaces.Box(-1, 1, (2, 3)), [ActionHead("continuous", 6)]),
        (
            gym.spaces.Tuple([gym.spaces.Discrete(2), gym.spaces.Box(-1, 1, (3,))]),
            [ActionHead("categorical", 2), ActionHead("continuous", 3)],
        ),
    ],
)
def test_the_heads_of_the_action_spaces(space, heads):
    codec = ActionCodec.from_space(space)
    assert list(codec.heads) == heads and codec.dims == tuple(h.size for h in heads)


def test_categorical_actions_are_one_hot_for_the_agent_and_indices_for_the_environments():
    codec = ActionCodec.from_space(gym.spaces.MultiDiscrete([2, 3]))
    assert codec.categorical and not codec.continuous
    actions = [F.one_hot(torch.tensor([1, 0]), 2).float(), F.one_hot(torch.tensor([2, 1]), 3).float()]
    assert np.array_equal(codec.to_env(actions), [[1, 2], [0, 1]])
    stored = codec.to_store(actions)
    assert stored.shape == (2, 5)
    assert all(torch.equal(a, b) for a, b in zip(codec.split(torch.as_tensor(stored)), actions))
    # The random actions of the environments, one-hot as the agent's
    assert np.array_equal(codec.from_env(np.array([[1, 2], [0, 1]])), stored)


def test_continuous_actions_are_the_same_for_the_agent_and_the_environments():
    codec = ActionCodec.from_space(gym.spaces.Box(-1, 1, (3,)))
    actions = [torch.tensor([[0.1, -0.2, 0.3]])]
    assert np.array_equal(codec.to_env(actions), actions[0].numpy())
    assert np.array_equal(codec.to_store(actions), actions[0].numpy())
    assert np.array_equal(codec.from_env(np.array([[0.5, 0.5, 0.5]])), [[0.5, 0.5, 0.5]])


def test_an_actor_rejects_a_mixed_action_space_by_name():
    codec = ActionCodec.from_space(gym.spaces.Tuple([gym.spaces.Discrete(2), gym.spaces.Box(-1, 1, (3,))]))
    with pytest.raises(NotImplementedError, match="The actor of PPO"):
        codec.check("The actor of PPO")
