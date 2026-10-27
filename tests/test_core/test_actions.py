"""The translation between the actions of the agents and of the environments (`ActionCodec`)."""

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sheeprl.core.actions import ActionCodec, ActionHead, batched_env_actions, single_env_actions


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


def test_mixed_actions_have_the_structure_of_the_action_space():
    space = gym.spaces.Tuple([gym.spaces.Discrete(3, start=1), gym.spaces.Box(-1, 1, (2,), np.float32)])
    codec = ActionCodec.from_space(space)
    assert not codec.categorical and not codec.continuous and codec.continuous_heads == (False, True)
    actions = [F.one_hot(torch.tensor([2, 0]), 3).float(), torch.tensor([[0.1, -0.2], [0.3, 0.4]])]
    env_actions = codec.to_env(actions)
    # The index of the discrete action (from the start of its space), the values of the continuous one
    assert isinstance(env_actions, tuple)
    assert np.array_equal(env_actions[0], [3, 1]) and np.array_equal(env_actions[1], actions[1].numpy())
    stored = codec.to_store(actions)
    assert stored.shape == (2, 5)
    # The actions of the environments (e.g. the random ones) as the store keeps them
    assert np.array_equal(codec.from_env(env_actions), stored)
    # Every environment of the batched space takes its own part of the actions
    batched = gym.vector.utils.batch_space(space, 2)
    assert batched_env_actions(env_actions, batched) is env_actions
    single = single_env_actions(tuple(a[:1] for a in env_actions), space)
    assert space.contains(single) and single[0] == 3


def test_mixed_actions_of_a_dictionary_space_keep_its_keys_and_shapes():
    space = gym.spaces.Dict(
        {"buttons": gym.spaces.MultiDiscrete([2, 3]), "stick": gym.spaces.Box(-1, 1, (2, 2), np.float32)}
    )
    codec = ActionCodec.from_space(space)
    assert codec.dims == (2, 3, 4)
    random_actions = gym.vector.utils.batch_space(space, 4).sample()
    stored = codec.from_env(random_actions)
    assert stored.shape == (4, 9) and stored.dtype == np.float32
    env_actions = codec.to_env(codec.split(torch.as_tensor(stored)))
    assert np.array_equal(env_actions["buttons"], random_actions["buttons"])
    assert np.array_equal(env_actions["stick"], random_actions["stick"])
