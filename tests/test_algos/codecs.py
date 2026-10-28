"""The action codecs of the action spaces of the tests."""

from typing import Sequence

import gymnasium as gym

from sheeprl.core.actions import ActionCodec


def codec_of(actions_dim: Sequence[int], continuous: bool = False) -> ActionCodec:
    """The codec of `Box(-1, 1, (sum(actions_dim),))` if `continuous`, of discrete actions of `actions_dim` choices
    otherwise."""
    actions_dim = list(actions_dim)
    if continuous:
        return ActionCodec.from_space(gym.spaces.Box(-1, 1, (sum(actions_dim),)))
    if len(actions_dim) == 1:
        return ActionCodec.from_space(gym.spaces.Discrete(actions_dim[0]))
    return ActionCodec.from_space(gym.spaces.MultiDiscrete(actions_dim))
