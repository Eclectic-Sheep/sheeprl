"""The actions of the agents and of the environments, and the translation between them (`ActionCodec`)."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import List, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ActionHead:
    """One action of the agent: `categorical` among `size` choices, or `continuous` of `size` dimensions."""

    kind: str
    size: int


def _heads(space: gym.Space) -> List[ActionHead]:
    if isinstance(space, gym.spaces.Discrete):
        return [ActionHead("categorical", int(space.n))]
    if isinstance(space, gym.spaces.MultiDiscrete):
        return [ActionHead("categorical", int(n)) for n in np.asarray(space.nvec).reshape(-1)]
    if isinstance(space, gym.spaces.MultiBinary):
        return [ActionHead("categorical", 2) for _ in range(int(prod(space.shape)))]
    if isinstance(space, gym.spaces.Box):
        return [ActionHead("continuous", int(prod(space.shape)))]
    if isinstance(space, gym.spaces.Tuple):
        return [head for part in space.spaces for head in _heads(part)]
    if isinstance(space, gym.spaces.Dict):
        return [head for part in space.spaces.values() for head in _heads(part)]
    raise NotImplementedError(f"The action space {space} is not supported")


class ActionCodec:
    """Between the actions of the agent (what its distributions produce and the store keeps: one tensor per head,
    one-hot for the categorical heads) and the actions of the environments (what `env.step` takes: the indices of the
    categorical heads, the values of the continuous ones)."""

    def __init__(self, heads: Sequence[ActionHead]) -> None:
        if not heads:
            raise ValueError("An action space has at least one action")
        self.heads = tuple(heads)
        # As Python integers: `torch.compile` reads the NumPy ones as tensors
        self.dims: Tuple[int, ...] = tuple(int(head.size) for head in self.heads)
        self.continuous = all(head.kind == "continuous" for head in self.heads)
        self.categorical = all(head.kind == "categorical" for head in self.heads)

    @classmethod
    def from_space(cls, space: gym.Space) -> ActionCodec:
        """`Discrete`: one categorical head; `MultiDiscrete`: one per entry of `nvec`; `MultiBinary`: one categorical
        head of size 2 per entry; `Box`: one continuous head of all its dimensions; `Tuple` and `Dict`: the heads of
        their parts, in order."""
        return cls(_heads(space))

    def check(self, actor: str) -> None:
        """An actor supports the action spaces of all-categorical heads or of continuous ones: `actor` names it in the
        error of a mixed space."""
        if not (self.continuous or self.categorical):
            raise NotImplementedError(
                f"{actor} plays categorical or continuous actions, not both: got the heads {list(self.heads)}"
            )

    def to_env(self, actions: Sequence[Tensor]) -> np.ndarray:
        """The actions of the environments: the index of every categorical head, the values of the continuous ones."""
        if self.categorical:
            return torch.stack([a.argmax(dim=-1) for a in actions], dim=-1).cpu().numpy()
        return torch.cat(list(actions), dim=-1).cpu().numpy()

    def to_store(self, actions: Sequence[Tensor]) -> np.ndarray:
        """The actions as the store keeps them: the heads concatenated on the last dimension."""
        return torch.cat(list(actions), dim=-1).cpu().numpy()

    def from_env(self, env_actions: np.ndarray) -> np.ndarray:
        """The actions of the environments (e.g. the random ones before the training) as the store keeps them: the
        categorical heads one-hot, one row per environment."""
        if self.continuous:
            return env_actions
        num_envs = len(env_actions)
        # One row per environment, one column per head: one-hot each column
        per_head = np.asarray(env_actions).reshape(num_envs, len(self.dims)).T
        return np.concatenate(
            [F.one_hot(torch.as_tensor(a), dim).numpy() for a, dim in zip(per_head, self.dims)], axis=-1
        )

    def split(self, stored: Tensor) -> Sequence[Tensor]:
        """The stored actions, one tensor per head."""
        return torch.split(stored, list(self.dims), dim=-1)
