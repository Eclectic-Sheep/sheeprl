"""The actions of the agents and of the environments, and the translation between them (`ActionCodec`)."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Iterator, List, Optional, Sequence, Tuple

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
    categorical heads, the values of the continuous ones).

    The heads may mix categorical and continuous actions: the ones of a `Tuple` or a `Dict` action space (e.g. a
    discrete action and a continuous one), whose actions the environments take with the structure of the space.
    """

    def __init__(self, heads: Sequence[ActionHead], space: Optional[gym.Space] = None) -> None:
        if not heads:
            raise ValueError("An action space has at least one action")
        self.heads = tuple(heads)
        # As Python integers: `torch.compile` reads the NumPy ones as tensors
        self.dims: Tuple[int, ...] = tuple(int(head.size) for head in self.heads)
        # Whether every head is continuous (one flag per head in `continuous_heads`), or every head categorical:
        # neither for a mixed action space
        self.continuous_heads: Tuple[bool, ...] = tuple(head.kind == "continuous" for head in self.heads)
        self.continuous = all(self.continuous_heads)
        self.categorical = all(head.kind == "categorical" for head in self.heads)
        # The action space when it is a tuple or a dictionary of spaces: the actions of the environments have its
        # structure (`to_env`, `from_env`)
        self.space = space if isinstance(space, (gym.spaces.Tuple, gym.spaces.Dict)) else None

    @classmethod
    def from_space(cls, space: gym.Space) -> ActionCodec:
        """`Discrete`: one categorical head; `MultiDiscrete`: one per entry of `nvec`; `MultiBinary`: one categorical
        head of size 2 per entry; `Box`: one continuous head of all its dimensions; `Tuple` and `Dict`: the heads of
        their parts, in order."""
        return cls(_heads(space), space)

    def is_continuous(self, head: int) -> bool:
        """Whether the head `head` is continuous."""
        return self.heads[head].kind == "continuous"

    def to_env(self, actions: Sequence[Tensor]) -> Any:
        """The actions of the environments: the index of every categorical head, the values of the continuous ones;
        for a `Tuple` or a `Dict` action space, a tuple or a dictionary of the actions of its parts."""
        if self.space is not None:
            values = (
                (a.argmax(dim=-1) if head.kind == "categorical" else a).cpu().numpy()
                for a, head in zip(actions, self.heads)
            )
            return _assemble(self.space, values)
        if self.categorical:
            return torch.stack([a.argmax(dim=-1) for a in actions], dim=-1).cpu().numpy()
        return torch.cat(list(actions), dim=-1).cpu().numpy()

    def to_store(self, actions: Sequence[Tensor]) -> np.ndarray:
        """The actions as the store keeps them: the heads concatenated on the last dimension."""
        return torch.cat(list(actions), dim=-1).cpu().numpy()

    def from_env(self, env_actions: Any) -> np.ndarray:
        """The actions of the environments (e.g. the random ones before the training) as the store keeps them: the
        categorical heads one-hot, one row per environment."""
        if self.space is not None:
            return np.concatenate(_disassemble(self.space, env_actions), axis=-1)
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


def _assemble(space: gym.Space, values: Iterator[np.ndarray]) -> Any:
    """The actions of the environments of the action space `space`, from the values of its heads in order (the indices
    of the categorical heads, the values of the continuous ones), one row per environment: the leading dimensions of
    the values (e.g. the time step of the recurrent policies) are flattened into the rows."""
    if isinstance(space, gym.spaces.Tuple):
        return tuple(_assemble(part, values) for part in space.spaces)
    if isinstance(space, gym.spaces.Dict):
        return {key: _assemble(part, values) for key, part in space.spaces.items()}
    if isinstance(space, gym.spaces.Discrete):
        return next(values).reshape(-1) + int(space.start)
    if isinstance(space, gym.spaces.MultiDiscrete):
        heads = [next(values) for _ in range(int(np.prod(space.nvec.shape)))]
        return np.stack(heads, axis=-1).reshape(-1, *space.nvec.shape) + space.start
    if isinstance(space, gym.spaces.MultiBinary):
        heads = [next(values) for _ in range(int(np.prod(space.shape)))]
        return np.stack(heads, axis=-1).reshape(-1, *space.shape).astype(space.dtype)
    if isinstance(space, gym.spaces.Box):
        return next(values).reshape(-1, *space.shape).astype(space.dtype, copy=False)
    raise NotImplementedError(f"The action space {space} is not supported")


def _one_hot(indices: np.ndarray, size: int) -> np.ndarray:
    return F.one_hot(torch.as_tensor(indices, dtype=torch.int64), size).numpy().astype(np.float32)


def _disassemble(space: gym.Space, actions: Any) -> List[np.ndarray]:
    """The actions of the environments of the action space `space` as the store keeps them, one array per head in
    order (the categorical heads one-hot), one row per environment: the inverse of `_assemble`."""
    if isinstance(space, gym.spaces.Tuple):
        return [head for part, action in zip(space.spaces, actions) for head in _disassemble(part, action)]
    if isinstance(space, gym.spaces.Dict):
        return [head for key, part in space.spaces.items() for head in _disassemble(part, actions[key])]
    actions = np.asarray(actions)
    if isinstance(space, gym.spaces.Discrete):
        return [_one_hot(actions.reshape(-1) - int(space.start), int(space.n))]
    if isinstance(space, gym.spaces.MultiDiscrete):
        indices = (actions - space.start).reshape(len(actions), -1)
        return [_one_hot(indices[:, i], int(n)) for i, n in enumerate(space.nvec.reshape(-1))]
    if isinstance(space, gym.spaces.MultiBinary):
        bits = actions.reshape(len(actions), -1)
        return [_one_hot(bits[:, i], 2) for i in range(bits.shape[1])]
    if isinstance(space, gym.spaces.Box):
        return [actions.reshape(len(actions), -1).astype(np.float32)]
    raise NotImplementedError(f"The action space {space} is not supported")


def batched_env_actions(actions: Any, space: gym.Space) -> Any:
    """The actions `actions` (one row per environment) as the vectorized environments of the batched action space
    `space` take them: reshaped to its shape, or as they are for a tuple or a dictionary of spaces."""
    if isinstance(space, (gym.spaces.Tuple, gym.spaces.Dict)):
        return actions
    return actions.reshape(space.shape)


def single_env_actions(actions: Any, space: gym.Space) -> Any:
    """The actions of one environment of the action space `space`, from `actions` with one row for it."""
    if isinstance(space, gym.spaces.Tuple):
        return tuple(single_env_actions(action, part) for action, part in zip(actions, space.spaces))
    if isinstance(space, gym.spaces.Dict):
        return {key: single_env_actions(actions[key], part) for key, part in space.spaces.items()}
    return np.asarray(actions).reshape(space.shape)
