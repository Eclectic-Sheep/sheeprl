"""The interface every algorithm implements, and the training state it works on.

An algorithm is a subclass of `Algorithm` with four methods, plus an optional fifth:

- `build`: create the modules, the optimizers and the store of the collected data;
- `player`: the object that plays in the environments and writes what happens in the store;
- `batches`: the training data of one iteration, one batch per gradient step;
- `train_step`: one gradient step on a batch;
- `end_iteration` (optional): what changes once per iteration, e.g. annealed coefficients.

The training loop that calls them, identical for every algorithm, is `sheeprl.core.loop.run`.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterator, Protocol, Tuple

import gymnasium as gym
import torch
from lightning import Fabric
from torch import Tensor, nn

if TYPE_CHECKING:
    from sheeprl.core.runner import EnvRunner
    from sheeprl.core.schedule import TrainSchedule


@dataclasses.dataclass
class TrainState:
    """Everything that changes during training: modules, optimizers, learning-rate schedulers, annealed
    coefficients (kept as tensors), counters.

    An algorithm subclasses it as a dataclass and lists its fields. No logic lives in it: the training loop saves it
    in the checkpoints (one entry per field, with the name of the field) and restores it when a run is resumed.
    """

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of every field: `state_dict()` for the objects that have one (modules, optimizers,
        schedulers), the value itself for tensors and plain Python values."""
        state = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            state[field.name] = value.state_dict() if hasattr(value, "state_dict") else value
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore, in place, the fields saved in `state`. Fields missing from `state` keep their value, with a
        warning (e.g. a field added after the checkpoint was saved)."""
        missing = []
        for field in dataclasses.fields(self):
            if field.name not in state:
                missing.append(field.name)
                continue
            value = getattr(self, field.name)
            if isinstance(value, nn.Module):
                load_module_state_dict(value, state[field.name])
            elif hasattr(value, "load_state_dict"):
                value.load_state_dict(state[field.name])
            elif isinstance(value, Tensor):
                value.copy_(state[field.name])
            else:
                setattr(self, field.name, state[field.name])
        if missing:
            warnings.warn(f"Not found in the checkpoint, so not restored: {', '.join(missing)}", UserWarning)


def load_module_state_dict(module: nn.Module, state: Dict[str, Tensor]) -> None:
    """Copy `state` into the parameters and buffers of `module`, like `module.load_state_dict(state)`.

    `load_state_dict` fails when some submodules are wrapped by Fabric (see `sheeprl.core.update.setup_module`): it
    expects the names of the wrappers in the keys, while `state_dict()` (and so every checkpoint) doesn't have them.
    The keys of `module.state_dict()` are the ones of the checkpoint, and with `keep_vars=True` its values are the
    parameters and buffers themselves.
    """
    targets = module.state_dict(keep_vars=True)
    missing, unexpected = targets.keys() - state.keys(), state.keys() - targets.keys()
    if missing or unexpected:
        raise RuntimeError(
            f"Error(s) in loading the state of {type(module).__name__}: "
            f"missing keys {sorted(missing)}, unexpected keys {sorted(unexpected)}"
        )
    with torch.no_grad():
        for name, target in targets.items():
            target.copy_(state[name])


class Player(Protocol):
    """Plays in the environments. Built by `Algorithm.player` from the current training state."""

    def step(self, env: EnvRunner, store: Any) -> None:
        """Choose the actions for `env.obs`, step the environments with `env.step(actions)` and write what happened
        in `store`."""


class Algorithm:
    """Base class of the algorithms. See the module docstring for what each method does."""

    # Environment steps played by every environment before each training phase
    # (e.g. the rollout length of on-policy algorithms; 1 for off-policy algorithms)
    steps_per_iteration: int = 1
    # Whether the algorithm trains on a replay buffer of past steps. Then the training starts after
    # `algo.learning_starts` policy steps, does `algo.replay_ratio` gradient steps per policy step, and the buffer
    # (the store returned by `build`) is saved in the checkpoints when `buffer.checkpoint` is set
    off_policy: bool = False

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        self.fabric = fabric
        self.cfg = cfg

    def build(
        self,
        obs_space: gym.spaces.Dict,
        action_space: gym.Space,
        schedule: TrainSchedule,
        log_dir: str,
    ) -> Tuple[TrainState, Any]:
        """Create the training state (modules on the device, optimizers, ...) and the store of the collected data.

        When a run is resumed, the training loop restores the returned state from the checkpoint.
        """
        raise NotImplementedError

    def player(self, state: TrainState) -> Player:
        """Return the object that plays the current policy in the environments."""
        raise NotImplementedError

    def batches(
        self, state: TrainState, store: Any, n_steps: int | None, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        """Prepare the training data of the iteration `iteration` and yield one batch per gradient step.

        `n_steps` is the number of gradient steps asked by the replay ratio of off-policy algorithms (never 0: the
        training loop doesn't call `batches` then); it is `None` for on-policy algorithms, which decide it from the
        rollout (epochs x minibatches).
        """
        raise NotImplementedError

    def train_step(self, state: TrainState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        """One gradient step: update `state` in place and return the metrics to log, as tensors.

        `step` counts the gradient steps done by this process since the start of the training: use it for the
        updates that don't happen at every step (e.g. target networks). Don't read tensors on the CPU here
        (`.item()`, `if` on a tensor): the metrics are read once per log interval.
        """
        raise NotImplementedError

    def end_iteration(self, state: TrainState, iteration: int) -> Dict[str, float]:
        """Optional: what changes once per iteration, after the training steps (e.g. annealed coefficients).

        Returns values to log at every iteration (e.g. the learning rate that was used).
        """
        return {}
