"""Where the players write the collected data and the algorithms read their training data from.

On-policy algorithms write their steps in a `Rollout`; off-policy algorithms in a `ReplayStore` (a replay buffer of
`sheeprl.data.buffers` read by a sampler of `sheeprl.data.samplers`), which the training loop saves in the checkpoints
when `buffer.checkpoint` is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, TypeVar

import numpy as np
from lightning import Fabric

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.store import ReplayStore

Store = TypeVar("Store")


@dataclass
class Rollout:
    """The store of the on-policy algorithms: the steps of the current rollout, and the observations that follow
    its last step, whose value bootstraps the returns."""

    buffer: ReplayBuffer
    next_obs: Dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, data: Dict[str, Any], next_obs: Dict[str, np.ndarray], validate_args: bool = False) -> None:
        """Write one step (`data`, with shape `[1, num_envs, ...]`) and remember the observations that follow it."""
        self.buffer.add(data, validate_args=validate_args)
        self.next_obs = next_obs


def load_replay_buffer(fabric: Fabric, saved: Any, store: Store) -> Store:
    """The replay buffer of this process, from the one saved in a checkpoint (`saved`) in place of `store`.

    A checkpoint holds the list of the buffers of all the processes, or a single buffer, which every process then
    starts from. A `ReplayStore` takes the storage and the sampler of the saved one (`ReplayStore.load`), or the saved
    storage of a checkpoint of sheeprl up to 0.8.2.
    """
    if isinstance(saved, list):
        if len(saved) != fabric.world_size:
            raise RuntimeError(
                f"The checkpoint holds {len(saved)} replay buffer(s), "
                f"but {fabric.world_size} processes are instantiated"
            )
        saved = saved[fabric.global_rank]
    if isinstance(store, ReplayStore):
        return store.load(saved)
    if isinstance(saved, type(store)):
        return saved
    raise RuntimeError(
        f"The checkpoint holds a replay buffer of type {type(saved).__name__}, not {type(store).__name__}"
    )
