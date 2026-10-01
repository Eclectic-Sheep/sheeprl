"""Where the players write the collected data and the algorithms read their training data from.

On-policy algorithms write their steps in a `Rollout`; off-policy algorithms in a replay buffer of
`sheeprl.data.buffers`, which the training loop saves in the checkpoints when `buffer.checkpoint` is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, TypeVar

import numpy as np
from lightning import Fabric

from sheeprl.data.buffers import ReplayBuffer

Buffer = TypeVar("Buffer")


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


def load_replay_buffer(fabric: Fabric, saved: Any, buffer: Buffer) -> Buffer:
    """The replay buffer of this process, from the one saved in a checkpoint (`saved`) in place of `buffer`.

    A checkpoint holds the list of the buffers of all the processes, or a single buffer, which every process then
    starts from.
    """
    if isinstance(saved, list) and len(saved) == fabric.world_size:
        return saved[fabric.global_rank]
    if isinstance(saved, type(buffer)):
        return saved
    raise RuntimeError(
        f"The checkpoint holds {len(saved) if isinstance(saved, list) else 1} replay buffer(s), "
        f"but {fabric.world_size} processes are instantiated"
    )
