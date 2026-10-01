"""Where the players write the collected data and the algorithms read their training data from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from sheeprl.data.buffers import ReplayBuffer


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
