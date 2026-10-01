"""The step bookkeeping of a run."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional


class TrainSchedule:
    """How many iterations a run lasts, where a resumed run starts, how many gradient steps each iteration does.

    One iteration plays `steps_per_iteration` steps in every environment of every process, then trains.
    The checkpoint of a resumed run is the one saved at the end of an iteration: the run starts from the next one.

    Policy steps are counted over all the environments of all the processes; gradient steps per process.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        world_size: int,
        steps_per_iteration: int,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.world_size = world_size
        # Policy steps played in one iteration by the environments of one process, and of all processes
        self.policy_steps_per_rank = cfg.env.num_envs * steps_per_iteration
        self.policy_steps_per_iter = self.policy_steps_per_rank * world_size
        self.total_iters = cfg.algo.total_steps // self.policy_steps_per_iter if not cfg.dry_run else 1

        self.start_iter = 1
        self.policy_step = 0
        self.gradient_step = 0
        if checkpoint is not None:
            # `iter_num` is saved multiplied by the number of processes
            self.start_iter = checkpoint["iter_num"] // world_size + 1
            self.policy_step = checkpoint["iter_num"] * self.policy_steps_per_rank
            self.gradient_step = checkpoint.get("per_rank_gradient_steps", 0)

        if cfg.metric.log_level > 0 and cfg.metric.log_every % self.policy_steps_per_iter != 0:
            warnings.warn(
                f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
                f"policy_steps_per_iter value ({self.policy_steps_per_iter}), so "
                "the metrics will be logged at the nearest greater multiple of the "
                "policy_steps_per_iter value."
            )
        if cfg.checkpoint.every % self.policy_steps_per_iter != 0:
            warnings.warn(
                f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
                f"policy_steps_per_iter value ({self.policy_steps_per_iter}), so "
                "the checkpoint will be saved at the nearest greater multiple of the "
                "policy_steps_per_iter value."
            )

    def iterations(self) -> range:
        """The iterations still to do, numbered from 1."""
        return range(self.start_iter, self.total_iters + 1)

    def gradient_steps(self, policy_step: int) -> Optional[int]:
        """The gradient steps the replay ratio asks for after `policy_step` policy steps.

        `None` for on-policy algorithms, which decide it from the rollout: they are the only ones ported so far.
        The replay ratio of the off-policy algorithms (`algo.replay_ratio`, `algo.learning_starts`) comes with them.
        """
        return None
