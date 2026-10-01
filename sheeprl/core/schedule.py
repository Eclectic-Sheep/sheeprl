"""The step bookkeeping of a run."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from sheeprl.utils.utils import Ratio


class TrainSchedule:
    """How many iterations a run lasts, where a resumed run starts, how many gradient steps each iteration does.

    One iteration plays `steps_per_iteration` steps in every environment of every process, then trains.
    The checkpoint of a resumed run is the one saved at the end of an iteration: the run starts from the next one.

    Policy steps are counted over all the environments of all the processes; gradient steps per process.

    Off-policy algorithms (`off_policy=True`) play random actions for the first `algo.learning_starts` policy steps
    (rounded down to whole iterations), to fill their replay buffer, then do `algo.replay_ratio` gradient steps per
    policy step (`Ratio`).
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        world_size: int,
        steps_per_iteration: int,
        checkpoint: Optional[Dict[str, Any]] = None,
        off_policy: bool = False,
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

        # The replay ratio of the off-policy algorithms; `None` for on-policy algorithms
        self.ratio: Optional[Ratio] = None
        # The iterations that play random actions: 1 to `learning_starts`
        self.learning_starts = 0
        if off_policy:
            self.learning_starts = cfg.algo.learning_starts // self.policy_steps_per_iter if not cfg.dry_run else 0
            # The replay ratio counts the policy steps from the start of the last iteration of random actions
            self.prefill_steps = self.learning_starts - int(self.learning_starts > 0)
            if checkpoint is not None:
                # A resumed run plays random actions again, even when the replay buffer was saved in the
                # checkpoint (known issue #42)
                self.learning_starts += self.start_iter
                self.prefill_steps += self.start_iter
            self.ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
            if checkpoint is not None:
                self.ratio.load_state_dict(checkpoint["ratio"])
            # One gradient step per iteration, whatever the replay ratio (`exp=sac_benchmarks`)
            self.run_benchmarks = bool(cfg.get("run_benchmarks", False))

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

    @property
    def off_policy(self) -> bool:
        return self.ratio is not None

    def iterations(self) -> range:
        """The iterations still to do, numbered from 1."""
        return range(self.start_iter, self.total_iters + 1)

    def warmup(self, policy_step: int) -> bool:
        """Whether the environments play random actions after `policy_step` policy steps: in the iterations from 1 to
        `learning_starts` of the off-policy algorithms."""
        return policy_step < self.learning_starts * self.policy_steps_per_iter

    def gradient_steps(self, iteration: int) -> Optional[int]:
        """The gradient steps every process does at the end of `iteration`.

        `None` for on-policy algorithms, which decide it from the rollout. For off-policy algorithms, 0 until the
        iteration `learning_starts`, then `algo.replay_ratio` gradient steps per policy step of the process.
        """
        if self.ratio is None:
            return None
        if iteration < self.learning_starts:
            return 0
        if self.run_benchmarks:
            return 1
        # The policy steps played from the start of the last iteration of random actions, by all processes
        policy_steps = (iteration - self.prefill_steps) * self.policy_steps_per_iter
        # `Ratio` can return a negative number, e.g. after resuming
        return max(0, self.ratio(policy_steps / self.world_size))
