"""When and how a run logs its metrics and saves its checkpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from lightning import Fabric
from torch import Tensor

from sheeprl.core.algorithm import TrainState
from sheeprl.core.schedule import TrainSchedule
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.timer import timer


class Cadence:
    """Accumulates the metrics of the training steps, logs them every `metric.log_every` policy steps and saves a
    checkpoint every `checkpoint.every` policy steps (and at the end of the run if `checkpoint.save_last`)."""

    def __init__(
        self,
        fabric: Fabric,
        cfg: Dict[str, Any],
        log_dir: str,
        aggregator: Optional[MetricAggregator],
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.log_dir = log_dir
        self.aggregator = aggregator
        self.last_log = checkpoint["last_log"] if checkpoint is not None else 0
        self.last_checkpoint = checkpoint["last_checkpoint"] if checkpoint is not None else 0
        # Training phases done by all the processes, to measure the training speed
        self.train_step = 0
        self.last_train = 0

    def accumulate(self, metrics: Dict[str, Tensor]) -> None:
        """Add the metrics of one training step. Only the metrics listed in `metric.aggregator.metrics` are kept."""
        if self.aggregator is not None and not self.aggregator.disabled:
            for name, value in metrics.items():
                if name in self.aggregator:
                    self.aggregator.update(name, value)

    def log(self, policy_step: int, iteration: int, schedule: TrainSchedule) -> None:
        """Log the accumulated metrics and the speed of the run, if it's time to."""
        cfg = self.cfg
        if cfg.metric.log_level == 0:
            return
        if policy_step - self.last_log < cfg.metric.log_every and iteration != schedule.total_iters:
            return
        if self.aggregator is not None and not self.aggregator.disabled:
            self.fabric.log_dict(self.aggregator.compute(), policy_step)
            self.aggregator.reset()
        if schedule.off_policy:
            # Gradient steps of all the processes per policy step
            self.fabric.log(
                "Params/replay_ratio", schedule.gradient_step * self.fabric.world_size / policy_step, policy_step
            )
        if not timer.disabled:
            timer_metrics = timer.compute()
            if timer_metrics.get("Time/train_time", 0) > 0:
                self.fabric.log(
                    "Time/sps_train",
                    (self.train_step - self.last_train) / timer_metrics["Time/train_time"],
                    policy_step,
                )
            if timer_metrics.get("Time/env_interaction_time", 0) > 0:
                self.fabric.log(
                    "Time/sps_env_interaction",
                    ((policy_step - self.last_log) / self.fabric.world_size * cfg.env.action_repeat)
                    / timer_metrics["Time/env_interaction_time"],
                    policy_step,
                )
            timer.reset()
        self.last_log = policy_step
        self.last_train = self.train_step

    def checkpoint(
        self,
        state: TrainState,
        schedule: TrainSchedule,
        policy_step: int,
        iteration: int,
        replay_buffer: Optional[Any] = None,
    ) -> None:
        """Save the training state and the counters needed to resume the run, if it's time to.

        `replay_buffer`, if given, is saved too: with several processes, the buffers of all of them, in the
        checkpoint of rank 0.
        """
        cfg = self.cfg
        every = cfg.checkpoint.every > 0 and policy_step - self.last_checkpoint >= cfg.checkpoint.every
        last = iteration == schedule.total_iters and cfg.checkpoint.save_last
        if not (every or last):
            return
        self.last_checkpoint = policy_step
        world_size = self.fabric.world_size
        ckpt = {
            **state.state_dict(),
            "iter_num": iteration * world_size,
            "batch_size": cfg.algo.per_rank_batch_size * world_size,
            "per_rank_gradient_steps": schedule.gradient_step,
            "last_log": self.last_log,
            "last_checkpoint": self.last_checkpoint,
        }
        if schedule.ratio is not None:
            ckpt["ratio"] = schedule.ratio.state_dict()
        ckpt_path = os.path.join(self.log_dir, f"checkpoint/ckpt_{policy_step}_{self.fabric.global_rank}.ckpt")
        self.fabric.call(
            "on_checkpoint",
            fabric=self.fabric,
            ckpt_path=ckpt_path,
            state=ckpt,
            replay_buffer=replay_buffer,
        )
