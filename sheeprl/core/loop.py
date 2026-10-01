"""The training loop, the same for every algorithm."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import hydra
import torch
from lightning import Fabric
from torchmetrics import SumMetric

from sheeprl.core.algorithm import Algorithm, TrainState
from sheeprl.core.cadence import Cadence
from sheeprl.core.runner import EnvRunner
from sheeprl.core.schedule import TrainSchedule
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import save_configs


def run(fabric: Fabric, cfg: Dict[str, Any], algo: Algorithm) -> Tuple[TrainState, str]:
    """Train `algo` as configured by `cfg`, resuming from `cfg.checkpoint.resume_from` if set.

    Every iteration plays `algo.steps_per_iteration` steps in every environment, then trains on the batches of
    `algo.batches`, then logs and saves a checkpoint when it's time to.

    Returns:
        The trained state and the log directory of the run.
    """
    checkpoint = None
    if cfg.checkpoint.resume_from:
        checkpoint = fabric.load(cfg.checkpoint.resume_from, weights_only=False)
        cfg.algo.per_rank_batch_size = checkpoint["batch_size"] // fabric.world_size

    # The logger is created only on the rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    aggregator = None
    if not MetricAggregator.disabled:
        aggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(fabric.device)

    schedule = TrainSchedule(cfg, fabric.world_size, algo.steps_per_iteration, checkpoint)
    env = EnvRunner(fabric, cfg, log_dir, aggregator, policy_step=schedule.policy_step)
    state, store = algo.build(env.observation_space, env.action_space, schedule, log_dir)
    if checkpoint is not None:
        state.load_state_dict(checkpoint)
    if fabric.is_global_zero:
        save_configs(cfg, log_dir)
    cadence = Cadence(fabric, cfg, log_dir, aggregator, checkpoint)
    player = algo.player(state)

    env.reset()
    for iteration in schedule.iterations():
        # Play: the time includes the forward pass of the player
        with torch.inference_mode(), timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
            for _ in range(algo.steps_per_iteration):
                player.step(env, store)

        # Train
        trained = False
        with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
            for batch in algo.batches(state, store, schedule.gradient_steps(env.policy_step)):
                cadence.accumulate(algo.train_step(state, batch, schedule.gradient_step))
                schedule.gradient_step += 1
                trained = True
        if trained:
            cadence.train_step += fabric.world_size

        info = algo.end_iteration(state, iteration)
        if cfg.metric.log_level > 0 and info:
            fabric.log_dict(info, env.policy_step)
        cadence.log(env.policy_step, iteration, schedule.total_iters)
        cadence.checkpoint(state, schedule, env.policy_step, iteration)

    env.close()
    return state, log_dir
