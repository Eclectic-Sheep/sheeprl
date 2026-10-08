"""The training loop, the same for every algorithm."""

from __future__ import annotations

import copy
import tempfile
import warnings
from typing import Any, Dict, Tuple

import gymnasium as gym
import hydra
import torch
from lightning import Fabric

from sheeprl.core.algorithm import Algorithm, TrainState
from sheeprl.core.cadence import Cadence
from sheeprl.core.collector import Collector
from sheeprl.core.environment import GymEnvironment
from sheeprl.core.schedule import TrainSchedule
from sheeprl.data.store import load_replay_buffer
from sheeprl.utils import fs
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.timer import phase_timer, training_timer
from sheeprl.utils.utils import dotdict, save_configs


def run(fabric: Fabric, cfg: Dict[str, Any], algo: Algorithm) -> Tuple[TrainState, str, int]:
    """Train `algo` as configured by `cfg`, resuming from `cfg.checkpoint.resume_from` if set.

    Every iteration plays `algo.steps_per_iteration` steps in every environment, then trains on the batches of
    `algo.batches`, then logs and saves a checkpoint when it's time to.

    Returns:
        The trained state, the log directory of the run and its last policy step.
    """
    checkpoint = None
    if cfg.checkpoint.resume_from:
        checkpoint = fs.load_checkpoint(fabric, cfg.checkpoint.resume_from, weights_only=False)
        cfg.algo.per_rank_batch_size = checkpoint["batch_size"] // fabric.world_size

    # The logger is created only on the rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name, log_root=cfg.log_root)
    fabric.print(f"Log dir: {log_dir}")

    aggregator = None
    if not MetricAggregator.disabled:
        aggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(fabric.device)

    schedule = TrainSchedule(cfg, fabric.world_size, algo.steps_per_iteration, checkpoint, algo.off_policy)
    env = GymEnvironment.from_config(fabric, cfg, log_dir, restart_on_exception=algo.restart_crashed_envs)
    state, store = algo.build(env.observation_space, env.action_space, schedule, log_dir)
    # The replay buffer of the off-policy algorithms is saved in the checkpoints
    save_buffer = algo.off_policy and cfg.buffer.checkpoint
    if checkpoint is not None:
        state.load_state_dict(checkpoint)
        if save_buffer:
            store = load_replay_buffer(fabric, checkpoint["rb"], store)
    if fabric.is_global_zero:
        save_configs(cfg, log_dir)
    cadence = Cadence(fabric, cfg, log_dir, aggregator, checkpoint, policy_step=schedule.policy_step)
    policy = algo.policy(state)
    collector = Collector(env, policy, algo.writer(state, policy), store, schedule, cadence, algo.random_warmup)
    with torch.inference_mode():
        collector.reset()
    for iteration in schedule.iterations():
        # Play: the time includes the forward pass of the policy
        with torch.inference_mode(), phase_timer("Time/env_interaction_time"):
            collector.collect(algo.steps_per_iteration)

        # Train: `n_steps` is None for on-policy algorithms (they decide it from the rollout), and is 0 for
        # off-policy algorithms before `algo.learning_starts`
        n_steps = schedule.gradient_steps(iteration)
        if n_steps != 0:
            # The timer waits for the GPU to finish the training, which would otherwise be timed with the interaction
            with training_timer(fabric.device):
                for batch in algo.batches(state, store, n_steps, iteration):
                    cadence.accumulate(algo.train_step(state, batch, schedule.gradient_step))
                    schedule.gradient_step += 1

        info = algo.end_iteration(state, iteration)
        if cfg.metric.log_level > 0 and info:
            fabric.log_dict(info, schedule.policy_step)
        cadence.log(schedule.policy_step, iteration, schedule)
        cadence.checkpoint(state, schedule, schedule.policy_step, iteration, store if save_buffer else None)

    # Every process returns once all of them have trained: the process of rank 0 doesn't go on (to test the agent, or
    # to end the run and remove its files) while the others still train (and write in their files)
    fabric.barrier()
    env.close()
    return state, log_dir, schedule.policy_step


def load_trained_state(
    fabric: Fabric,
    cfg: Dict[str, Any],
    algo: Algorithm,
    checkpoint: Dict[str, Any],
    observation_space: gym.spaces.Dict,
    action_space: gym.Space,
) -> TrainState:
    """The training state of `algo`, restored from `checkpoint`: the trained models, e.g. to evaluate or register them.

    The state is built by `algo.build`, as for training. The store of the collected data is not needed: it is built in
    a temporary directory and discarded.
    """
    with warnings.catch_warnings():
        # The warnings of the schedule are about the logging and checkpoint intervals of a training
        warnings.simplefilter("ignore")
        schedule = TrainSchedule(cfg, fabric.world_size, algo.steps_per_iteration, off_policy=algo.off_policy)
    # The store of the collected data is built as in a dry run (small and in memory): the size of the one of the
    # training could exceed the memory, or the disk
    training_cfg = algo.cfg
    algo.cfg = dotdict(copy.deepcopy(training_cfg.as_dict()))
    algo.cfg.dry_run = True
    algo.cfg.buffer.memmap = False
    try:
        with tempfile.TemporaryDirectory() as store_dir:
            state, _ = algo.build(observation_space, action_space, schedule, store_dir)
    finally:
        algo.cfg = training_cfg
    state.load_state_dict(checkpoint)
    return state
