from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch.optim import Optimizer

from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import polynomial_decay, unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/value_loss", "Loss/policy_loss", "Loss/entropy_loss"}
MODELS_TO_REGISTER = {"agent"}


def anneal(
    cfg: Dict[str, Any],
    optimizer: Optimizer,
    iteration: int,
    total_iters: int,
    initial_clip_coef: float,
    initial_ent_coef: float,
) -> None:
    """Set the learning rate of `optimizer` and the coefficients `cfg.algo.clip_coef` and `cfg.algo.ent_coef` that
    `algo.anneal_lr`, `algo.anneal_clip_coef` and `algo.anneal_ent_coef` anneal to their values for the iteration
    `iteration` (numbered from 1) of `total_iters`: a linear decay from the configured values, to 0 at the end of the
    training. They depend only on the iteration, so a resumed run follows the schedule of its own `algo.total_steps`
    (and starts from the values of the iteration it resumes from)."""
    # The first iteration uses the configured values
    decay = partial(polynomial_decay, iteration - 1, final=0.0, max_decay_steps=total_iters)
    if cfg.algo.anneal_lr:
        for group in optimizer.param_groups:
            group["lr"] = decay(initial=cfg.algo.optimizer.lr)
    if cfg.algo.anneal_clip_coef:
        cfg.algo.clip_coef = decay(initial=initial_clip_coef)
    if cfg.algo.anneal_ent_coef:
        cfg.algo.ent_coef = decay(initial=initial_ent_coef)


def bootstrap_truncated(
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    final_values: Callable[[np.ndarray], np.ndarray],
    gamma: float,
) -> np.ndarray:
    """The rewards of a step of the environments, with the discounted value of the final observation added to the
    reward of every episode truncated by the time limit.

    An episode both truncated and terminated in the same step (gymnasium's `TimeLimit` truncates also when the
    termination falls on the last allowed step) ended: it isn't bootstrapped. The rewards are expected already clipped,
    if they are: the value of the final observation is not a reward to clip.

    Args:
        rewards (np.ndarray): the rewards of the environments, one per environment.
        terminated (np.ndarray): whether the episode of every environment terminated.
        truncated (np.ndarray): whether the episode of every environment was truncated.
        final_values (Callable[[np.ndarray], np.ndarray]): the values of the final observations of the environments
            whose indices it receives.
        gamma (float): the discount factor.

    Returns:
        The bootstrapped rewards (a new array).
    """
    # A copy, in floating point (of the precision of the rewards, at least float32)
    rewards = np.array(rewards, dtype=np.result_type(np.asarray(rewards).dtype, np.float32))
    truncated_envs = np.nonzero(np.logical_and(truncated, np.logical_not(terminated)))[0]
    if len(truncated_envs) > 0:
        rewards[truncated_envs] += gamma * np.asarray(final_values(truncated_envs)).reshape(len(truncated_envs))
    return rewards


def log_models(
    cfg: Dict[str, Any],
    models_to_log: Dict[str, torch.nn.Module | _FabricModule],
    run_id: str,
    experiment_id: str | None = None,
    run_name: str | None = None,
) -> Dict[str, "ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=True) as _:
        model_info = {}
        unwrapped_models = {}
        for k in cfg.model_manager.models.keys():
            if k not in models_to_log:
                warnings.warn(f"Model {k} not found in models_to_log, skipping.", category=UserWarning)
                continue
            unwrapped_models[k] = unwrap_fabric(models_to_log[k])
            model_info[k] = mlflow.pytorch.log_model(unwrapped_models[k], name=k, serialization_format="pickle")
        mlflow.log_dict(cfg, "config.json")
    return model_info


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    from sheeprl.algos.ppo.ppo import PPO
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    return log_trained_models(
        fabric, env, cfg, state, PPO(fabric, cfg.to_log), lambda trained: {"agent": trained.agent}
    )
