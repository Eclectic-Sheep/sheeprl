from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal

from sheeprl.data.buffers import EnvIndependentReplayBuffer
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.memmap import MemmapArray
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/post_entropy",
    "State/prior_entropy",
    "State/kl",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
    "Params/exploration_amount",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic"}


def add_is_first(rb: EnvIndependentReplayBuffer) -> EnvIndependentReplayBuffer:
    """Complete a replay buffer saved before DreamerV1 stored `is_first` (a resumed run, or a finetuning that loads the
    buffer of its exploration, would fail to add rows with it).

    A row is the first of an episode when the row before it, in the buffer of the same environment, ends one
    (`terminated` or `truncated`: the next row holds the first observation of the new episode), or when it is the first
    row of a buffer not filled yet.
    """
    for buffer in rb.buffer:
        if buffer.empty or "is_first" in buffer.buffer:
            continue
        terminated, truncated = (
            value.array if isinstance(value, MemmapArray) else value
            for value in (buffer["terminated"], buffer["truncated"])
        )
        is_first = np.roll(np.logical_or(terminated, truncated), 1, axis=0)
        if not buffer.full:
            is_first[0] = True
        buffer["is_first"] = is_first.astype(terminated.dtype)
    return rb


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    done_mask: Tensor,
    last_values: Tensor,
    horizon: int = 15,
    lmbda: float = 0.95,
) -> Tensor:
    """
    Compute the lambda values by keeping the gradients of the variables.

    Args:
        rewards (Tensor): the estimated rewards in the latent space.
        values (Tensor): the estimated values in the latent space.
        done_mask (Tensor): 1s for the entries that are relative to a terminal step, 0s otherwise.
        last_values (Tensor): the next values for the last state in the horzon.
        horizon: (int, optional): the horizon of imagination.
            Default to 15.
        lmbda (float, optional): the discout lmbda factor for the lambda values computation.
            Default to 0.95.

    Returns:
        The tensor of the computed lambda values.
    """
    last_values = torch.clone(last_values)
    last_lambda_values = 0
    lambda_targets = []
    for step in reversed(range(horizon - 1)):
        if step == horizon - 2:
            next_values = last_values
        else:
            next_values = values[step + 1] * (1 - lmbda)
        delta = rewards[step] + next_values * done_mask[step]
        last_lambda_values = delta + lmbda * done_mask[step] * last_lambda_values
        lambda_targets.append(last_lambda_values)
    return torch.stack(list(reversed(lambda_targets)), dim=0)


def compute_stochastic_state(
    state_information: Tensor, event_shape: int = 1, min_std: float = 0.1
) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
    """
    Compute the stochastic state from the information of the distribution of the stochastic state.

    Args:
        state_information (Tensor): information about the distribution of the stochastic state,
            it is the output of either the representation model or the transition model.
        event_shape (int): how many batch dimensions have to be reinterpreted as event dims.
            Default to 1.
        min_std (float): the minimum value for the standard deviation.
            Default to 0.1.

    Returns:
        The mean and the standard deviation of the distribution of the stochastic state.
        The sampled stochastic state.
    """
    mean, std = torch.chunk(state_information, 2, -1)
    std = F.softplus(std) + min_std
    state_distribution: Distribution = Normal(mean, std)
    if event_shape:
        # it is necessary an Independent distribution because
        # it is necessary to create (batch_size * sequence_length) independent distributions,
        # each producing a sample of size equal to the stochastic size
        state_distribution = Independent(state_distribution, event_shape)
    stochastic_state = state_distribution.rsample()
    return (mean, std), stochastic_state


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
    from sheeprl.algos.dreamer_v1.dreamer_v1 import DreamerV1
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    return log_trained_models(
        fabric,
        env,
        cfg,
        state,
        DreamerV1(fabric, cfg.to_log),
        lambda trained: {"world_model": trained.world_model, "actor": trained.actor, "critic": trained.critic},
    )
