from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor

from sheeprl.algos.sac.agent import SACPolicy
from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/alpha_loss",
}
MODELS_TO_REGISTER = {"agent"}


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, mlp_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Tensor:
    with fabric.device:
        torch_obs = torch.cat([torch.as_tensor(obs[k].copy(), dtype=torch.float32) for k in mlp_keys], dim=-1)
    return torch_obs.reshape(num_envs, -1)


@torch.no_grad()
def test(actor: SACPolicy, fabric: Fabric, cfg: Dict[str, Any], log_dir: str, policy_step: int = 0):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    actor.eval()
    done = False
    cumulative_rew = 0
    obs = env.reset(seed=cfg.seed)[0]
    while not done:
        # Act greedly through the environment
        torch_obs = prepare_obs(fabric, obs, mlp_keys=cfg.algo.mlp_keys.encoder)
        action = actor.get_actions(torch_obs, greedy=True)

        # Single environment step
        obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, policy_step)
    env.close()


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
    from sheeprl.algos.sac.sac import SAC
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    return log_trained_models(
        fabric, env, cfg, state, SAC(fabric, cfg.to_log), lambda trained: {"agent": trained.agent}
    )
