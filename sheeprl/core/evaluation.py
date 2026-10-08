"""Evaluation and registration of the trained models, the same for every algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import nn

from sheeprl.core.algorithm import Algorithm, TrainState
from sheeprl.core.loop import load_trained_state
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

    from sheeprl.core.collector import Policy


@torch.no_grad()
def run_test(
    policy: Policy,
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    policy_step: int = 0,
    greedy: bool = True,
    test_name: str = "",
) -> float:
    """Play one episode of a test environment with `policy` and log its return (`Test/cumulative_reward`) at
    `policy_step`: the test of every algorithm (`Algorithm.test`). With `greedy`, the policy plays its most likely
    actions. A dry run plays one step. Returns the return of the episode."""
    env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name else ""), vector_env_idx=0)()
    training = isinstance(policy, nn.Module) and policy.training
    if isinstance(policy, nn.Module):
        policy.eval()
    policy.init_states(1)
    obs, _ = env.reset(seed=cfg.seed)
    done = False
    cumulative_rew = 0
    while not done:
        # The policy plays one environment: its observations are a row
        act = policy.act({k: v[np.newaxis] for k, v in obs.items()}, greedy=greedy)
        obs, reward, terminated, truncated, _ = env.step(act.env_actions.reshape(env.action_space.shape))
        done = terminated or truncated or cfg.dry_run
        cumulative_rew += reward
    env.close()
    if training:
        policy.train()
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0 and len(fabric.loggers) > 0:
        fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, policy_step)
    return cumulative_rew


def evaluate(fabric: Fabric, cfg: Dict[str, Any], checkpoint: Dict[str, Any], algo: Algorithm) -> None:
    """Test the policy of `algo` trained in `checkpoint` (`Algorithm.test`): the evaluation of `sheeprl-eval`."""
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name, log_root=cfg.log_root)
    fabric.print(f"Log dir: {log_dir}")

    # The spaces of the environment, to build the models
    env = make_env(cfg, cfg.seed, 0, log_dir, "test", vector_env_idx=0)()
    trained = load_trained_state(fabric, cfg, algo, checkpoint, env.observation_space, env.action_space)
    env.close()
    algo.test(trained, log_dir)


def log_models_from_checkpoint(
    fabric: Fabric,
    env: gym.Env | gym.Wrapper,
    cfg: Dict[str, Any],
    checkpoint: Dict[str, Any],
    algo: Algorithm,
    models: Callable[[TrainState], Dict[str, Any]],
) -> Dict[str, "ModelInfo"]:
    """Log in MLflow the `models` of the training state of `algo` restored from `checkpoint`, by name, and the
    configuration of the training (`cfg.to_log`), in the run `cfg.run.id` (a new one if `None`)."""
    import mlflow

    trained = load_trained_state(fabric, cfg.to_log, algo, checkpoint, env.observation_space, env.action_space)
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        for name, model in models(trained).items():
            model_info[name] = mlflow.pytorch.log_model(unwrap_fabric(model), name=name, serialization_format="pickle")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
