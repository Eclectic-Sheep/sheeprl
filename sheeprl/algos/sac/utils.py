from __future__ import annotations

from typing import Any, Dict, Sequence

import gymnasium as gym
import mlflow
import torch
from lightning import Fabric
from mlflow.models.model import ModelInfo

from sheeprl.algos.sac.agent import SACActor, build_agent
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import unwrap_fabric

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/alpha_loss",
}
MODELS_TO_REGISTER = {"agent"}


@torch.no_grad()
def test(actor: SACActor, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    actor.eval()
    done = False
    cumulative_rew = 0
    with fabric.device:
        o = env.reset(seed=cfg.seed)[0]
        next_obs = torch.cat(
            [torch.tensor(o[k], dtype=torch.float32) for k in cfg.algo.mlp_keys.encoder], dim=-1
        ).unsqueeze(
            0
        )  # [N_envs, N_obs]
    while not done:
        # Act greedly through the environment
        action = actor.get_greedy_actions(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        with fabric.device:
            next_obs = torch.cat(
                [torch.tensor(next_obs[k], dtype=torch.float32) for k in cfg.algo.mlp_keys.encoder], dim=-1
            )

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence[ModelInfo]:
    # Create the models
    agent = build_agent(fabric, cfg, env.observation_space, env.action_space, state["agent"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["agent"] = mlflow.pytorch.log_model(unwrap_fabric(agent), artifact_path="agent")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
