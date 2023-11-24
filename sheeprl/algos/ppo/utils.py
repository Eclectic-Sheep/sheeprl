from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import mlflow
import numpy as np
import torch
from git import Sequence
from lightning import Fabric
from mlflow.models.model import ModelInfo
from torch import Tensor

from sheeprl.algos.ppo.agent import PPOAgent, build_agent
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import unwrap_fabric

AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/value_loss", "Loss/policy_loss", "Loss/entropy_loss"}
MODELS_TO_REGISTER = {"agent"}


@torch.no_grad()
def test(agent: PPOAgent, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    o = env.reset(seed=cfg.seed)[0]
    obs = {}
    for k in o.keys():
        if k in cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
            if k in cfg.algo.cnn_keys.encoder:
                torch_obs = torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255 - 0.5
            if k in cfg.algo.mlp_keys.encoder:
                torch_obs = torch_obs.float()
            obs[k] = torch_obs

    while not done:
        # Act greedly through the environment
        if agent.is_continuous:
            actions = torch.cat(agent.get_greedy_actions(obs), dim=-1)
        else:
            actions = torch.cat([act.argmax(dim=-1) for act in agent.get_greedy_actions(obs)], dim=-1)

        # Single environment step
        o, reward, done, truncated, _ = env.step(actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        obs = {}
        for k in o.keys():
            if k in cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder:
                torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
                if k in cfg.algo.cnn_keys.encoder:
                    torch_obs = torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255 - 0.5
                if k in cfg.algo.mlp_keys.encoder:
                    torch_obs = torch_obs.float()
                obs[k] = torch_obs

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


def normalize_obs(
    obs: Dict[str, np.ndarray | Tensor], cnn_keys: Sequence[str], obs_keys: Sequence[str]
) -> Dict[str, np.ndarray | Tensor]:
    return {k: obs[k] / 255 - 0.5 if k in cnn_keys else obs[k] for k in obs_keys}


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence[ModelInfo]:
    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    agent = build_agent(fabric, actions_dim, is_continuous, cfg, env.observation_space, state["agent"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["agent"] = mlflow.pytorch.log_model(unwrap_fabric(agent), artifact_path="agent")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
