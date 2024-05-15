from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor

from sheeprl.algos.ppo.utils import AGGREGATOR_KEYS as ppo_aggregator_keys
from sheeprl.algos.ppo.utils import MODELS_TO_REGISTER as ppo_models_to_register
from sheeprl.algos.ppo.utils import normalize_obs
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOPlayer, build_agent
from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


AGGREGATOR_KEYS = ppo_aggregator_keys
MODELS_TO_REGISTER = ppo_models_to_register


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    with fabric.device:
        for k, v in obs.items():
            torch_obs[k] = torch.as_tensor(v.copy(), dtype=torch.float32, device=fabric.device)
            if k in cnn_keys:
                torch_obs[k] = torch_obs[k].view(1, num_envs, -1, *v.shape[-2:])
            else:
                torch_obs[k] = torch_obs[k].view(1, num_envs, -1)
    return normalize_obs(torch_obs, cnn_keys, torch_obs.keys())


@torch.no_grad()
def test(agent: "RecurrentPPOPlayer", fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    agent.num_envs = 1
    obs = env.reset(seed=cfg.seed)[0]
    with fabric.device:
        state = (
            torch.zeros(1, 1, agent.rnn_hidden_size, device=fabric.device),
            torch.zeros(1, 1, agent.rnn_hidden_size, device=fabric.device),
        )
        actions = torch.zeros(1, 1, sum(agent.actions_dim), device=fabric.device)
    while not done:
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        # Act greedly through the environment
        actions, state = agent.get_actions(torch_obs, actions, state, greedy=True)
        if agent.actor.is_continuous:
            real_actions = torch.stack(actions, -1)
            actions = torch.cat(actions, dim=-1).view(1, 1, -1)
        else:
            real_actions = torch.stack([act.argmax(dim=-1) for act in actions], dim=-1)
            actions = torch.cat([act for act in actions], dim=-1).view(1, 1, -1)

        # Single environment step
        obs, reward, done, truncated, info = env.step(real_actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

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
