from __future__ import annotations

from typing import Any, Dict, Sequence

import gymnasium as gym
import mlflow
import torch
from lightning import Fabric
from mlflow.models.model import ModelInfo

from sheeprl.algos.ppo.utils import AGGREGATOR_KEYS as ppo_aggregator_keys
from sheeprl.algos.ppo.utils import MODELS_TO_REGISTER as ppo_models_to_register
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent, build_agent
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import unwrap_fabric

AGGREGATOR_KEYS = ppo_aggregator_keys
MODELS_TO_REGISTER = ppo_models_to_register


@torch.no_grad()
def test(agent: "RecurrentPPOAgent", fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    agent.num_envs = 1
    with fabric.device:
        o = env.reset(seed=cfg.seed)[0]
        next_obs = {
            k: torch.tensor(o[k], dtype=torch.float32, device=fabric.device).view(1, 1, -1, *o[k].shape[-2:]) / 255
            for k in cfg.algo.cnn_keys.encoder
        }
        next_obs.update(
            {
                k: torch.tensor(o[k], dtype=torch.float32, device=fabric.device).view(1, 1, -1)
                for k in cfg.algo.mlp_keys.encoder
            }
        )
        state = (
            torch.zeros(1, 1, agent.rnn_hidden_size, device=fabric.device),
            torch.zeros(1, 1, agent.rnn_hidden_size, device=fabric.device),
        )
        actions = torch.zeros(1, 1, sum(agent.actions_dim), device=fabric.device)
    while not done:
        # Act greedly through the environment
        actions, state = agent.get_greedy_actions(next_obs, state, actions)
        if agent.is_continuous:
            real_actions = torch.cat(actions, -1)
            actions = torch.cat(actions, dim=-1).view(1, 1, -1)
        else:
            real_actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)
            actions = torch.cat([act for act in actions], dim=-1).view(1, 1, -1)

        # Single environment step
        o, reward, done, truncated, info = env.step(real_actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        with fabric.device:
            next_obs = {
                k: torch.as_tensor(o[k], dtype=torch.float32).view(1, 1, -1, *o[k].shape[-2:]) / 255
                for k in cfg.algo.cnn_keys.encoder
            }
            next_obs.update(
                {k: torch.as_tensor(o[k], dtype=torch.float32).view(1, 1, -1) for k in cfg.algo.mlp_keys.encoder}
            )

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


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
