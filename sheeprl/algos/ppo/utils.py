from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from git import Sequence
from lightning import Fabric
from torch import Tensor

from sheeprl.algos.ppo.agent import PPOAgent
from sheeprl.utils.env import make_env

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
        if k in cfg.mlp_keys.encoder + cfg.cnn_keys.encoder:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
            if k in cfg.cnn_keys.encoder:
                torch_obs = torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255 - 0.5
            if k in cfg.mlp_keys.encoder:
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
            if k in cfg.mlp_keys.encoder + cfg.cnn_keys.encoder:
                torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
                if k in cfg.cnn_keys.encoder:
                    torch_obs = torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255 - 0.5
                if k in cfg.mlp_keys.encoder:
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
