from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
from lightning import Fabric
from torch import Tensor

from sheeprl.algos.ppo.agent import PPOPlayer
from sheeprl.utils.env import make_env

AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/value_loss", "Loss/policy_loss"}


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, mlp_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {k: torch.from_numpy(obs[k].copy()).to(fabric.device).float().reshape(num_envs, -1) for k in mlp_keys}
    return torch_obs


@torch.no_grad()
def test(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    obs = env.reset(seed=cfg.seed)[0]

    while not done:
        # Convert observations to tensors
        torch_obs = prepare_obs(fabric, obs, mlp_keys=cfg.algo.mlp_keys.encoder)

        # Act greedly through the environment
        actions = agent.get_actions(torch_obs, greedy=True)
        if agent.actor.is_continuous:
            actions = torch.cat(actions, dim=-1)
        else:
            actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)

        # Single environment step
        obs, reward, done, truncated, _ = env.step(actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
