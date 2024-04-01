from __future__ import annotations

from typing import Any, Dict

import torch
from lightning import Fabric

from sheeprl.algos.ppo.agent import PPOPlayer
from sheeprl.utils.env import make_env

AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/value_loss", "Loss/policy_loss"}


@torch.no_grad()
def test(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    o = env.reset(seed=cfg.seed)[0]
    obs = {}
    for k in o.keys():
        if k in cfg.algo.mlp_keys.encoder:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
            torch_obs = torch_obs.float()
            obs[k] = torch_obs

    while not done:
        # Act greedly through the environment
        actions = agent.get_actions(obs, greedy=True)
        if agent.actor.is_continuous:
            actions = torch.cat(actions, dim=-1)
        else:
            actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)

        # Single environment step
        o, reward, done, truncated, _ = env.step(actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        obs = {}
        for k in o.keys():
            if k in cfg.algo.mlp_keys.encoder:
                torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
                torch_obs = torch_obs.float()
                obs[k] = torch_obs

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
