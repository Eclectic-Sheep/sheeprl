from typing import Any, Dict

import torch
from lightning import Fabric

from sheeprl.algos.sac.agent import SACActor
from sheeprl.utils.env import make_env

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/alpha_loss",
}


@torch.no_grad()
def test(actor: SACActor, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    actor.eval()
    done = False
    cumulative_rew = 0
    with fabric.device:
        o = env.reset(seed=cfg.seed)[0]
        next_obs = torch.cat([torch.tensor(o[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1).unsqueeze(
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
            next_obs = torch.cat([torch.tensor(next_obs[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1)

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
