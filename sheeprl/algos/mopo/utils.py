from __future__ import annotations

from typing import Any

import torch
from lightning import Fabric

from sheeprl.algos.sac.agent import SACPlayer
from sheeprl.algos.sac.utils import AGGREGATOR_KEYS as SAC_AGGREGATOR_KEYS
from sheeprl.algos.sac.utils import prepare_obs
from sheeprl.envs.d4rl import D4RLWrapper
from sheeprl.utils.utils import dotdict

AGGREGATOR_KEYS = SAC_AGGREGATOR_KEYS.copy()
AGGREGATOR_KEYS.update({"Loss/validation_world_model", "Loss/train_world_model", "Grads/world_model"})


@torch.inference_mode()
def test(actor: SACPlayer, fabric: Fabric, env: D4RLWrapper, cfg: dotdict[str, Any], epoch: int):
    """Test function for the SAC Player in the D4RL environments."""
    done = False
    cumulative_rew = 0.0
    obs = env.reset(seed=cfg.seed)[0]
    while not done:
        # Act greedly through the environment
        torch_obs = prepare_obs(fabric, obs, mlp_keys=cfg.algo.mlp_keys.encoder)
        action = actor.get_actions(torch_obs, greedy=True)

        # Single environment step
        obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward

    normalized_score = env.get_normalized_score(cumulative_rew)
    fabric.print(f"Test (epoch = {epoch}) - Reward: {cumulative_rew}")
    fabric.print(f"Test (epoch = {epoch}) - Normalized Score: {normalized_score}")
    if cfg.metric.log_level > 0:
        fabric.logger.log_metrics(
            {"Test/cumulative_reward": cumulative_rew, "Test/normalized_score": normalized_score}, epoch
        )
