"""The environments of one process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from lightning import Fabric

from sheeprl.utils.env import get_episode_stats, get_vector_env_cls, make_env
from sheeprl.utils.metric import MetricAggregator


@dataclass
class EnvStep:
    """What happened in one step of the vectorized environments."""

    # The observations the actions were chosen from
    obs: Dict[str, np.ndarray]
    # The observations returned by the step: for an environment whose episode has just ended, it's the first
    # observation of the next episode (gymnasium's same-step autoreset), the last one is in `final_obs`
    next_obs: Dict[str, np.ndarray]
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: Dict[str, Any]

    def final_obs(self, env_idxes: Sequence[int], keys: Sequence[str]) -> Dict[str, np.ndarray]:
        """The last observations of the episodes that have just ended in the environments `env_idxes`."""
        return {k: np.stack([self.info["final_obs"][env_idx][k] for env_idx in env_idxes]) for k in keys}


class EnvRunner:
    """The vectorized environments of one process: creation, seeding, stepping and episode statistics.

    Every process has `cfg.env.num_envs` environments, seeded differently on every process. `obs` holds the current
    observations, and `policy_step` counts the steps played by all the environments of all the processes.
    """

    def __init__(
        self,
        fabric: Fabric,
        cfg: Dict[str, Any],
        log_dir: str,
        aggregator: Optional[MetricAggregator] = None,
        policy_step: int = 0,
    ) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.aggregator = aggregator
        self.num_envs = cfg.env.num_envs
        self.policy_step = policy_step
        rank = fabric.global_rank
        self._first_seed = cfg.seed + rank * self.num_envs
        vector_env_cls = get_vector_env_cls(cfg.env.sync_env)
        self.envs = vector_env_cls(
            [
                make_env(
                    cfg,
                    self._first_seed + i,
                    rank * self.num_envs,
                    log_dir if rank == 0 else None,
                    "train",
                    vector_env_idx=i,
                )
                for i in range(self.num_envs)
            ]
        )
        # Seed the random actions (e.g. those played before the training starts)
        self.envs.action_space.seed(cfg.seed + rank)
        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space
        self.obs: Dict[str, np.ndarray] = {}

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset every environment with its own seed and return the first observations."""
        self.obs = self.envs.reset(seed=self._first_seed)[0]
        return self.obs

    def random_actions(self) -> np.ndarray:
        """Uniformly random actions, one row per environment (e.g. to fill a replay buffer before the training)."""
        return self.envs.action_space.sample()

    def step(self, actions: np.ndarray) -> EnvStep:
        """Play `actions` (one row per environment) and return what happened. Ended episodes are logged."""
        next_obs, rewards, terminated, truncated, info = self.envs.step(actions.reshape(self.envs.action_space.shape))
        self.policy_step += self.num_envs * self.fabric.world_size
        if self.cfg.metric.log_level > 0:
            for i, ep_rew, ep_len in get_episode_stats(info):
                if self.aggregator and "Rewards/rew_avg" in self.aggregator:
                    self.aggregator.update("Rewards/rew_avg", ep_rew)
                if self.aggregator and "Game/ep_len_avg" in self.aggregator:
                    self.aggregator.update("Game/ep_len_avg", ep_len)
                self.fabric.print(f"Rank-0: policy_step={self.policy_step}, reward_env_{i}={ep_rew}")
        step = EnvStep(self.obs, next_obs, rewards, terminated, truncated, info)
        self.obs = next_obs
        return step

    def close(self) -> None:
        self.envs.close()
