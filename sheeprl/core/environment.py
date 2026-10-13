"""The environments of one process: what the collector steps (`Environment`, `EnvStep`) and its gymnasium
implementation (`GymEnvironment`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional, Protocol, Sequence

import gymnasium as gym
import numpy as np
from lightning import Fabric

from sheeprl.envs.normalization import ObsNormalizer, RewardNormalizer
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.utils.env import get_episode_stats, get_vector_env_cls, make_env


@dataclass
class Episode:
    """An episode that has just ended."""

    # The environment that played it
    env_idx: int
    reward: float
    length: int


@dataclass
class EnvStep:
    """What happened in one step of the vectorized environments.

    The core and the writers read only its fields: `info` holds what the environments returned besides them (e.g. the
    `info` of gymnasium), for the code written for those environments.
    """

    # The observations the actions were chosen from
    obs: Dict[str, np.ndarray]
    # The observations returned by the step: for an environment whose episode has just ended, it's the first
    # observation of the next episode (same-step autoreset), the last one is in `final_obs`
    next_obs: Dict[str, np.ndarray]
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    # For every environment, the last observation of the episode that has just ended in it (None if it hasn't ended)
    final_obs: Sequence[Optional[Dict[str, np.ndarray]]]
    # Whether every environment has been created again after a crash in this step (`Algorithm.restart_crashed_envs`):
    # its `next_obs` is the first observation of a new episode. If the episode it was playing hadn't ended
    # (`terminated` and `truncated` are false), the episode has been cut short, without a last observation
    restarted: np.ndarray
    # The episodes that have just ended, logged by the training loop
    episodes: Sequence[Episode] = ()
    info: Dict[str, Any] = field(default_factory=dict)

    def stack_final_obs(self, env_idxes: Sequence[int], keys: Sequence[str]) -> Dict[str, np.ndarray]:
        """The last observations of the episodes that have just ended in the environments `env_idxes`, stacked."""
        return {k: np.stack([self.final_obs[env_idx][k] for env_idx in env_idxes]) for k in keys}


class Environment(Protocol):
    """The vectorized environments of one process, stepped by the collector (`sheeprl.core.collector.Collector`):
    `num_envs` environments, whose current observations are `obs`. `GymEnvironment` implements it with gymnasium
    environments.
    """

    num_envs: int
    observation_space: gym.spaces.Dict
    action_space: gym.Space
    # The current observations, one row per environment
    obs: Dict[str, np.ndarray]

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset every environment and return the first observations."""

    def random_actions(self) -> np.ndarray:
        """Uniformly random actions, one row per environment (e.g. to fill a replay buffer before the training)."""

    def step(self, actions: np.ndarray) -> EnvStep:
        """Play `actions` (one row per environment) and return what happened. An environment whose episode ends
        starts the next one in the same step."""

    def close(self) -> None:
        """Close the environments."""

    def state_dict(self) -> Dict[str, Any]:
        """What the checkpoints save of the environments to resume them (e.g. the statistics of the normalization):
        nothing by default."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore what `state_dict` saved, before the first reset."""

    def sync(self, fabric: Fabric) -> None:
        """Make what the environments of the processes learn the same in every process (e.g. the statistics of the
        normalization), at the end of the rollout of every iteration: nothing by default."""


class GymEnvironment(Environment):
    """The `Environment` of a gymnasium vectorized environment with same-step autoreset (`get_vector_env_cls`),
    created by `from_config`: it seeds the resets with `seed` and translates the `info` of gymnasium into the fields of
    `EnvStep`. The ended episodes are read from the statistics of `RecordEpisodeStatistics`.

    With `obs_normalizer` and `reward_normalizer` (`env.normalization`), the observations of the steps (`obs`,
    `next_obs` and `final_obs`) and their rewards are normalized with running statistics, updated at every step with
    the ones of the environments and merged with the ones of the other processes by `sync`. The ended episodes
    (`episodes`) keep the rewards of the environments.
    """

    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        seed: Optional[int] = None,
        obs_normalizer: Optional[ObsNormalizer] = None,
        reward_normalizer: Optional[RewardNormalizer] = None,
    ) -> None:
        self.envs = envs
        self.seed = seed
        self.num_envs = envs.num_envs
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.obs_normalizer = obs_normalizer
        self.reward_normalizer = reward_normalizer
        self.obs: Dict[str, np.ndarray] = {}

    @classmethod
    def from_config(
        cls, fabric: Fabric, cfg: Dict[str, Any], log_dir: str, restart_on_exception: bool = False
    ) -> GymEnvironment:
        """The `cfg.env.num_envs` environments of the process (`make_env`), seeded differently on every process.

        With `restart_on_exception`, an environment that raises an exception is created again (`RestartOnException`)
        instead of stopping the run: its step then returns the first observation of a new episode, with `restarted`
        set.
        """
        num_envs = cfg.env.num_envs
        rank = fabric.global_rank
        first_seed = cfg.seed + rank * num_envs
        env_fns = [
            make_env(
                cfg,
                first_seed + i,
                rank * num_envs,
                log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(num_envs)
        ]
        if restart_on_exception:
            env_fns = [partial(RestartOnException, env_fn) for env_fn in env_fns]
        envs = get_vector_env_cls(cfg.env.sync_env)(env_fns)
        # Seed the random actions (e.g. those played before the training starts)
        envs.action_space.seed(cfg.seed + rank)
        return cls(
            envs,
            seed=first_seed,
            obs_normalizer=ObsNormalizer.from_config(cfg, envs.single_observation_space),
            reward_normalizer=RewardNormalizer.from_config(cfg, num_envs),
        )

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset every environment with its own seed and return the first observations."""
        obs = self.envs.reset(seed=self.seed)[0]
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize(obs)
        self.obs = obs
        return self.obs

    def random_actions(self) -> np.ndarray:
        """Uniformly random actions, one row per environment (e.g. to fill a replay buffer before the training)."""
        return self.envs.action_space.sample()

    def step(self, actions: np.ndarray) -> EnvStep:
        """Play `actions` (one row per environment) and return what happened."""
        next_obs, rewards, terminated, truncated, info = self.envs.step(actions.reshape(self.envs.action_space.shape))
        # The same-step autoreset and `RestartOnException` report them in `info`
        final_obs = info.get("final_obs", [None] * self.num_envs)
        restarted = info.get("restart_on_exception", np.zeros(self.num_envs, dtype=bool))
        if self.obs_normalizer is not None:
            # The last observations of the ended episodes are normalized, but not added to the statistics
            self.obs_normalizer.update(next_obs)
            next_obs = self.obs_normalizer.normalize(next_obs)
            final_obs = [None if obs is None else self.obs_normalizer.normalize(obs) for obs in final_obs]
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer(rewards, terminated | truncated | restarted)
        step = EnvStep(
            self.obs,
            next_obs,
            rewards,
            terminated,
            truncated,
            final_obs=final_obs,
            restarted=restarted,
            # The statistics of `RecordEpisodeStatistics`, with the rewards of the environments
            episodes=tuple(Episode(*episode) for episode in get_episode_stats(info)),
            info=info,
        )
        self.obs = next_obs
        return step

    def close(self) -> None:
        self.envs.close()

    def state_dict(self) -> Dict[str, Any]:
        """The statistics of the normalization of the observations and of the rewards, if they are normalized."""
        state = {}
        if self.obs_normalizer is not None:
            state["obs_normalizer"] = self.obs_normalizer.state_dict()
        if self.reward_normalizer is not None:
            state["reward_normalizer"] = self.reward_normalizer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        if self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(state["reward_normalizer"])

    def sync(self, fabric: Fabric) -> None:
        """Merge the statistics of the normalization of all the processes: then they are the same in every process."""
        if self.obs_normalizer is not None:
            self.obs_normalizer.sync(fabric)
        if self.reward_normalizer is not None:
            self.reward_normalizer.sync(fabric)
