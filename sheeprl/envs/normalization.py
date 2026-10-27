"""The normalization of the observations and of the rewards of the vectorized environments with running statistics
(`env.normalization`), as `VecNormalize` of Stable-Baselines3: the vector observations are standardized with their
running mean and variance, and the rewards are divided by the running standard deviation of the discounted return.

The statistics are those of the environments of all the processes: `sheeprl.core.environment.GymEnvironment` updates
them at every step with the observations and the rewards of its environments, and merges them with the ones of the
other processes at the end of the rollout of every iteration (`sync`). The checkpoints save them, and the test plays
with the ones of the observations, frozen.

The other normalizations live where they act:

- the actions: `NormalizeAction` (`sheeprl/envs/wrappers.py`, `algo.normalize_actions`) exposes the bounded actions
  as [-1, 1];
- the images: `prepare_obs` and `scale_images` (`sheeprl/utils/obs.py`) scale them to [-0.5, 0.5] as the agents read
  them;
- the vector observations of DreamerV3 and DreamerV3.5: the symlog of their encoders (`symlog`,
  `sheeprl/utils/utils.py`);
- the advantages of the on-policy algorithms: `normalize_tensor` (`sheeprl/utils/utils.py`,
  `algo.normalize_advantages`);
- the returns of DreamerV3 and DreamerV3.5: `Moments` (`sheeprl/algos/dreamer_v3/utils.py`), from their percentiles;
- the layers of the networks: `LayerNorm`, `LayerNormChannelLast` and `LayerNormGRUCell` (`sheeprl/models/models.py`),
  `RMSNorm` (`sheeprl/algos/dreamer_v3_5/agent.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch

from sheeprl.utils.utils import dotdict

if TYPE_CHECKING:
    from lightning import Fabric


class SampleMoments(NamedTuple):
    """The mean, the variance and the number of a set of samples."""

    mean: np.ndarray
    var: np.ndarray
    count: float

    def merge(self, other: SampleMoments) -> SampleMoments:
        """The moments of the union of the two sets of samples
        (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm)."""
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / count
        m2 = self.var * self.count + other.var * other.count + delta**2 * self.count * other.count / count
        return SampleMoments(mean, m2 / count, count)


class RunningMeanStd:
    """The running mean and variance of a stream of batches of arrays of shape `shape`, merged batch by batch.

    The statistics start from a mean of 0 and a variance of 1 with a tiny count (`epsilon`): the first batch replaces
    them almost entirely.

    With several processes, every process adds its own batches, and `sync` merges the ones added by all of them since
    the last sync: the statistics are then the same in every process.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.moments = SampleMoments(np.zeros(shape), np.ones(shape), epsilon)
        # The statistics at the last `sync`, the same in every process, and the ones of the batches added since
        self.synced = self.moments
        self.added = SampleMoments(np.zeros(shape), np.zeros(shape), 0)

    @property
    def mean(self) -> np.ndarray:
        return self.moments.mean

    @property
    def var(self) -> np.ndarray:
        return self.moments.var

    @property
    def count(self) -> float:
        return self.moments.count

    def update(self, x: np.ndarray) -> None:
        """Add the batch `x`, one row per sample."""
        x = np.asarray(x, dtype=np.float64)
        batch = SampleMoments(x.mean(axis=0), x.var(axis=0), x.shape[0])
        self.moments = self.moments.merge(batch)
        self.added = self.added.merge(batch)

    def sync(self, fabric: Fabric) -> None:
        """Merge the batches added by every process since the last sync, in the order of the processes (the same in
        every process, so the statistics are the same, bit for bit). With one process there is nothing to merge."""
        if fabric.world_size == 1:
            return
        shape, size = self.added.mean.shape, self.added.mean.size
        added = np.concatenate([self.added.mean.ravel(), self.added.var.ravel(), [self.added.count]])
        # One row per process
        gathered = fabric.all_gather(torch.as_tensor(added, device=fabric.device)).cpu().numpy()
        moments = self.synced
        for row in gathered:
            mean, var, count = row[:size].reshape(shape), row[size:-1].reshape(shape), float(row[-1])
            moments = moments.merge(SampleMoments(mean, var, count))
        self.moments = self.synced = moments
        self.added = SampleMoments(np.zeros(shape), np.zeros(shape), 0)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.moments = self.synced = SampleMoments(
            np.array(state["mean"], dtype=np.float64), np.array(state["var"], dtype=np.float64), state["count"]
        )
        self.added = SampleMoments(np.zeros_like(self.mean), np.zeros_like(self.mean), 0)


class ObsNormalizer:
    """Standardizes the observations of `keys` with their running mean and variance, then clips them to
    [-`clip`, `clip`]; the other observations are left as they are. The standardized observations are float32.

    `update` adds the observations of a step to the statistics, `normalize` standardizes observations with the current
    ones: they work on the observations of the environments (one row per environment) as well as on the observations
    of a single environment (without the rows), as long as `update` gets rows.
    """

    def __init__(
        self, observation_space: gym.spaces.Dict, keys: Sequence[str], clip: float = 10.0, epsilon: float = 1e-8
    ) -> None:
        self.stats = {k: RunningMeanStd(observation_space[k].shape) for k in keys}
        self.clip = clip
        self.epsilon = epsilon

    @classmethod
    def from_config(cls, cfg: dotdict, observation_space: gym.spaces.Dict) -> Optional[ObsNormalizer]:
        """The normalization of the vector observations the agent encodes (`algo.mlp_keys.encoder`), if
        `env.normalization.obs` is set; `None` otherwise (also for the configurations saved before the option)."""
        normalization = cfg.env.get("normalization") or {}
        if not normalization.get("obs", False):
            return None
        keys = [k for k in cfg.algo.mlp_keys.encoder if k in observation_space.spaces]
        return cls(observation_space, keys, clip=normalization.clip_obs, epsilon=normalization.epsilon)

    def update(self, obs: Dict[str, np.ndarray]) -> None:
        """Add `obs`, one row per environment, to the statistics."""
        for k, stats in self.stats.items():
            stats.update(obs[k])

    def sync(self, fabric: Fabric) -> None:
        """Merge the statistics of all the processes (`RunningMeanStd.sync`)."""
        for stats in self.stats.values():
            stats.sync(fabric)

    def normalize(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """`obs` with the observations of the normalized keys standardized (a new dictionary)."""
        obs = dict(obs)
        for k, stats in self.stats.items():
            normalized = (obs[k] - stats.mean) / np.sqrt(stats.var + self.epsilon)
            obs[k] = np.clip(normalized, -self.clip, self.clip).astype(np.float32)
        return obs

    def state_dict(self) -> Dict[str, Any]:
        return {k: stats.state_dict() for k, stats in self.stats.items()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for k, stats in self.stats.items():
            stats.load_state_dict(state[k])


class RewardNormalizer:
    """Divides the rewards by the running standard deviation of the discounted return of every environment (the
    rewards are not centered: their sign doesn't change), then clips them to [-`clip`, `clip`]."""

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.stats = RunningMeanStd()
        # The discounted return of the episode every environment is playing
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon

    @classmethod
    def from_config(cls, cfg: dotdict, num_envs: int) -> Optional[RewardNormalizer]:
        """The normalization of the rewards of `num_envs` environments with the discount factor of the algorithm
        (`algo.gamma`), if `env.normalization.rewards` is set; `None` otherwise (also for the configurations saved
        before the option)."""
        normalization = cfg.env.get("normalization") or {}
        if not normalization.get("rewards", False):
            return None
        return cls(num_envs, cfg.algo.gamma, clip=normalization.clip_rewards, epsilon=normalization.epsilon)

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Add the rewards of a step to the returns and to the statistics, and return them normalized (with their
        dtype if it's a floating-point one). `dones` marks the environments whose episode has ended in the step: their
        return starts again from 0."""
        self.returns = self.returns * self.gamma + rewards
        self.stats.update(self.returns)
        normalized = np.clip(rewards / np.sqrt(self.stats.var + self.epsilon), -self.clip, self.clip)
        self.returns[dones] = 0.0
        return normalized.astype(np.result_type(rewards.dtype, np.float32))

    def sync(self, fabric: Fabric) -> None:
        """Merge the statistics of all the processes (`RunningMeanStd.sync`); the returns stay the ones of the
        environments of the process."""
        self.stats.sync(fabric)

    def state_dict(self) -> Dict[str, Any]:
        # The returns are not saved: the environments of a resumed run start new episodes
        return self.stats.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.stats.load_state_dict(state)
