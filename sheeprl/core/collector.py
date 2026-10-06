"""The collection of the training data, the same for every algorithm: a `Policy` chooses the actions (`Act`), the
`Collector` plays them in the `Environment`, and a `Writer` writes what happened in the store of the algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    from sheeprl.core.cadence import Cadence
    from sheeprl.core.runner import Environment, EnvStep
    from sheeprl.core.schedule import TrainSchedule


@dataclass
class Act:
    """The actions chosen by a policy for the current observations of the environments."""

    # The actions played in the environments, one row per environment
    env_actions: np.ndarray
    # What the writer stores with the step, one row per environment: e.g. the actions as the agent trains on them
    # (one-hot for discrete actions), their log-probabilities, the values of the observations
    columns: Dict[str, np.ndarray] = field(default_factory=dict)
    # Anything else the writer reads, e.g. tensors kept on the device
    extras: Dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    """Chooses the actions to play in the environments, and keeps the state of every environment if it has one (e.g. a
    recurrent state). The policy modules of the algorithms implement it, e.g. `class PPOPolicy(nn.Module, Policy)`."""

    def act(self, obs: Dict[str, np.ndarray]) -> Act:
        """The actions for `obs`, the current observations of the environments (one row per environment)."""
        raise NotImplementedError

    def random(self, env: Environment) -> Act:
        """Uniformly random actions, played before the training starts (`TrainSchedule.warmup`); by default, the ones of
        the environments, without columns."""
        return Act(env.random_actions())

    def reset(self, env_idxes: Optional[Sequence[int]] = None) -> None:
        """Reset the state of the environments `env_idxes` (all of them if `None`), which start new episodes. A policy
        without state does nothing."""


class Writer(Protocol):
    """Writes the steps of the environments in the store of an algorithm (e.g. a rollout or a replay buffer)."""

    def write(self, store: Any, step: EnvStep, act: Act) -> None:
        """Write in `store` the step `step`, played with the actions `act`."""


class Collector:
    """Plays `policy` in `env` and writes every step in `store` with `writer`.

    Until `algo.learning_starts` the actions are random (`Policy.random`), when `random_warmup` is set. The state of
    the policy is reset for the environments that start a new episode: whose episode has ended or which have been
    created again after a crash (`EnvStep.restarted`). Every step counts the policy steps of the schedule and gives the
    ended episodes to the cadence, which logs them.
    """

    def __init__(
        self,
        env: Environment,
        policy: Policy,
        writer: Writer,
        store: Any,
        schedule: TrainSchedule,
        cadence: Cadence,
        random_warmup: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy
        self.writer = writer
        self.store = store
        self.schedule = schedule
        self.cadence = cadence
        self.random_warmup = random_warmup

    def reset(self) -> None:
        """Reset the environments and the state of the policy."""
        self.env.reset()
        self.policy.reset()

    def step(self) -> EnvStep:
        """Play one step in every environment and write it."""
        schedule = self.schedule
        if self.random_warmup and schedule.warmup(schedule.policy_step):
            act = self.policy.random(self.env)
        else:
            act = self.policy.act(self.env.obs)
        step = self.env.step(act.env_actions)
        self.writer.write(self.store, step, act)

        new_episodes = np.logical_or(np.logical_or(step.terminated, step.truncated), step.restarted).nonzero()[0]
        if len(new_episodes) > 0:
            self.policy.reset(new_episodes.tolist())
        schedule.policy_step += schedule.policy_steps_per_step
        self.cadence.accumulate_episodes(step.episodes, schedule.policy_step)
        return step

    def collect(self, steps: int) -> None:
        """Play `steps` steps in every environment and write them."""
        for _ in range(steps):
            self.step()
