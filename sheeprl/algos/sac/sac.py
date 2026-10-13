"""Soft Actor-Critic (https://arxiv.org/abs/1812.05905), written on the shared training loop of `sheeprl.core`:
`SAC` says how to build, play and train; `sheeprl.core.loop.run` does the rest."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer

from sheeprl.algos.sac.agent import SACAgent, SACPolicy, build_agent
from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.core import Act, Algorithm, EnvStep, TrainSchedule, TrainState, Writer, run, transition_store, update
from sheeprl.data.store import ReplayStore
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.fabric import autocast_cache_scope
from sheeprl.utils.registry import register_algorithm


def critic_loss_fn(
    agent: SACAgent,
    observations: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_observations: Tensor,
    terminated: Tensor,
    gamma: float,
) -> Tensor:
    """The loss of the critics (Eq. 5), with the targets of the target critics on the next observations."""
    next_target_qf_value = agent.get_next_target_q_values(next_observations, rewards, terminated, gamma)
    qf_values = agent.get_q_values(observations, actions)
    return critic_loss(qf_values, next_target_qf_value, agent.num_critics)


def actor_loss_fn(agent: SACAgent, observations: Tensor) -> Tuple[Tensor, Tensor]:
    """The loss of the actor (Eq. 7) and the log-probabilities of its actions, for the loss of the temperature."""
    actions, logprobs = agent.get_actions_and_log_probs(observations)
    qf_values = agent.get_q_values(observations, actions)
    min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
    return policy_loss(agent.log_alpha.exp().detach(), logprobs, min_qf_values), logprobs.detach()


@dataclass
class SACState(TrainState):
    # Actor, critics, target critics and the entropy coefficient (its logarithm, `log_alpha`); a `DROQAgent` for DroQ
    agent: SACAgent
    qf_optimizer: Optimizer
    actor_optimizer: Optimizer
    alpha_optimizer: Optimizer


class SACWriter(Writer):
    """Writes every step in the replay buffer, with the actions played in the environments.

    With `dtype`, the observations, rewards and episode flags are written with that dtype; otherwise the observations
    and rewards keep the dtype of the environments and the flags are `uint8`.
    """

    def __init__(self, cfg: Dict[str, Any], dtype: Optional[np.dtype] = None) -> None:
        self.cfg = cfg
        self.dtype = dtype
        self.mlp_keys = cfg.algo.mlp_keys.encoder

    def cast(self, value: np.ndarray) -> np.ndarray:
        return value if self.dtype is None else value.astype(self.dtype)

    def write(self, buffer: ReplayStore, step: EnvStep, act: Act) -> None:
        num_envs = len(step.rewards)

        # The observations that follow the actions: for the episodes that have just ended, their last observation,
        # not the first one of the next episode
        next_obs = {k: step.next_obs[k].copy() for k in self.mlp_keys}
        ended_envs = np.nonzero(np.logical_or(step.terminated, step.truncated))[0]
        if len(ended_envs) > 0:
            for k, final_obs in step.stack_final_obs(ended_envs, self.mlp_keys).items():
                next_obs[k][ended_envs] = final_obs

        flags_dtype = np.uint8 if self.dtype is None else self.dtype
        data = {
            "terminated": step.terminated.reshape(1, num_envs, -1).astype(flags_dtype),
            "truncated": step.truncated.reshape(1, num_envs, -1).astype(flags_dtype),
            "actions": act.env_actions.reshape(1, num_envs, -1),
            "observations": self.cast(np.concatenate([step.obs[k] for k in self.mlp_keys], axis=-1))[np.newaxis],
        }
        if not self.cfg.buffer.sample_next_obs:
            next_obs = np.concatenate([next_obs[k] for k in self.mlp_keys], axis=-1).astype(np.float32)
            data["next_observations"] = next_obs[np.newaxis]
        data["rewards"] = self.cast(step.rewards.reshape(num_envs, -1))[np.newaxis]
        buffer.add(data, validate_args=self.cfg.buffer.validate_args)


def train(
    fabric: Fabric,
    agent: SACAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    data: Dict[str, Tensor],
    iteration: int,
    cfg: Dict[str, Any],
    policy_steps_per_iter: int,
):
    metrics: Dict[str, Tensor] = {}
    # The losses are compiled when `algo.compile.enabled` is set
    mark_gradient_step(fabric, cfg)

    # Update the soft-critic
    with autocast_cache_scope(fabric):
        qf_loss = compiled(critic_loss_fn, fabric, cfg)(
            agent,
            data["observations"],
            data["actions"],
            data["rewards"],
            data["next_observations"],
            data["terminated"],
            cfg.algo.gamma,
        )
    update(fabric, qf_loss, qf_optimizer)

    # Update the target networks with EMA
    if iteration % (cfg.algo.critic.target_network_frequency // policy_steps_per_iter + 1) == 0:
        agent.qfs_target_ema()

    # Update the actor
    with autocast_cache_scope(fabric):
        actor_loss, logprobs = compiled(actor_loss_fn, fabric, cfg)(agent, data["observations"])
    update(fabric, actor_loss, actor_optimizer)

    # Update the entropy value
    alpha_loss = entropy_loss(agent.log_alpha, logprobs, agent.target_entropy)
    update(fabric, alpha_loss, alpha_optimizer)

    metrics["Loss/value_loss"] = qf_loss
    metrics["Loss/policy_loss"] = actor_loss
    metrics["Loss/alpha_loss"] = alpha_loss
    return metrics


class SAC(Algorithm):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each on its own batch sampled from the buffer: the critics,
    then the actor, then the entropy coefficient."""

    off_policy = True
    # The name of the algorithm in the error messages
    name = "SAC"
    # The dtype of the values written in the replay buffer (`SACWriter`)
    buffer_dtype: Optional[np.dtype] = None

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        if "minedojo" in cfg.env.wrapper._target_.lower():
            raise ValueError(
                f"MineDojo is not currently supported by {self.name} agent, since it does not take "
                "into consideration the action masks provided by the environment, but needed "
                "in order to play correctly the game. "
                "As an alternative you can use one of the Dreamers' agents."
            )
        if len(cfg.algo.cnn_keys.encoder) > 0:
            warnings.warn(
                f"{self.name} algorithm cannot allow to use images as observations, the CNN keys will be ignored"
            )
            cfg.algo.cnn_keys.encoder = []

    def make_agent(self, obs_space: gym.spaces.Dict, action_space: gym.spaces.Box) -> Tuple[SACAgent, SACPolicy]:
        """The actor, the critics and the entropy coefficient set up on the device, and the policy to play with, which
        shares its weights with the actor."""
        return build_agent(self.fabric, self.cfg, obs_space, action_space)

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[SACState, ReplayStore]:
        cfg = self.cfg
        fabric = self.fabric
        mlp_keys = cfg.algo.mlp_keys.encoder
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError(f"Only continuous action space is supported for the {self.name} agent")
        if not isinstance(obs_space, gym.spaces.Dict):
            raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {obs_space}")
        if len(mlp_keys) == 0:
            raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
        for k in mlp_keys:
            if len(obs_space[k].shape) > 1:
                raise ValueError(
                    f"Only environments with vector-only observations are supported by the {self.name} agent. "
                    f"The observation with key '{k}' has shape {obs_space[k].shape}. "
                    f"Provided environment: {cfg.env.id}"
                )
        if cfg.metric.log_level > 0:
            fabric.print("Encoder MLP keys:", mlp_keys)

        agent, self._policy = self.make_agent(obs_space, action_space)

        qf_optimizer = hydra.utils.instantiate(
            cfg.algo.critic.optimizer, params=agent.qfs.parameters(), _convert_="all"
        )
        actor_optimizer = hydra.utils.instantiate(
            cfg.algo.actor.optimizer, params=agent.actor.parameters(), _convert_="all"
        )
        alpha_optimizer = hydra.utils.instantiate(cfg.algo.alpha.optimizer, params=[agent.log_alpha], _convert_="all")
        qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
            qf_optimizer, actor_optimizer, alpha_optimizer
        )

        state = SACState(
            agent=agent, qf_optimizer=qf_optimizer, actor_optimizer=actor_optimizer, alpha_optimizer=alpha_optimizer
        )
        self.schedule = schedule
        return state, transition_store(fabric, cfg, log_dir)

    def policy(self, state: SACState) -> SACPolicy:
        """The policy to play with: it shares its weights with the trained actor (`build_agent`)."""
        return self._policy

    def writer(self, state: SACState, policy: SACPolicy) -> SACWriter:
        return SACWriter(self.cfg, dtype=self.buffer_dtype)

    def batches(
        self, state: SACState, buffer: ReplayStore, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        # `train` updates the target critics in one iteration out of
        # `target_network_frequency // policy_steps_per_iter + 1`
        self.iteration = iteration
        # The batches of all the gradient steps, sampled at once
        yield from buffer.batches(n_steps, self.cfg.algo.per_rank_batch_size)

    def train_step(self, state: SACState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        metrics = train(
            self.fabric,
            state.agent,
            state.actor_optimizer,
            state.qf_optimizer,
            state.alpha_optimizer,
            batch,
            self.iteration,
            self.cfg,
            self.schedule.policy_steps_per_iter,
        )
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = SAC(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        algo.test(state, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
