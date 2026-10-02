"""Soft Actor-Critic (https://arxiv.org/abs/1812.05905), written on the shared training loop of `sheeprl.core`:
`SAC` says how to build, play and train; `sheeprl.core.loop.run` does the rest."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from math import prod
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DistributedSampler

from sheeprl.algos.sac.agent import SACActor, SACAgent, SACCritic, SACPlayer
from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.algos.sac.utils import prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.registry import register_algorithm


@dataclass
class SACState(TrainState):
    # Actor, critics, target critics and the entropy coefficient (its logarithm, `log_alpha`); a `DROQAgent` for DroQ
    agent: SACAgent
    qf_optimizer: Optimizer
    actor_optimizer: Optimizer
    alpha_optimizer: Optimizer


class ReplayPlayer:
    """Plays in the environments and writes every step in the replay buffer: random actions until
    `algo.learning_starts`, then actions sampled from the policy.

    With `dtype`, the observations, rewards and episode flags are written with that dtype; otherwise the observations
    and rewards keep the dtype of the environments and the flags are `uint8`.
    """

    def __init__(
        self,
        fabric: Fabric,
        cfg: Dict[str, Any],
        policy: SACPlayer,
        schedule: TrainSchedule,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.policy = policy
        self.schedule = schedule
        self.dtype = dtype
        self.mlp_keys = cfg.algo.mlp_keys.encoder

    def cast(self, value: np.ndarray) -> np.ndarray:
        return value if self.dtype is None else value.astype(self.dtype)

    def step(self, env: EnvRunner, buffer: ReplayBuffer) -> None:
        num_envs = env.num_envs
        if self.schedule.warmup(env.policy_step):
            actions = env.random_actions()
        else:
            obs = prepare_obs(self.fabric, env.obs, mlp_keys=self.mlp_keys, num_envs=num_envs)
            actions = self.policy(obs).cpu().numpy()

        step = env.step(actions)

        # The observations that follow the actions: for the episodes that have just ended, their last observation,
        # not the first one of the next episode
        next_obs = {k: step.next_obs[k].copy() for k in self.mlp_keys}
        ended_envs = np.nonzero(np.logical_or(step.terminated, step.truncated))[0]
        if len(ended_envs) > 0:
            for k, final_obs in step.final_obs(ended_envs, self.mlp_keys).items():
                next_obs[k][ended_envs] = final_obs

        flags_dtype = np.uint8 if self.dtype is None else self.dtype
        data = {
            "terminated": step.terminated.reshape(1, num_envs, -1).astype(flags_dtype),
            "truncated": step.truncated.reshape(1, num_envs, -1).astype(flags_dtype),
            "actions": actions.reshape(1, num_envs, -1),
            "observations": self.cast(np.concatenate([step.obs[k] for k in self.mlp_keys], axis=-1))[np.newaxis],
        }
        if not self.cfg.buffer.sample_next_obs:
            next_obs = np.concatenate([next_obs[k] for k in self.mlp_keys], axis=-1).astype(np.float32)
            data["next_observations"] = next_obs[np.newaxis]
        data["rewards"] = self.cast(step.rewards.reshape(num_envs, -1))[np.newaxis]
        buffer.add(data, validate_args=self.cfg.buffer.validate_args)


def sample_batches(
    fabric: Fabric, cfg: Dict[str, Any], buffer: ReplayBuffer, n_samples: int, sample_next_obs: bool = False
) -> Tuple[Dict[str, Tensor], Iterable[int]]:
    """Sample `n_samples` rows from the buffer of every process and return them with the indices this process trains
    on: all of them with one process; with several, its share of the rows of all the processes."""
    sample = buffer.sample_tensors(
        batch_size=n_samples,
        sample_next_obs=sample_next_obs,
        dtype=None,
        device=fabric.device,
        from_numpy=cfg.buffer.from_numpy,
    )  # [1, N_Samples, ...]
    if fabric.world_size == 1:
        data = {k: v.float().reshape(-1, *v.shape[2:]) for k, v in sample.items()}
        return data, range(n_samples)
    data = fabric.all_gather(sample)  # [World_Size, 1, N_Samples, ...]
    data = {k: v.float().reshape(-1, *sample[k].shape[2:]) for k, v in data.items()}
    sampler = DistributedSampler(
        list(range(n_samples * fabric.world_size)),
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=True,
        seed=cfg.seed,
        drop_last=False,
    )
    return data, sampler


class SAC(Algorithm):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each on its own batch sampled from the buffer: the critics,
    then the actor, then the entropy coefficient."""

    off_policy = True
    # The name of the algorithm in the error messages
    name = "SAC"
    # The dtype of the values written in the replay buffer (`ReplayPlayer`)
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

    def make_agent(self, obs_dim: int, act_dim: int, action_space: gym.spaces.Box) -> SACAgent:
        """The actor, the critics and the entropy coefficient, before they are set up on the device."""
        cfg = self.cfg
        actor = SACActor(
            observation_dim=obs_dim,
            action_dim=act_dim,
            distribution_cfg=cfg.distribution,
            hidden_size=cfg.algo.actor.hidden_size,
            action_low=action_space.low,
            action_high=action_space.high,
        )
        critics = [
            SACCritic(observation_dim=obs_dim + act_dim, hidden_size=cfg.algo.critic.hidden_size, num_critics=1)
            for _ in range(cfg.algo.critic.n)
        ]
        return SACAgent(
            actor,
            critics,
            target_entropy=-act_dim,
            alpha=cfg.algo.alpha.alpha,
            tau=cfg.algo.tau,
            device=self.fabric.device,
        )

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[SACState, ReplayBuffer]:
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

        act_dim = prod(action_space.shape)
        obs_dim = sum(prod(obs_space[k].shape) for k in mlp_keys)
        agent = self.make_agent(obs_dim, act_dim, action_space)
        agent.actor = setup_module(fabric, agent.actor)
        # Setting the critics also creates the target critics, as copies of them
        agent.critics = [setup_module(fabric, critic) for critic in agent.critics]
        agent.qfs_target = nn.ModuleList([setup_module(fabric, target) for target in agent.qfs_target])

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
        buffer = ReplayBuffer(
            cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 1,
            cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            seed=cfg.seed + fabric.global_rank,
        )
        self.schedule = schedule
        self.action_space = action_space
        return state, buffer

    def policy(self, state: SACState) -> SACPlayer:
        """The policy to play with: it shares its modules (and so its weights) with the trained actor."""
        actor = state.agent.actor.module
        # Its modules run in the precision of the run, as the actor does
        fabric = get_single_device_fabric(self.fabric)
        policy = SACPlayer(
            fabric.setup_module(actor.model),
            fabric.setup_module(actor.fc_mean),
            fabric.setup_module(actor.fc_logstd),
            action_low=self.action_space.low,
            action_high=self.action_space.high,
        )
        policy.action_scale = policy.action_scale.to(fabric.device)
        policy.action_bias = policy.action_bias.to(fabric.device)
        return policy

    def player(self, state: SACState) -> ReplayPlayer:
        return ReplayPlayer(self.fabric, self.cfg, self.policy(state), self.schedule, dtype=self.buffer_dtype)

    def batches(
        self, state: SACState, buffer: ReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        fabric = self.fabric
        # The target critics are updated at every gradient step of one iteration out of
        # `target_network_frequency // policy_steps_per_iter + 1`, so how often depends on the number of
        # environments (they should be updated every `target_network_frequency` gradient steps)
        period = cfg.algo.critic.target_network_frequency // self.schedule.policy_steps_per_iter + 1
        self.update_targets = iteration % period == 0

        # Sample the batches of all the gradient steps at once
        data, sampler = sample_batches(
            fabric, cfg, buffer, n_steps * cfg.algo.per_rank_batch_size, sample_next_obs=cfg.buffer.sample_next_obs
        )
        for batch_idxes in BatchSampler(sampler, batch_size=cfg.algo.per_rank_batch_size, drop_last=False):
            yield {k: v[batch_idxes] for k, v in data.items()}

    def train_step(self, state: SACState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        agent = state.agent

        # Critics: regress the soft Q-values towards the one-step target of the target critics
        with autocast(self.fabric):
            target_qf_values = agent.get_next_target_q_values(
                batch["next_observations"], batch["rewards"], batch["terminated"], cfg.gamma
            )
            qf_values = agent.get_q_values(batch["observations"], batch["actions"])
            qf_loss = critic_loss(qf_values, target_qf_values, agent.num_critics)
        update(self.fabric, qf_loss, state.qf_optimizer)
        if self.update_targets:
            agent.qfs_target_ema()

        # Actor: maximize the smallest Q-value of its actions plus their entropy
        with autocast(self.fabric):
            actions, logprobs = agent.get_actions_and_log_probs(batch["observations"])
            qf_values = agent.get_q_values(batch["observations"], actions)
            min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
            actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
        update(self.fabric, actor_loss, state.actor_optimizer)

        # Entropy coefficient: towards the target entropy
        alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
        update(self.fabric, alpha_loss, state.alpha_optimizer)

        return {
            "Loss/value_loss": qf_loss.detach(),
            "Loss/policy_loss": actor_loss.detach(),
            "Loss/alpha_loss": alpha_loss.detach(),
        }


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = SAC(fabric, cfg)
    state, log_dir = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
