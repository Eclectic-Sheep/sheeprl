"""DroQ (https://arxiv.org/abs/2110.02034): SAC with dropout and layer normalization in its critics, trained with a
high replay ratio. Written on the shared training loop of `sheeprl.core`, as `SAC` with its own critics and updates."""

from __future__ import annotations

from typing import Any, Dict, Iterator

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor
from torch.utils.data import BatchSampler

from sheeprl.algos.droq.agent import DROQAgent, DROQCritic
from sheeprl.algos.sac.agent import SACActor
from sheeprl.algos.sac.loss import entropy_loss, policy_loss
from sheeprl.algos.sac.sac import SAC, SACState, sample_batches
from sheeprl.algos.sac.utils import test
from sheeprl.core import autocast, run, update
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.registry import register_algorithm


class DroQ(SAC):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each one updating every critic on its own batch, and finally
    updates the actor and the entropy coefficient once, on another batch. Its training state is the one of SAC, with
    the DroQ agent."""

    name = "DroQ"
    buffer_dtype = np.float32

    def make_agent(self, obs_dim: int, act_dim: int, action_space: gym.spaces.Box) -> DROQAgent:
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
            DROQCritic(
                observation_dim=obs_dim + act_dim,
                hidden_size=cfg.algo.critic.hidden_size,
                num_critics=1,
                dropout=cfg.algo.critic.dropout,
            )
            for _ in range(cfg.algo.critic.n)
        ]
        return DROQAgent(
            actor,
            critics,
            target_entropy=-act_dim,
            alpha=cfg.algo.alpha.alpha,
            tau=cfg.algo.tau,
            device=self.fabric.device,
        )

    def batches(
        self, state: SACState, buffer: ReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        batch_size = cfg.algo.per_rank_batch_size
        # The batches of the critics for all the gradient steps, then the one of the actor, sampled before training
        critic_data, critic_sampler = sample_batches(
            self.fabric, cfg, buffer, n_steps * batch_size, sample_next_obs=cfg.buffer.sample_next_obs
        )
        actor_data, actor_sampler = sample_batches(self.fabric, cfg, buffer, batch_size)
        actor_idxes = list(actor_sampler)
        critic_batches = list(BatchSampler(critic_sampler, batch_size=batch_size, drop_last=False))
        for i, batch_idxes in enumerate(critic_batches):
            batch = {k: v[batch_idxes] for k, v in critic_data.items()}
            if i == len(critic_batches) - 1:
                # The actor and the entropy coefficient are updated once, after the last critic update
                batch["actor_observations"] = actor_data["observations"][actor_idxes]
            yield batch

    def train_step(self, state: SACState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        agent = state.agent

        # Critics: each one regresses its Q-values towards the same target with its own optimizer step, then its
        # target critic follows it
        with autocast(self.fabric):
            target_qf_values = agent.get_next_target_q_values(
                batch["next_observations"], batch["rewards"], batch["terminated"], cfg.gamma
            )
        qf_losses = []
        for i, critic in enumerate(agent.qfs):
            with autocast(self.fabric):
                qf_values = agent.get_ith_q_value(batch["observations"], batch["actions"], i)
                qf_loss = F.mse_loss(qf_values, target_qf_values)
            update(self.fabric, qf_loss, state.qf_optimizer, params=critic.parameters())
            agent.qfs_target_ema(critic_idx=i)
            qf_losses.append(qf_loss.detach())
        metrics = {"Loss/value_loss": torch.stack(qf_losses).mean()}
        if "actor_observations" not in batch:
            return metrics

        # Actor: maximize the mean Q-value of its actions (not the smallest, as SAC) plus their entropy
        obs = batch["actor_observations"]
        with autocast(self.fabric):
            actions, logprobs = agent.get_actions_and_log_probs(obs)
            qf_values = agent.get_q_values(obs, actions)
            mean_qf_values = torch.mean(qf_values, dim=-1, keepdim=True)
            actor_loss = policy_loss(agent.alpha, logprobs, mean_qf_values)
        update(self.fabric, actor_loss, state.actor_optimizer)

        # Entropy coefficient: towards the target entropy
        alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
        update(self.fabric, alpha_loss, state.alpha_optimizer)

        metrics["Loss/policy_loss"] = actor_loss.detach()
        metrics["Loss/alpha_loss"] = alpha_loss.detach()
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = DroQ(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
