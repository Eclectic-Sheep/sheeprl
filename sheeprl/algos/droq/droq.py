"""DroQ (https://arxiv.org/abs/2110.02034): SAC with dropout and layer normalization in its critics, trained with a
high replay ratio. Written on the shared training loop of `sheeprl.core`, as `SAC` with its own critics and updates."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor

from sheeprl.algos.droq.agent import DROQAgent, build_agent
from sheeprl.algos.sac.agent import SACPolicy
from sheeprl.algos.sac.loss import entropy_loss, policy_loss
from sheeprl.algos.sac.sac import SAC, SACState
from sheeprl.core import autocast, run, update
from sheeprl.data.samplers import TransitionSampler
from sheeprl.data.store import ReplayStore
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.registry import register_algorithm


def next_target_fn(
    agent: DROQAgent, next_observations: Tensor, rewards: Tensor, terminated: Tensor, gamma: float
) -> Tensor:
    """The targets of the critics, from the target critics on the next observations (Line 7 - Algorithm 2)."""
    return agent.get_next_target_q_values(next_observations, rewards, terminated, gamma)


def critic_loss_fn(agent: DROQAgent, observations: Tensor, actions: Tensor, targets: Tensor, critic_idx: int) -> Tensor:
    """The loss of the critic `critic_idx` (Line 8 - Algorithm 2)."""
    return F.mse_loss(agent.get_ith_q_value(observations, actions, critic_idx), targets)


def actor_loss_fn(agent: DROQAgent, observations: Tensor) -> Tuple[Tensor, Tensor]:
    """The loss of the actor, on the mean of the critics, and the log-probabilities of its actions, for the loss of the
    temperature (Line 10 - Algorithm 2)."""
    actions, logprobs = agent.get_actions_and_log_probs(observations)
    qf_values = agent.get_q_values(observations, actions)
    mean_qf_values = torch.mean(qf_values, dim=-1, keepdim=True)
    return policy_loss(agent.log_alpha.exp().detach(), logprobs, mean_qf_values), logprobs.detach()


class DroQ(SAC):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each one updating every critic on its own batch, and finally
    updates the actor and the entropy coefficient once, on another batch. Its training state is the one of SAC, with
    the DroQ agent."""

    name = "DroQ"
    buffer_dtype = np.float32

    def make_agent(self, obs_space: gym.spaces.Dict, action_space: gym.spaces.Box) -> Tuple[DROQAgent, SACPolicy]:
        return build_agent(self.fabric, self.cfg, obs_space, action_space)

    def batches(
        self, state: SACState, buffer: ReplayStore, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        batch_size = self.cfg.algo.per_rank_batch_size
        # The batches of the critics for all the gradient steps, then the one of the actor, sampled before training.
        # The new transitions of the online queue (`buffer.online`) go to the critics: the batch of the actor is sampled
        # uniformly, without the next observations, by a sampler that shares the generator of the critics' one
        actor_sampler = TransitionSampler(rng=buffer.sampler.rng, queue=buffer.sampler.queue)
        for i, batch in enumerate(buffer.batches(n_steps, batch_size)):
            if i == 0:
                # After the batches of the critics, which `batches` samples (or prefetched) all at once
                actor_observations = buffer.sample(batch_size, sampler=actor_sampler)["observations"][0].float()
            if i == n_steps - 1:
                # The actor and the entropy coefficient are updated once, after the last critic update
                batch["actor_observations"] = actor_observations
            yield batch

    def train_step(self, state: SACState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        fabric, cfg = self.fabric, self.cfg
        agent = state.agent
        # The losses are compiled when `algo.compile.enabled` is set
        mark_gradient_step(fabric, cfg)

        # Critics: each one regresses its Q-values towards the same target with its own optimizer step, then its
        # target critic follows it
        target_qf_values = compiled(next_target_fn, fabric, cfg)(
            agent, batch["next_observations"], batch["rewards"], batch["terminated"], cfg.algo.gamma
        )
        qf_losses = []
        for i in range(agent.num_critics):
            with autocast(fabric):
                qf_loss = compiled(critic_loss_fn, fabric, cfg)(
                    agent, batch["observations"], batch["actions"], target_qf_values, i
                )
            update(fabric, qf_loss, state.qf_optimizer, params=agent.qfs[i].parameters())
            agent.qfs_target_ema(critic_idx=i)
            qf_losses.append(qf_loss.detach())
        metrics = {"Loss/value_loss": torch.stack(qf_losses).mean()}
        if "actor_observations" not in batch:
            return metrics

        # Actor: maximize the mean Q-value of its actions (not the smallest, as SAC) plus their entropy
        mark_gradient_step(fabric, cfg)
        with autocast(fabric):
            actor_loss, logprobs = compiled(actor_loss_fn, fabric, cfg)(agent, batch["actor_observations"])
        update(fabric, actor_loss, state.actor_optimizer)

        # Entropy coefficient: towards the target entropy
        alpha_loss = entropy_loss(agent.log_alpha, logprobs, agent.target_entropy)
        update(fabric, alpha_loss, state.alpha_optimizer)

        metrics["Loss/policy_loss"] = actor_loss.detach()
        metrics["Loss/alpha_loss"] = alpha_loss.detach()
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = DroQ(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        algo.test(state, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
