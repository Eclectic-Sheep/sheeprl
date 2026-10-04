"""Dreamer-V3 implementation from [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
Adapted from the original implementation from https://github.com/danijar/dreamerv3

Written on the shared training loop of `sheeprl.core`: `DreamerV3` says how to build, play and train;
`sheeprl.core.loop.run` does the rest. Plan2Explore (`sheeprl.algos.p2e_dv3`) reuses the player
(`SequencePlayer`) and the two phases of a gradient step (`world_model_learning`, `behaviour_learning`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, OneHotCategorical
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.utils import actor_objective, env_buffer_size, reinforce_weight, sample_batches
from sheeprl.algos.dreamer_v3.agent import Actor, MinedojoActor, PlayerDV3, WorldModel, build_agent, clip_actions
from sheeprl.algos.dreamer_v3.loss import reconstruction_loss
from sheeprl.algos.dreamer_v3.utils import Moments, compute_lambda_values, prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, TrainSchedule, TrainState, run
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.distribution import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    TwoHotEncodingDistribution,
)
from sheeprl.utils.distribution import entropy as policy_entropy
from sheeprl.utils.fabric import autocast_cache_scope, update
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"


@dataclass
class DreamerV3State(TrainState):
    # Encoder, RSSM, decoder, reward and continue models
    world_model: WorldModel
    actor: Actor | MinedojoActor
    critic: nn.Module
    # Slow copy of the critic, the regularizer of its targets
    target_critic: nn.Module
    world_optimizer: Optimizer
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer
    # Percentiles of the lambda-values, which normalize the advantages of the actor
    moments: Moments


class SequencePlayer:
    """Plays in the environments and writes in the replay buffer the sequences the world model learns from.

    Every row holds an observation, the action played from it, and the reward, `terminated`, `truncated` and
    `is_first` of the step that led to it. When an episode ends, a last row holds its final observation (with a zero
    action) and the next row is the first one of the new episode.

    With `random_warmup`, the actions are uniformly random until `algo.learning_starts`; otherwise they come from
    `policy` (`PlayerDV3`), whose recurrent state is reset at the start of every episode.
    """

    def __init__(
        self,
        fabric: Fabric,
        cfg: Dict[str, Any],
        policy: PlayerDV3,
        schedule: TrainSchedule,
        actions_dim: Sequence[int],
        is_continuous: bool,
        random_warmup: bool,
    ) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.policy = policy
        self.schedule = schedule
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.random_warmup = random_warmup
        self.obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
        # The row written at the next step; created from the first observations of the environments
        self.step_data: Optional[Dict[str, np.ndarray]] = None

    def step(self, env: EnvRunner, buffer: EnvIndependentReplayBuffer) -> None:
        cfg = self.cfg
        num_envs = env.num_envs
        if self.step_data is None:
            # The first observations start the episodes
            self.step_data = {k: env.obs[k][np.newaxis] for k in self.obs_keys}
            self.step_data["rewards"] = np.zeros((1, num_envs, 1))
            self.step_data["truncated"] = np.zeros((1, num_envs, 1))
            self.step_data["terminated"] = np.zeros((1, num_envs, 1))
            self.step_data["is_first"] = np.ones_like(self.step_data["terminated"])
            self.policy.init_states()
        step_data = self.step_data

        # The actions are stored one-hot for discrete actions, while the environments take their indices
        if self.random_warmup and self.schedule.warmup(env.policy_step):
            real_actions = actions = np.array(env.random_actions())
            if not self.is_continuous:
                # One row per environment, one column per discrete action: one-hot each column
                per_action = actions.reshape(num_envs, len(self.actions_dim)).T
                actions = np.concatenate(
                    [
                        F.one_hot(torch.as_tensor(act), act_dim).numpy()
                        for act, act_dim in zip(per_action, self.actions_dim)
                    ],
                    axis=-1,
                )
        else:
            torch_obs = prepare_obs(self.fabric, env.obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=num_envs)
            mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
            real_actions = actions = self.policy.get_actions(torch_obs, mask=mask if len(mask) > 0 else None)
            actions = torch.cat(actions, -1).cpu().numpy()
            if self.is_continuous:
                real_actions = torch.stack(real_actions, dim=-1).cpu().numpy()
            else:
                real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

        step_data["actions"] = actions.reshape((1, num_envs, -1))
        buffer.add(step_data, validate_args=cfg.buffer.validate_args)

        step = env.step(real_actions)
        dones = np.logical_or(step.terminated, step.truncated).astype(np.uint8)

        step_data["is_first"] = np.zeros_like(step_data["terminated"])
        if "restart_on_exception" in step.info:
            restarted_envs = []
            for i, agent_roe in enumerate(step.info["restart_on_exception"]):
                if agent_roe and not dones[i]:
                    # The last observation stored for the restarted environment ends its episode
                    last_inserted_idx = (buffer.buffer[i]._pos - 1) % buffer.buffer[i].buffer_size
                    buffer.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                        buffer.buffer[i]["terminated"][last_inserted_idx]
                    )
                    buffer.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                        buffer.buffer[i]["truncated"][last_inserted_idx]
                    )
                    # The observation returned after the restart starts a new episode
                    step_data["is_first"][:, i] = np.ones_like(step_data["is_first"][:, i])
                    restarted_envs.append(i)
            if len(restarted_envs) > 0:
                self.policy.init_states(restarted_envs)

        for k in self.obs_keys:
            step_data[k] = step.next_obs[k][np.newaxis]
        rewards = step.rewards.reshape((1, num_envs, -1))
        step_data["terminated"] = step.terminated.reshape((1, num_envs, -1))
        step_data["truncated"] = step.truncated.reshape((1, num_envs, -1))
        step_data["rewards"] = np.tanh(rewards) if cfg.env.clip_rewards else rewards

        # The episodes that have just ended get a last row with their final observation; the next row, the first
        # observation of the new episode, gets zero reward and `is_first`
        dones_idxes = dones.nonzero()[0].tolist()
        reset_envs = len(dones_idxes)
        if reset_envs > 0:
            final_obs = step.final_obs(dones_idxes, self.obs_keys)
            reset_data = {k: final_obs[k].astype(step.next_obs[k].dtype, copy=False)[np.newaxis] for k in self.obs_keys}
            reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
            reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
            reset_data["actions"] = np.zeros((1, reset_envs, np.sum(self.actions_dim)))
            reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
            reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
            buffer.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

            step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
            step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
            step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
            step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
            self.policy.init_states(dones_idxes)


def world_model_loss_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The configuration of `world_model_loss`, as plain values."""
    world_model_cfg = cfg.algo.world_model
    return {
        "cnn_keys": tuple(cfg.algo.cnn_keys.encoder),
        "mlp_keys": tuple(cfg.algo.mlp_keys.encoder),
        "cnn_decoder_keys": tuple(cfg.algo.cnn_keys.decoder),
        "mlp_decoder_keys": tuple(cfg.algo.mlp_keys.decoder),
        "stochastic_size": world_model_cfg.stochastic_size,
        "discrete_size": world_model_cfg.discrete_size,
        "decoupled_rssm": bool(world_model_cfg.decoupled_rssm),
        "kl_dynamic": world_model_cfg.kl_dynamic,
        "kl_representation": world_model_cfg.kl_representation,
        "kl_free_nats": world_model_cfg.kl_free_nats,
        "kl_regularizer": world_model_cfg.kl_regularizer,
        "continue_scale_factor": world_model_cfg.continue_scale_factor,
    }


def world_model_loss(
    world_model: WorldModel,
    data: Dict[str, Tensor],
    *,
    cnn_keys: Sequence[str],
    mlp_keys: Sequence[str],
    cnn_decoder_keys: Sequence[str],
    mlp_decoder_keys: Sequence[str],
    stochastic_size: int,
    discrete_size: int,
    decoupled_rssm: bool,
    kl_dynamic: float,
    kl_representation: float,
    kl_free_nats: float,
    kl_regularizer: float,
    continue_scale_factor: float,
    detach_heads: bool = False,
    entropies: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """The loss of the world model on a batch of sequences (Eq. 4 in the paper), without the optimization step: it
    can be compiled (`algo.compile`). The configuration is given by `world_model_loss_kwargs`.

    Args:
        detach_heads: the reward and continue models learn from the latent states without changing them (P2E).
        entropies: whether to compute the entropies of the posteriors and of the priors (metrics).

    Returns:
        The loss, the posteriors and the recurrent states of the batch (the starting points of the imagination), and
        the metrics.
    """
    # The environment interaction goes like this:
    # Actions:           a0       a1       a2      a4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     v  /     v  /     v /
    # Observations:  o0       o1       o2       o3
    # Rewards:       0        r1       r2       r3
    # Dones:         0        d1       d2       d3
    # Is-first       1        i1       i2       i3
    sequence_length, batch_size = data["actions"].shape[:2]
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cnn_keys}
    batch_obs.update({k: data[k] for k in mlp_keys})

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Embed observations from the environment
    embedded_obs = world_model.encoder(batch_obs)

    # The outputs of every step are concatenated at the end of the unroll: writing them in place into
    # preallocated tensors makes the backward pass copy the gradient of the whole tensor at every step
    recurrent_states = []
    # The initial states, where the episodes start, are the same at every step
    initial_states = world_model.rssm.get_initial_states((1, batch_size))
    recurrent_state = torch.zeros_like(initial_states[0])
    if decoupled_rssm:
        posteriors_logits, posteriors = world_model.rssm._representation(embedded_obs)
        for i in range(0, sequence_length):
            if i == 0:
                posterior = torch.zeros_like(posteriors[:1])
            else:
                posterior = posteriors[i - 1 : i]
            recurrent_state = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                data["is_first"][i : i + 1],
                initial_states,
            )
            recurrent_states.append(recurrent_state)
    else:
        posterior = torch.zeros_like(initial_states[1])
        posteriors, posteriors_logits = [], []
        # The part of the representation model that depends on the observations, for the whole sequence at once
        observations_projection = world_model.rssm.project_observations(embedded_obs)
        for i in range(0, sequence_length):
            recurrent_state, posterior, posterior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                observations_projection[i : i + 1],
                data["is_first"][i : i + 1],
                initial_states,
                projected=True,
            )
            recurrent_states.append(recurrent_state)
            posteriors.append(posterior)
            posteriors_logits.append(posterior_logits)
        posteriors = torch.cat(posteriors, dim=0)
        posteriors_logits = torch.cat(posteriors_logits, dim=0)
    recurrent_states = torch.cat(recurrent_states, dim=0)
    # The priors don't take part in the recurrence: computed for the whole sequence at once
    priors_logits = world_model.rssm.prior_logits(recurrent_states)
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

    # Compute predictions for the observations
    reconstructed_obs: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)

    # Compute the distribution over the reconstructed observations
    po = {k: MSEDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:])) for k in cnn_decoder_keys}
    po.update(
        {
            k: SymlogDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
            for k in mlp_decoder_keys
        }
    )

    # Compute the distributions over the rewards and over the terminal steps
    heads_input = latent_states.detach() if detach_heads else latent_states
    pr = TwoHotEncodingDistribution(world_model.reward_model(heads_input), dims=1)
    pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(heads_input)), 1)
    continues_targets = 1 - data["terminated"]

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

    # World model optimization step. Eq. 4 in the paper
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        po,
        batch_obs,
        pr,
        data["rewards"],
        priors_logits,
        posteriors_logits,
        kl_dynamic,
        kl_representation,
        kl_free_nats,
        kl_regularizer,
        pc,
        continues_targets,
        continue_scale_factor,
    )
    metrics = {
        "Loss/world_model_loss": rec_loss.detach(),
        "Loss/observation_loss": observation_loss.detach(),
        "Loss/reward_loss": reward_loss.detach(),
        "Loss/state_loss": state_loss.detach(),
        "Loss/continue_loss": continue_loss.detach(),
        "State/kl": kl.mean().detach(),
    }
    if entropies:
        metrics["State/post_entropy"] = (
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach()
        )
        metrics["State/prior_entropy"] = (
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach()
        )
    return rec_loss, posteriors, recurrent_states, metrics


def world_model_learning(
    fabric: Fabric,
    cfg: Dict[str, Any],
    world_model: WorldModel,
    world_optimizer: Optimizer,
    data: Dict[str, Tensor],
    detach_heads: bool = False,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """One update of the world model on a batch of sequences (dynamic learning, Eq. 4 in the paper).

    Args:
        detach_heads: the reward and continue models learn from the latent states without changing them (P2E).

    Returns:
        The posteriors and the recurrent states of the batch, the starting points of the imagination, and the metrics
        (with the norm of the gradients before clipping, `Grads/world_model`, when they are clipped).
    """
    # Every sequence starts an episode: the world model starts from its initial state
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])
    mark_gradient_step(fabric, cfg)
    # Cast the weights to low precision once for the whole forward pass, not at every step of the unroll
    with autocast_cache_scope(fabric):
        rec_loss, posteriors, recurrent_states, metrics = compiled(world_model_loss, fabric, cfg)(
            world_model,
            data,
            **world_model_loss_kwargs(cfg),
            detach_heads=detach_heads,
            entropies=not MetricAggregator.disabled,
        )
    # The gradients of all the weights of the world model are averaged over the processes, also the ones of the
    # learnable initial recurrent state, which is in no module
    world_model_grads = update(
        fabric, rec_loss, world_optimizer, cfg.algo.world_model.clip_gradients, error_if_nonfinite=False
    )
    if world_model_grads:
        metrics["Grads/world_model"] = world_model_grads.mean().detach()
    return posteriors, recurrent_states, metrics


def imagine(
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    posteriors: Tensor,
    recurrent_states: Tensor,
    terminated: Tensor,
    *,
    horizon: int,
    gamma: float,
    lmbda: float,
    action_clip: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Imagine `horizon` steps from every latent state of the batch, with the actions of the actor, and estimate
    their lambda-values. Can be compiled (`algo.compile`).

    Args:
        action_clip: the magnitude the recurrent model clips the continuous actions to (`clip_actions`), 0 for the
            discrete ones. The returned actions are the samples of the actor, as DreamerV3 clips them in its RSSM:
            REINFORCE takes their log-probabilities.

    Returns:
        The imagined latent states and actions, the values predicted by the critic, the lambda-values and the
        discounts of the imagined steps.
    """
    stoch_state_size = posteriors.shape[-2] * posteriors.shape[-1]
    recurrent_state_size = recurrent_states.shape[-1]
    imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    actions = torch.cat(actor(imagined_latent_state.detach(), clip=False)[0], dim=-1)
    imagined_trajectories = [imagined_latent_state]
    imagined_actions = [actions]

    # The imagination goes like this, with H=3:
    # Actions:           a'0      a'1      a'2     a'4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:        z0 ---> z'1 ---> z'2 ---> z'3
    # Rewards:       r'0     r'1      r'2      r'3
    # Values:        v'0     v'1      v'2      v'3
    # Lambda-values:         l'1      l'2      l'3
    # Continues:     c0      c'1      c'2      c'3
    # where z0 comes from the posterior, while z'i is the imagined states (prior)

    # Imagine trajectories in the latent space
    for i in range(1, horizon + 1):
        imagined_prior, recurrent_state = world_model.rssm.imagination(
            imagined_prior, recurrent_state, clip_actions(actions, action_clip)
        )
        imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        actions = torch.cat(actor(imagined_latent_state.detach(), clip=False)[0], dim=-1)
        imagined_trajectories.append(imagined_latent_state)
        imagined_actions.append(actions)
    imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
    imagined_actions = torch.cat(imagined_actions, dim=0)

    # Predict values, rewards and continues
    predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean
    predicted_rewards = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean
    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
    true_continue = (1 - terminated).flatten().reshape(1, -1, 1)
    continues = torch.cat((true_continue, continues[1:]))

    # Estimate lambda-values
    lambda_values = compute_lambda_values(
        predicted_rewards[1:], predicted_values[1:], continues[1:] * gamma, lmbda=lmbda
    )

    # Compute the discounts to multiply the lambda values to
    discount = (torch.cumprod(continues * gamma, dim=0) / gamma).detach()
    return imagined_trajectories, imagined_actions, predicted_values, lambda_values, discount


def actor_loss(
    actor: nn.Module,
    imagined_trajectories: Tensor,
    imagined_actions: Tensor,
    predicted_values: Tensor,
    lambda_values: Tensor,
    discount: Tensor,
    offset: Tensor,
    invscale: Tensor,
    *,
    objective_mix: Optional[float],
    is_continuous: bool,
    actions_dim: Sequence[int],
    ent_coef: float,
) -> Tensor:
    """The loss of the actor (Eq. 11 in the paper), from the imagined trajectories and the normalization of the
    returns (`offset`, `invscale`): the dynamics backpropagation of the advantages and REINFORCE, mixed by
    `objective_mix` (`actor_objective`), with the entropy of the policies. Can be compiled (`algo.compile`)."""
    # Given the following diagram, with H=3
    # Actions:          [a'0]    [a'1]    [a'2]    a'3
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:       [z0] -> [z'1] -> [z'2] ->  z'3
    # Values:       [v'0]   [v'1]    [v'2]     v'3
    # Lambda-values:        [l'1]    [l'2]    [l'3]
    # Entropies:    [e'0]   [e'1]    [e'2]
    policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]

    baseline = predicted_values[:-1]
    normed_lambda_values = (lambda_values - offset) / invscale
    normed_baseline = (baseline - offset) / invscale
    advantage = normed_lambda_values - normed_baseline

    def reinforce() -> Tensor:
        return (
            torch.stack(
                [
                    p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                    for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                ],
                dim=-1,
            ).sum(dim=-1)
            * advantage.detach()
        )

    objective = actor_objective(objective_mix, is_continuous, advantage, reinforce)
    # The tanh-normal policies have no analytic entropy: it is estimated from samples
    entropy = ent_coef * torch.stack([policy_entropy(p) for p in policies], -1).sum(dim=-1)
    return -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))


def critic_loss(
    critic: nn.Module, target_critic: nn.Module, imagined_trajectories: Tensor, lambda_values: Tensor, discount: Tensor
) -> Tensor:
    """The loss of the critic (Eq. 10 in the paper): the lambda-values and the values of the target critic as
    targets. Can be compiled (`algo.compile`)."""
    qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
    predicted_target_values = TwoHotEncodingDistribution(
        target_critic(imagined_trajectories.detach()[:-1]), dims=1
    ).mean
    value_loss = -qv.log_prob(lambda_values.detach())
    value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    return torch.mean(value_loss * discount[:-1].squeeze(-1))


def behaviour_learning(
    fabric: Fabric,
    cfg: Dict[str, Any],
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    moments: Moments,
    posteriors: Tensor,
    recurrent_states: Tensor,
    terminated: Tensor,
    is_continuous: bool,
    actions_dim: Sequence[int],
) -> Dict[str, Tensor]:
    """One update of the actor and one of the critic, on trajectories imagined from the latent states of the batch
    (behaviour learning, Eq. 10 and 11 in the paper).

    Returns:
        The losses (`policy_loss`, `value_loss`) and, when they are clipped, the norms of the gradients before
        clipping (`actor_grads`, `critic_grads`).
    """
    metrics = {}
    # The actor learns by REINFORCE, from the imagined actions and the lambda-values without their gradients, and by
    # the dynamics backpropagation of the lambda-values, mixed by `algo.actor.objective_mix` (by default the dynamics
    # for the continuous actions, REINFORCE for the discrete ones): the imagination needs a computational graph only
    # for the dynamics backpropagation
    objective_mix = cfg.algo.actor.objective_mix
    with autocast_cache_scope(fabric), torch.set_grad_enabled(reinforce_weight(objective_mix, is_continuous) < 1):
        imagined_trajectories, imagined_actions, predicted_values, lambda_values, discount = compiled(
            imagine, fabric, cfg
        )(
            world_model,
            actor,
            critic,
            posteriors,
            recurrent_states,
            terminated,
            horizon=cfg.algo.horizon,
            gamma=cfg.algo.gamma,
            lmbda=cfg.algo.lmbda,
            action_clip=float(cfg.algo.actor.action_clip) if is_continuous else 0.0,
        )
    with autocast_cache_scope(fabric):
        # The normalization of the returns, from their percentiles (not compiled: it updates its state in place)
        offset, invscale = moments(lambda_values, fabric)
        policy_loss = compiled(actor_loss, fabric, cfg)(
            actor,
            imagined_trajectories,
            imagined_actions,
            predicted_values,
            lambda_values,
            discount,
            offset,
            invscale,
            objective_mix=objective_mix,
            is_continuous=is_continuous,
            actions_dim=tuple(int(dim) for dim in actions_dim),
            ent_coef=cfg.algo.actor.ent_coef,
        )
    actor_grads = update(fabric, policy_loss, actor_optimizer, cfg.algo.actor.clip_gradients, error_if_nonfinite=False)
    if actor_grads:
        metrics["actor_grads"] = actor_grads.mean().detach()

    with autocast_cache_scope(fabric):
        value_loss = compiled(critic_loss, fabric, cfg)(
            critic, target_critic, imagined_trajectories, lambda_values, discount
        )
    critic_grads = update(
        fabric, value_loss, critic_optimizer, cfg.algo.critic.clip_gradients, error_if_nonfinite=False
    )
    if critic_grads:
        metrics["critic_grads"] = critic_grads.mean().detach()
    metrics["policy_loss"] = policy_loss.detach()
    metrics["value_loss"] = value_loss.detach()
    return metrics


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    target_critic: torch.nn.Module,
    world_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    data: Dict[str, Tensor],
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
    moments: Moments,
) -> None:
    """Runs one-step update of the agent: the world model learns from the batch (`world_model_learning`), then the
    actor and the critic from the trajectories imagined from it (`behaviour_learning`).

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        actor (_FabricModule): the actor model wrapped with Fabric.
        critic (_FabricModule): the critic model wrapped with Fabric.
        target_critic (nn.Module): the target critic model.
        world_optimizer (Optimizer): the world optimizer.
        actor_optimizer (Optimizer): the actor optimizer.
        critic_optimizer (Optimizer): the critic optimizer.
        data (Dict[str, Tensor]): the batch of data to use for training.
        cfg (DictConfig): the configs.
        is_continuous (bool): whether or not the environment is continuous.
        actions_dim (Sequence[int]): the actions dimension.
        moments (Moments): the moments for normalizing the lambda values.
    """
    posteriors, recurrent_states, metrics = world_model_learning(fabric, cfg, world_model, world_optimizer, data)
    behaviour = behaviour_learning(
        fabric,
        cfg,
        world_model,
        actor,
        critic,
        target_critic,
        actor_optimizer,
        critic_optimizer,
        moments,
        posteriors,
        recurrent_states,
        data["terminated"],
        is_continuous,
        actions_dim,
    )

    # Log metrics
    metrics["Loss/policy_loss"] = behaviour["policy_loss"]
    metrics["Loss/value_loss"] = behaviour["value_loss"]
    if "actor_grads" in behaviour:
        metrics["Grads/actor"] = behaviour["actor_grads"]
    if "critic_grads" in behaviour:
        metrics["Grads/critic"] = behaviour["critic_grads"]

    # Reset everything
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)
    return metrics


class DreamerV3(Algorithm):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each on its own batch of sequences: the world model, then the
    actor and the critic on trajectories imagined from the batch."""

    off_policy = True
    restart_crashed_envs = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        # These arguments cannot be changed
        cfg.env.frame_stack = -1
        if 2 ** int(np.log2(cfg.env.screen_size)) != cfg.env.screen_size:
            raise ValueError(f"The screen size must be a power of 2, got: {cfg.env.screen_size}")

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[DreamerV3State, EnvIndependentReplayBuffer]:
        cfg = self.cfg
        fabric = self.fabric
        self.is_continuous = isinstance(action_space, gym.spaces.Box)
        is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        self.actions_dim = tuple(
            action_space.shape
            if self.is_continuous
            else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
        )
        if not isinstance(obs_space, gym.spaces.Dict):
            raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {obs_space}")
        if (
            len(set(cfg.algo.cnn_keys.encoder).intersection(set(cfg.algo.cnn_keys.decoder))) == 0
            and len(set(cfg.algo.mlp_keys.encoder).intersection(set(cfg.algo.mlp_keys.decoder))) == 0
        ):
            raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjointed")
        if len(set(cfg.algo.cnn_keys.decoder) - set(cfg.algo.cnn_keys.encoder)) > 0:
            raise RuntimeError(
                "The CNN keys of the decoder must be contained in the encoder ones. "
                f"Those keys are decoded without being encoded: {list(set(cfg.algo.cnn_keys.decoder))}"
            )
        if len(set(cfg.algo.mlp_keys.decoder) - set(cfg.algo.mlp_keys.encoder)) > 0:
            raise RuntimeError(
                "The MLP keys of the decoder must be contained in the encoder ones. "
                f"Those keys are decoded without being encoded: {list(set(cfg.algo.mlp_keys.decoder))}"
            )
        if cfg.metric.log_level > 0:
            fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
            fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
            fabric.print("Decoder CNN keys:", cfg.algo.cnn_keys.decoder)
            fabric.print("Decoder MLP keys:", cfg.algo.mlp_keys.decoder)

        world_model, actor, critic, target_critic, self._policy = build_agent(
            fabric, self.actions_dim, self.is_continuous, cfg, obs_space
        )

        world_optimizer = hydra.utils.instantiate(
            cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
        )
        actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters(), _convert_="all")
        critic_optimizer = hydra.utils.instantiate(
            cfg.algo.critic.optimizer, params=critic.parameters(), _convert_="all"
        )
        world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
            world_optimizer, actor_optimizer, critic_optimizer
        )
        moments = Moments(
            cfg.algo.actor.moments.decay,
            cfg.algo.actor.moments.max,
            cfg.algo.actor.moments.percentile.low,
            cfg.algo.actor.moments.percentile.high,
        )
        state = DreamerV3State(
            world_model=world_model,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            world_optimizer=world_optimizer,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            moments=moments,
        )
        # One buffer of sequences per environment, sampled independently
        buffer = EnvIndependentReplayBuffer(
            env_buffer_size(fabric, cfg, dry_run_size=2),
            n_envs=cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            buffer_cls=SequentialReplayBuffer,
            seed=cfg.seed + fabric.global_rank,
        )
        self.schedule = schedule
        return state, buffer

    def policy(self, state: DreamerV3State) -> PlayerDV3:
        """The policy to play with: it shares its weights with the trained agent (`build_agent`)."""
        return self._policy

    def player(self, state: DreamerV3State) -> SequencePlayer:
        # Random actions until `algo.learning_starts`, except with MineDojo (its action masks)
        random_warmup = "minedojo" not in self.cfg.env.wrapper._target_.lower()
        return SequencePlayer(
            self.fabric,
            self.cfg,
            self.policy(state),
            self.schedule,
            self.actions_dim,
            self.is_continuous,
            random_warmup,
        )

    def batches(
        self, state: DreamerV3State, buffer: EnvIndependentReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        yield from sample_batches(self.fabric, self.cfg, buffer, n_steps)

    def train_step(self, state: DreamerV3State, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg
        # The target critic follows the critic: every `critic.per_rank_target_network_update_freq` gradient steps, an
        # exponential moving average with `critic.tau`; at the first gradient step, a copy
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            tau = 1 if step == 0 else cfg.algo.critic.tau
            for cp, tcp in zip(state.critic.module.parameters(), state.target_critic.parameters()):
                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
        metrics = train(
            self.fabric,
            state.world_model,
            state.actor,
            state.critic,
            state.target_critic,
            state.world_optimizer,
            state.actor_optimizer,
            state.critic_optimizer,
            batch,
            cfg,
            self.is_continuous,
            self.actions_dim,
            state.moments,
        )
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = DreamerV3(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir, greedy=False, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": state.world_model,
            "actor": state.actor,
            "critic": state.critic,
            "target_critic": state.target_critic,
            "moments": state.moments,
        }
        register_model(fabric, log_models, cfg, models_to_log)
