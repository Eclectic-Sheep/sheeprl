"""DreamerV3 as published in Nature (Hafner et al., 2025, "Mastering diverse control tasks through world models",
https://www.nature.com/articles/s41586-025-08744-2), from the official implementation
https://github.com/danijar/dreamerv3.

Compared with DreamerV3 of the 2023 paper (`sheeprl.algos.dreamer_v3`), besides the architecture (`agent.py`):
- the world model, the actor and the critic are trained together, on one loss with one optimizer (LaProp with
  adaptive gradient clipping, `optim.py`); the imagination doesn't backpropagate into the world model;
- every action is learned with REINFORCE, also the continuous ones;
- the continue model predicts the discount (`algo.world_model.continue_discount`), and the imagined continues are
  probabilities;
- the critic is also trained on the replayed sequences (`algo.critic.replay_loss`), towards lambda-returns that
  bootstrap from the imagined ones, also backpropagating into the world model;
- the critic is regularized towards a slow copy of itself, and bootstraps from itself (not from the slow copy);
- the replay buffer keeps the latent states of the steps (computed by the player and refreshed by every training on
  them): a sequence starts from the latent state of the step before it (`algo.replay_context`) instead of zeros;
- the batches start with the sequences of the new steps (the online queue of the replay buffer, `buffer.online`), and
  only the rest of them is sampled uniformly: every step is trained on soon after it is played;
- the policy plays from the first step (no random actions): `algo.learning_starts` only delays the training.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.dreamer_v2 import actions_dim_of, check_keys
from sheeprl.algos.dreamer_v2.utils import MAX_SAMPLED_BATCHES, env_buffer_size
from sheeprl.algos.dreamer_v3.loss import categorical_kl
from sheeprl.algos.dreamer_v3_5.agent import Actor, PlayerDV3_5, build_agent
from sheeprl.algos.dreamer_v3_5.loss import TwoHot, binary_loss, lambda_return, mse, symlog_mse
from sheeprl.algos.dreamer_v3_5.utils import Moments, prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, Metrics, TrainSchedule, TrainState, run
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer, get_tensor
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.fabric import autocast_cache_scope, update
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import dotdict

# The keys of the latent states kept in the replay buffer, and of the position of the steps in it
LATENT_KEYS = ("deter", "stoch")
STEP_ID_KEY = "stepid"


def world_model_loss_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The configuration of `world_model_loss`, as plain values."""
    world_model_cfg = cfg.algo.world_model
    scales = cfg.algo.loss_scales
    return {
        "cnn_keys": tuple(cfg.algo.cnn_keys.encoder),
        "mlp_keys": tuple(cfg.algo.mlp_keys.encoder),
        "cnn_decoder_keys": tuple(cfg.algo.cnn_keys.decoder),
        "mlp_decoder_keys": tuple(cfg.algo.mlp_keys.decoder),
        "kl_free_nats": float(world_model_cfg.kl_free_nats),
        "continue_target": float(cfg.algo.gamma) if world_model_cfg.continue_discount else 1.0,
        "reward_grad": bool(world_model_cfg.reward_grad),
        "replay_grad": bool(cfg.algo.critic.replay_grad),
        "observation_scale": float(scales.observation),
        "reward_scale": float(scales.reward),
        "continue_scale": float(scales["continue"]),
        "dynamic_scale": float(scales.dynamic),
        "representation_scale": float(scales.representation),
    }


def world_model_loss(
    world_model: WorldModel,
    obs: Dict[str, Tensor],
    prev_actions: Tensor,
    is_first: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    recurrent_state: Tensor,
    posterior: Tensor,
    *,
    cnn_keys: Sequence[str],
    mlp_keys: Sequence[str],
    cnn_decoder_keys: Sequence[str],
    mlp_decoder_keys: Sequence[str],
    kl_free_nats: float,
    continue_target: float,
    reward_grad: bool,
    replay_grad: bool,
    observation_scale: float,
    reward_scale: float,
    continue_scale: float,
    dynamic_scale: float,
    representation_scale: float,
    entropies: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """The loss of the world model on a batch of sequences (time first), without the optimization step: it can be
    compiled (`algo.compile`). The configuration is given by `world_model_loss_kwargs`.

    Args:
        obs: the observations (the images as uint8 or float values in [0, 255]).
        prev_actions: the actions that led to every step (the ones of the step before).
        is_first: whether the steps start an episode.
        rewards: the rewards of the steps.
        terminated: whether the episodes terminated at the steps.
        recurrent_state: the recurrent state before the first step, of shape `[1, B, recurrent_state_size]`.
        posterior: the stochastic state before the first step (flattened), of shape `[1, B, stochastic_size]`.
        entropies: whether to compute the entropies of the posteriors and of the priors (metrics).

    Returns:
        The loss (the losses of the world model scaled by `algo.loss_scales`), the latent states of the batch with
        their gradients if the replay loss of the critic trains the world model (`algo.critic.replay_grad`), the
        posteriors and the recurrent states of the batch without gradients (the starting points of the imagination and
        the latent states written back in the replay buffer), and the metrics.
    """
    sequence_length, batch_size = is_first.shape[:2]
    encoder_obs = {k: obs[k].float() / 255.0 - 0.5 for k in cnn_keys}
    encoder_obs.update({k: obs[k].float() for k in mlp_keys})
    embedded_obs = world_model.encoder(encoder_obs)

    # The parts of the recurrent and representation models that depend on the observations and on the actions, for
    # the whole sequence at once. The actions that lead to the first step of an episode are zeroed
    observation_projection = world_model.rssm.project_observations(embedded_obs)
    action_embedding = world_model.rssm.embed_actions(prev_actions * (1 - is_first))
    recurrent_states, posteriors, posteriors_logits = [], [], []
    for i in range(sequence_length):
        recurrent_state, posterior, posterior_logits = world_model.rssm.dynamic(
            posterior,
            recurrent_state,
            action_embedding[i : i + 1],
            observation_projection[i : i + 1],
            is_first[i : i + 1],
        )
        recurrent_states.append(recurrent_state)
        posteriors.append(posterior)
        posteriors_logits.append(posterior_logits)
    recurrent_states = torch.cat(recurrent_states, 0)
    posteriors = torch.cat(posteriors, 0)
    posteriors_logits = torch.cat(posteriors_logits, 0)
    # The priors don't take part in the recurrence: computed for the whole sequence at once
    priors_logits = world_model.rssm.prior_logits(recurrent_states)
    latent_states = torch.cat((posteriors, recurrent_states), -1)

    # Reconstruction: the images in [0, 1], the vectors in the symlog space
    reconstructed_obs = world_model.observation_model(latent_states)
    observation_loss = sum(mse(reconstructed_obs[k], obs[k].float() / 255.0, 3) for k in cnn_decoder_keys) + sum(
        symlog_mse(reconstructed_obs[k], obs[k], obs[k].dim() - 2) for k in mlp_decoder_keys
    )
    reward_inputs = latent_states if reward_grad else latent_states.detach()
    reward_loss = TwoHot(world_model.reward_model(reward_inputs)).loss(rewards.squeeze(-1))
    continue_loss = binary_loss(
        world_model.continue_model(latent_states).squeeze(-1), (1 - terminated.squeeze(-1)) * continue_target
    )
    kl = categorical_kl(posteriors_logits.detach(), priors_logits)
    dynamic_loss = torch.clamp(kl, min=kl_free_nats)
    representation_loss = torch.clamp(categorical_kl(posteriors_logits, priors_logits.detach()), min=kl_free_nats)
    state_loss = dynamic_scale * dynamic_loss.mean() + representation_scale * representation_loss.mean()
    loss = (
        observation_scale * observation_loss.mean()
        + reward_scale * reward_loss.mean()
        + continue_scale * continue_loss.mean()
        + state_loss
    )
    metrics = {
        "Loss/world_model_loss": loss.detach(),
        "Loss/observation_loss": observation_loss.mean().detach(),
        "Loss/reward_loss": reward_loss.mean().detach(),
        "Loss/state_loss": state_loss.detach(),
        "Loss/continue_loss": continue_loss.mean().detach(),
        "State/kl": kl.mean().detach(),
    }
    if entropies:
        metrics["State/post_entropy"] = -(posteriors_logits.exp() * posteriors_logits).sum((-2, -1)).mean().detach()
        metrics["State/prior_entropy"] = -(priors_logits.exp() * priors_logits).sum((-2, -1)).mean().detach()
    return (
        loss,
        latent_states if replay_grad else latent_states.detach(),
        posteriors.detach(),
        recurrent_states.detach(),
        metrics,
    )


def imagine(
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    posteriors: Tensor,
    recurrent_states: Tensor,
    *,
    horizon: int,
    discount: float,
    lmbda: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Imagine `horizon` steps from every latent state of the batch, with the actions of the actor, and estimate their
    lambda-returns. Nothing backpropagates through it: it runs without gradients and can be compiled (`algo.compile`).

    Args:
        discount: the discount of the returns: 1 when the continue model predicts the discount
            (`algo.world_model.continue_discount`), `algo.gamma` otherwise.

    Returns:
        The imagined latent states and the actions taken in them (`horizon + 1` steps, the first one the starting
        point), the values of the critic and of the slow critic in them, the lambda-returns of the first `horizon`
        steps and the weights of the steps (the cumulative product of their discounts).
    """
    stochastic_size, recurrent_state_size = posteriors.shape[-1], recurrent_states.shape[-1]
    prior = posteriors.reshape(1, -1, stochastic_size)
    recurrent_state = recurrent_states.reshape(1, -1, recurrent_state_size)
    latent_state = torch.cat((prior, recurrent_state), -1)
    latent_states, actions = [latent_state], []
    for _ in range(horizon):
        action = torch.cat(actor(latent_state)[0], -1)
        actions.append(action)
        prior, recurrent_state = world_model.rssm.imagination(prior, recurrent_state, action)
        latent_state = torch.cat((prior, recurrent_state), -1)
        latent_states.append(latent_state)
    actions.append(torch.cat(actor(latent_state)[0], -1))
    latent_states = torch.cat(latent_states, 0)
    actions = torch.cat(actions, 0)

    # The rewards, the continues (probabilities) and the values of the imagined steps
    rewards = TwoHot(world_model.reward_model(latent_states)).mean
    continues = torch.sigmoid(world_model.continue_model(latent_states).squeeze(-1).float())
    values = TwoHot(critic(latent_states)).mean
    slow_values = TwoHot(target_critic(latent_states)).mean
    returns = lambda_return(torch.zeros_like(continues), 1 - continues, rewards, values, discount, lmbda)
    weights = torch.cumprod(discount * continues, 0) / discount
    return latent_states, actions, values, slow_values, returns, weights


def actor_critic_loss(
    actor: Actor | nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    imagined_latent_states: Tensor,
    imagined_actions: Tensor,
    values: Tensor,
    slow_values: Tensor,
    returns: Tensor,
    weights: Tensor,
    invscale: Tensor,
    latent_states: Tensor,
    last: Tensor,
    terminated: Tensor,
    rewards: Tensor,
    *,
    actions_dim: Sequence[int],
    ent_coef: float,
    slow_regularizer: float,
    gamma: float,
    lmbda: float,
    replay_loss: bool,
    policy_scale: float,
    value_scale: float,
    replay_value_scale: float,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """The losses of the actor and of the critic (`imag_loss` and `repl_loss` of DreamerV3), scaled by
    `algo.loss_scales`, from the imagined trajectories (`imagine`) and the scale of the returns (`invscale`). Can be
    compiled (`algo.compile`).

    - Actor: REINFORCE on the imagined actions, with the advantages of the lambda-returns over the values divided by
      the scale of the returns, and an entropy bonus.
    - Critic: the two-hot losses of the lambda-returns and of the values of the slow critic (a regularizer), on the
      imagined steps and, with `replay_loss`, on the replayed ones. The lambda-returns of the replayed sequence
      bootstrap from the imagined ones of its steps.

    Returns:
        The loss and the metrics.
    """
    # Actor
    dists = actor(imagined_latent_states[:-1], with_actions=False)[1]
    actions = (
        torch.split(imagined_actions[:-1], [int(d) for d in actions_dim], -1)
        if len(dists) > 1
        else [imagined_actions[:-1]]
    )
    log_prob = sum(d.log_prob(a) for d, a in zip(dists, actions))
    entropy = sum(d.entropy() for d in dists)
    advantage = (returns - values[:-1]) / invscale
    policy_loss = (weights[:-1] * -(log_prob * advantage + ent_coef * entropy)).mean()

    # Critic on the imagined steps
    value_dist = TwoHot(critic(imagined_latent_states[:-1]))
    value_loss = (
        weights[:-1] * (value_dist.loss(returns) + slow_regularizer * value_dist.loss(slow_values[:-1]))
    ).mean()
    loss = policy_scale * policy_loss + value_scale * value_loss
    metrics = {
        "Loss/policy_loss": policy_loss.detach(),
        "Loss/value_loss": value_loss.detach(),
        "Actor/entropy": entropy.mean().detach(),
        "Values/return": returns.mean().detach(),
        "Values/value": values.mean().detach(),
    }

    # Critic on the replayed steps, towards returns that bootstrap from the imagined returns of the steps (the last step
    # of the sequences has no return: sequences of one step have none)
    if replay_loss and latent_states.shape[0] > 1:
        sequence_length, batch_size = latent_states.shape[:2]
        bootstrap = returns[0].reshape(sequence_length, batch_size)
        replay_returns = lambda_return(last, terminated, rewards, bootstrap, gamma, lmbda)
        replay_value_dist = TwoHot(critic(latent_states[:-1]))
        slow_replay_values = TwoHot(target_critic(latent_states[:-1].detach())).mean.detach()
        replay_value_loss = (
            (1 - last[:-1])
            * (replay_value_dist.loss(replay_returns) + slow_regularizer * replay_value_dist.loss(slow_replay_values))
        ).mean()
        loss = loss + replay_value_scale * replay_value_loss
        metrics["Loss/replay_value_loss"] = replay_value_loss.detach()
    return loss, metrics


def train(
    fabric: Fabric,
    cfg: Dict[str, Any],
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    optimizer: Optimizer,
    moments: Moments,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    actions_dim: Sequence[int],
) -> Optional[Tuple[Tensor, Tensor]]:
    """One gradient step of the agent on a batch of sequences (time first): the world model, the actor and the critic
    together, on the sum of their losses.

    With `algo.replay_context` > 0, the first `algo.replay_context` steps of the sequences are their context: the
    training starts from the latent state of the last of them (kept in the replay buffer) and the actions taken there.

    Returns:
        The latent states of the trained steps (the recurrent states in float16 and the classes of the stochastic
        states), to write back in the replay buffer; `None` without replay context.
    """
    world_model_cfg = cfg.algo.world_model
    context = cfg.algo.replay_context
    stochastic_size, discrete_size = world_model_cfg.stochastic_size, world_model_cfg.discrete_size
    batch_size = data["is_first"].shape[1]
    if context > 0:
        recurrent_state = data["deter"][context - 1 : context].float()
        posterior = F.one_hot(data["stoch"][context - 1 : context].long(), discrete_size).flatten(-2).float()
        prev_actions = data["actions"][context - 1 : -1].float()
        data = {k: v[context:] for k, v in data.items() if k not in LATENT_KEYS}
        is_first = data["is_first"].float()
    else:
        recurrent_state = torch.zeros(
            1, batch_size, world_model_cfg.recurrent_model.recurrent_state_size, device=fabric.device
        )
        posterior = torch.zeros(1, batch_size, stochastic_size * discrete_size, device=fabric.device)
        actions = data["actions"].float()
        prev_actions = torch.cat((torch.zeros_like(actions[:1]), actions[:-1]), 0)
        # Every sequence starts an episode: the world model starts from zeros
        is_first = data["is_first"].float().clone()
        is_first[0] = 1
    obs = {k: data[k] for k in cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder}
    rewards = data["rewards"].float()
    terminated = data["terminated"].float()
    last = torch.maximum(terminated, data["truncated"].float())

    mark_gradient_step(fabric, cfg)
    scales = cfg.algo.loss_scales
    # Cast the weights to low precision once for the whole forward pass, not at every step of the unroll
    with autocast_cache_scope(fabric):
        wm_loss, latent_states, posteriors, recurrent_states, metrics = compiled(world_model_loss, fabric, cfg)(
            world_model,
            obs,
            prev_actions,
            is_first,
            rewards,
            terminated,
            recurrent_state,
            posterior,
            **world_model_loss_kwargs(cfg),
            entropies=aggregator is not None and not aggregator.disabled,
        )
        with torch.no_grad():
            imagined_latent_states, imagined_actions, values, slow_values, returns, weights = compiled(
                imagine, fabric, cfg
            )(
                world_model,
                actor,
                critic,
                target_critic,
                posteriors,
                recurrent_states,
                horizon=cfg.algo.horizon,
                discount=1.0 if world_model_cfg.continue_discount else float(cfg.algo.gamma),
                lmbda=float(cfg.algo.lmbda),
            )
        # The scale of the returns, from their percentiles (not compiled: it updates its state in place)
        _, invscale = moments(returns, fabric)
        ac_loss, ac_metrics = compiled(actor_critic_loss, fabric, cfg)(
            actor,
            critic,
            target_critic,
            imagined_latent_states,
            imagined_actions,
            values,
            slow_values,
            returns,
            weights,
            invscale,
            latent_states,
            last.squeeze(-1),
            terminated.squeeze(-1),
            rewards.squeeze(-1),
            actions_dim=tuple(int(dim) for dim in actions_dim),
            ent_coef=float(cfg.algo.actor.ent_coef),
            slow_regularizer=float(cfg.algo.critic.slow_regularizer),
            gamma=float(cfg.algo.gamma),
            lmbda=float(cfg.algo.lmbda),
            replay_loss=bool(cfg.algo.critic.replay_loss),
            policy_scale=float(scales.policy),
            value_scale=float(scales.value),
            replay_value_scale=float(scales.replay_value),
        )
    update(fabric, wm_loss + ac_loss, optimizer)

    if aggregator and not aggregator.disabled:
        metrics.update(ac_metrics)
        grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        metrics["Grads/agent"] = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        for name, value in metrics.items():
            aggregator.update(name, value)

    if context > 0:
        classes = posteriors.unflatten(-1, (stochastic_size, discrete_size)).argmax(-1)
        return recurrent_states.half(), classes.to(torch.uint8) if discrete_size <= 256 else classes
    return None


@torch.no_grad()
def update_target_critic(critic: nn.Module, target_critic: nn.Module, tau: float) -> None:
    """Move the weights of the slow critic towards the ones of the critic: `(1 - tau) * slow + tau * critic`."""
    target_params = list(target_critic.parameters())
    torch._foreach_lerp_(target_params, [p.to(t.dtype) for p, t in zip(critic.parameters(), target_params)], tau)


def sample_sequences(
    rb: EnvIndependentReplayBuffer,
    batch_size: int,
    sequence_length: int,
    n_samples: int,
    device: torch.device,
    from_numpy: bool = False,
    online: bool = False,
) -> Tuple[Dict[str, Tensor], np.ndarray]:
    """`n_samples` batches of sequences, of shape `[n_samples, sequence_length, batch_size, ...]` and in the dtypes of
    the buffer, on `device`, but the identifiers of their steps (`STEP_ID_KEY`), on the CPU. With `online`, the batches
    start with the sequences of the online queue of the buffer."""
    sample = rb.sample(batch_size=batch_size, sequence_length=sequence_length, n_samples=n_samples, online=online)
    step_ids = sample.pop(STEP_ID_KEY)
    return {k: get_tensor(v, device=device, from_numpy=from_numpy) for k, v in sample.items()}, step_ids


def write_latent_states(
    rb: EnvIndependentReplayBuffer, updates: List[Tuple[np.ndarray, Tensor, Tensor]], buffer_size: int
) -> None:
    """Write the latent states computed by the trainings (`train`) back in the replay buffer, at the steps they were
    computed for. The steps are found from their identifiers: the environment and the number of steps of that
    environment added before them, whose remainder by `buffer_size` is their position in its buffer."""
    if len(updates) == 0:
        return
    step_ids = np.concatenate([u[0] for u in updates], 1)
    # One copy from the device for all of them
    recurrent_states = torch.cat([u[1] for u in updates], 1).cpu().numpy()
    stochastic_states = torch.cat([u[2] for u in updates], 1).cpu().numpy()
    envs = step_ids[0, :, 0]
    positions = step_ids[..., 1] % buffer_size
    for env in np.unique(envs):
        columns = np.flatnonzero(envs == env)
        # The steps of the sequences in the order of the trainings: the sequences can overlap, and every step gets the
        # latent state of the last training
        rows = positions[:, columns].T.reshape(-1)
        deter = recurrent_states[:, columns].transpose(1, 0, 2).reshape(len(rows), -1)
        stoch = stochastic_states[:, columns].transpose(1, 0, 2).reshape(len(rows), -1)
        rows, last = np.unique(rows[::-1], return_index=True)
        last = len(deter) - 1 - last
        buffer = rb.buffer[env]
        buffer["deter"][rows, 0] = deter[last]
        buffer["stoch"][rows, 0] = stoch[last]


def step_counters(rb: EnvIndependentReplayBuffer) -> np.ndarray:
    """The number of steps added to the buffer of every environment, from the identifiers of the steps it holds (0 for
    an empty buffer)."""
    counters = np.zeros(rb.n_envs, dtype=np.int64)
    for i, buffer in enumerate(rb.buffer):
        if not buffer.empty:
            counters[i] = int(np.asarray(buffer[STEP_ID_KEY])[..., 1].max()) + 1
    return counters


@dataclass
class DreamerV3_5State(TrainState):
    # Encoder, RSSM, decoder, reward and continue models
    world_model: WorldModel
    actor: Actor
    critic: nn.Module
    # Slow copy of the critic, the regularizer of its targets
    target_critic: nn.Module
    # One optimizer for the world model, the actor and the critic
    optimizer: Optimizer
    # Percentiles of the lambda-values, which normalize the advantages of the actor
    moments: Moments


class LatentSequencePlayer:
    """Plays the policy in the environments from the first step and writes in the replay buffer the sequences the
    agent learns from, as `SequencePlayer` of DreamerV3 does, with the latent states of the player (`LATENT_KEYS`) and
    the identifier of every step (`STEP_ID_KEY`: the environment and the steps added to its buffer before it), where
    the trainings write back the latent states they compute (`write_latent_states`)."""

    def __init__(
        self, fabric: Fabric, cfg: Dict[str, Any], policy: PlayerDV3_5, actions_dim: Sequence[int], is_continuous: bool
    ) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.policy = policy
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
        self.stochastic_size = cfg.algo.world_model.stochastic_size
        self.stoch_dtype = np.uint8 if cfg.algo.world_model.discrete_size <= 256 else np.int64
        # The row written at the next step and the steps added to the buffer of every environment; created at the first
        # step, from the first observations and from the buffer (of a resumed run)
        self.step_data: Optional[Dict[str, np.ndarray]] = None
        self.counters: Optional[np.ndarray] = None

    def step(self, env: EnvRunner, buffer: EnvIndependentReplayBuffer) -> None:
        cfg = self.cfg
        num_envs = env.num_envs
        env_ids = np.arange(num_envs)
        if self.step_data is None:
            self.step_data = {k: env.obs[k][np.newaxis] for k in self.obs_keys}
            self.step_data["rewards"] = np.zeros((1, num_envs, 1), dtype=np.float32)
            self.step_data["truncated"] = np.zeros((1, num_envs, 1), dtype=np.float32)
            self.step_data["terminated"] = np.zeros((1, num_envs, 1), dtype=np.float32)
            self.step_data["is_first"] = np.ones_like(self.step_data["terminated"])
            self.counters = step_counters(buffer)
            self.policy.init_states()
        step_data, counters = self.step_data, self.counters

        # The latent states of the player go in the replay buffer with the step
        torch_obs = prepare_obs(self.fabric, env.obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=num_envs)
        mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        actions = self.policy.get_actions(torch_obs, mask=mask or None)
        latent_states = (
            self.policy.recurrent_state.half(),
            self.policy.stochastic_state.view(1, num_envs, self.stochastic_size, -1).argmax(-1),
        )
        if self.is_continuous:
            real_actions = torch.stack(actions, dim=-1).cpu().numpy()
        else:
            real_actions = torch.stack([act.argmax(dim=-1) for act in actions], dim=-1).cpu().numpy()
        step_data["actions"] = torch.cat(actions, -1).cpu().numpy().reshape((1, num_envs, -1))
        step_data["deter"] = latent_states[0].cpu().numpy()
        step_data["stoch"] = latent_states[1].cpu().numpy().astype(self.stoch_dtype)
        step_data[STEP_ID_KEY] = np.stack((env_ids, counters), -1)[np.newaxis]
        buffer.add(step_data, validate_args=cfg.buffer.validate_args)
        counters += 1

        step = env.step(real_actions)
        dones = np.logical_or(step.terminated, step.truncated).astype(np.uint8)

        step_data["is_first"] = np.zeros_like(step_data["terminated"])
        if "restart_on_exception" in step.info:
            restarted_envs = []
            for i, agent_roe in enumerate(step.info["restart_on_exception"]):
                if agent_roe and not dones[i]:
                    # The last observation stored for the restarted environment ends its episode
                    last_inserted_idx = (counters[i] - 1) % buffer.buffer[i].buffer_size
                    buffer.buffer[i]["terminated"][last_inserted_idx] = 0
                    buffer.buffer[i]["truncated"][last_inserted_idx] = 1
                    # The observation returned after the restart starts a new episode
                    step_data["is_first"][:, i] = 1
                    restarted_envs.append(i)
            if len(restarted_envs) > 0:
                self.policy.init_states(restarted_envs)

        for k in self.obs_keys:
            step_data[k] = step.next_obs[k][np.newaxis]
        rewards = step.rewards.reshape((1, num_envs, -1)).astype(np.float32)
        step_data["terminated"] = step.terminated.reshape((1, num_envs, -1)).astype(np.float32)
        step_data["truncated"] = step.truncated.reshape((1, num_envs, -1)).astype(np.float32)
        step_data["rewards"] = np.tanh(rewards) if cfg.env.clip_rewards else rewards

        dones_idxes = dones.nonzero()[0].tolist()
        reset_envs = len(dones_idxes)
        if reset_envs > 0:
            # The last observations of the episodes, with no action and no latent state (the next steps start new
            # episodes, from zeros)
            final_obs = step.final_obs(dones_idxes, self.obs_keys)
            reset_data = {k: final_obs[k].astype(step.next_obs[k].dtype, copy=False)[np.newaxis] for k in self.obs_keys}
            reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
            reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
            reset_data["actions"] = np.zeros((1, reset_envs, int(np.sum(self.actions_dim))), dtype=np.float32)
            reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
            reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
            reset_data["deter"] = np.zeros((1, reset_envs, step_data["deter"].shape[-1]), dtype=np.float16)
            reset_data["stoch"] = np.zeros((1, reset_envs, self.stochastic_size), dtype=self.stoch_dtype)
            reset_data[STEP_ID_KEY] = np.stack((env_ids[dones_idxes], counters[dones_idxes]), -1)[np.newaxis]
            buffer.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)
            counters[dones_idxes] += 1

            step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
            step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
            step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
            step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
            self.policy.init_states(dones_idxes)


class DreamerV3_5(Algorithm):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each on its own batch of sequences preceded by
    `algo.replay_context` steps, whose latent states start the sequences: the world model, the actor and the critic,
    with one optimizer. The latent states computed by the trainings are written back in the buffer."""

    off_policy = True
    restart_crashed_envs = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        # These arguments cannot be changed
        cfg.env.frame_stack = -1
        if cfg.algo.replay_context < 0:
            raise ValueError(f"`algo.replay_context` must be non-negative, got: {cfg.algo.replay_context}")

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[DreamerV3_5State, EnvIndependentReplayBuffer]:
        cfg = self.cfg
        fabric = self.fabric
        self.actions_dim, self.is_continuous = actions_dim_of(action_space)
        check_keys(fabric, cfg, obs_space)

        world_model, actor, critic, target_critic, self._policy = build_agent(
            fabric, self.actions_dim, self.is_continuous, cfg, obs_space
        )
        optimizer = hydra.utils.instantiate(
            cfg.algo.optimizer,
            params=[*world_model.parameters(), *actor.parameters(), *critic.parameters()],
            _convert_="all",
        )
        state = DreamerV3_5State(
            world_model=world_model,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            optimizer=fabric.setup_optimizers(optimizer),
            moments=Moments(
                cfg.algo.actor.moments.decay,
                cfg.algo.actor.moments.max,
                cfg.algo.actor.moments.percentile.low,
                cfg.algo.actor.moments.percentile.high,
            ),
        )
        # The sampled sequences also hold the steps of their context: the buffer of every environment holds one
        self.context = cfg.algo.replay_context
        self.sampled_length = cfg.algo.per_rank_sequence_length + self.context
        sampling_cfg = dotdict(copy.deepcopy(cfg.as_dict()))
        sampling_cfg.algo.per_rank_sequence_length = self.sampled_length
        self.buffer_size = env_buffer_size(fabric, sampling_cfg, dry_run_size=2)
        buffer = EnvIndependentReplayBuffer(
            self.buffer_size,
            n_envs=cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            buffer_cls=SequentialReplayBuffer,
            seed=cfg.seed + fabric.global_rank,
        )
        return state, buffer

    def policy(self, state: DreamerV3_5State) -> PlayerDV3_5:
        """The policy to play with: it shares its weights with the trained agent (`build_agent`)."""
        return self._policy

    def player(self, state: DreamerV3_5State) -> LatentSequencePlayer:
        return LatentSequencePlayer(self.fabric, self.cfg, self.policy(state), self.actions_dim, self.is_continuous)

    def batches(
        self, state: DreamerV3_5State, buffer: EnvIndependentReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        # The batches are sampled a few at a time: the latent states computed on them (`train_step`) are written back in
        # the buffer before the next ones are sampled
        for first in range(0, n_steps, MAX_SAMPLED_BATCHES):
            n_samples = min(MAX_SAMPLED_BATCHES, n_steps - first)
            sample, step_ids = sample_sequences(
                buffer,
                cfg.algo.per_rank_batch_size,
                self.sampled_length,
                n_samples,
                self.fabric.device,
                from_numpy=cfg.buffer.from_numpy,
                online=cfg.buffer.online,
            )
            self.latent_updates: List[Tuple[np.ndarray, Tensor, Tensor]] = []
            for i in range(n_samples):
                self.step_ids = step_ids[i, self.context :]
                yield {k: v[i] for k, v in sample.items()}
            write_latent_states(buffer, self.latent_updates, self.buffer_size)

    def train_step(self, state: DreamerV3_5State, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg
        metrics = Metrics()
        latents = train(
            self.fabric,
            cfg,
            state.world_model,
            state.actor,
            state.critic,
            state.target_critic,
            state.optimizer,
            state.moments,
            batch,
            metrics,
            self.actions_dim,
        )
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            update_target_critic(state.critic, state.target_critic, cfg.algo.critic.tau)
        if latents is not None:
            self.latent_updates.append((self.step_ids, *latents))
        return metrics.values


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = DreamerV3_5(fabric, cfg)
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
