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
- the policy plays from the first step (no random actions): `algo.learning_starts` only delays the training.
"""

from __future__ import annotations

import copy
import os
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.utils import MAX_SAMPLED_BATCHES, env_buffer_size
from sheeprl.algos.dreamer_v3.loss import categorical_kl
from sheeprl.algos.dreamer_v3_5.agent import Actor, build_agent
from sheeprl.algos.dreamer_v3_5.loss import TwoHot, binary_loss, lambda_return, mse, symlog_mse
from sheeprl.algos.dreamer_v3_5.utils import Moments, prepare_obs, test
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer, get_tensor
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.env import get_episode_stats, get_vector_env_cls, make_env
from sheeprl.utils.fabric import autocast_cache_scope, update
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import phase_timer, timer, training_timer
from sheeprl.utils.utils import dotdict, off_policy_schedule, save_configs

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
) -> Tuple[Dict[str, Tensor], np.ndarray]:
    """`n_samples` batches of sequences, of shape `[n_samples, sequence_length, batch_size, ...]` and in the dtypes of
    the buffer, on `device`, but the identifiers of their steps (`STEP_ID_KEY`), on the CPU."""
    sample = rb.sample(batch_size=batch_size, sequence_length=sequence_length, n_samples=n_samples)
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


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from, weights_only=False)

    # These arguments cannot be changed
    cfg.env.frame_stack = -1
    if cfg.algo.replay_context < 0:
        raise ValueError(f"`algo.replay_context` must be non-negative, got: {cfg.algo.replay_context}")

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    vectorized_env = get_vector_env_cls(cfg.env.sync_env)
    envs = vectorized_env(
        [
            partial(
                RestartOnException,
                make_env(
                    cfg,
                    cfg.seed + rank * cfg.env.num_envs + i,
                    rank * cfg.env.num_envs,
                    log_dir if rank == 0 else None,
                    "train",
                    vector_env_idx=i,
                ),
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

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
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    world_model, actor, critic, target_critic, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"] if cfg.checkpoint.resume_from else None,
        state["actor"] if cfg.checkpoint.resume_from else None,
        state["critic"] if cfg.checkpoint.resume_from else None,
        state["target_critic"] if cfg.checkpoint.resume_from else None,
    )

    # One optimizer for the world model, the actor and the critic
    optimizer = hydra.utils.instantiate(
        cfg.algo.optimizer,
        params=[*world_model.parameters(), *actor.parameters(), *critic.parameters()],
        _convert_="all",
    )
    if cfg.checkpoint.resume_from:
        optimizer.load_state_dict(state["optimizer"])
    optimizer = fabric.setup_optimizers(optimizer)
    moments = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    if cfg.checkpoint.resume_from:
        moments.load_state_dict(state["moments"])

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # The sampled sequences also hold the steps of their context
    context = cfg.algo.replay_context
    sampled_length = cfg.algo.per_rank_sequence_length + context
    sampling_cfg = dotdict(copy.deepcopy(cfg.as_dict()))
    sampling_cfg.algo.per_rank_sequence_length = sampled_length

    # Local data
    # The buffer of every environment holds a sequence
    buffer_size = env_buffer_size(fabric, sampling_cfg, dry_run_size=2)
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=SequentialReplayBuffer,
        seed=cfg.seed + rank,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], EnvIndependentReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")
    # The steps added to the buffer of every environment: they identify the steps whose latent states are written back
    counters = step_counters(rb)

    # Global variables
    train_step = 0
    last_train = 0
    start_iter = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["iter_num"] // fabric.world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    # The policy step since which the interaction is timed: a resumed run times only its own steps, not the ones
    # played after the last log of the run it resumes
    last_timed_step = policy_step
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
    total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
    # The policy plays from the first step: the training starts after `learning_starts` policy steps (when every
    # environment has played a sampled sequence)
    _, train_starts, pretrain_steps, total_iters, ratio = off_policy_schedule(
        sampling_cfg,
        state if cfg.checkpoint.resume_from else None,
        start_iter,
        total_iters,
        policy_steps_per_iter,
        fabric.world_size,
    )

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )
    if cfg.checkpoint.every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )

    # Get the first environment observation and start the optimization
    num_envs = cfg.env.num_envs
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    stoch_dtype = np.uint8 if discrete_size <= 256 else np.int64
    env_ids = np.arange(num_envs)
    step_data = {}
    obs = envs.reset(seed=cfg.seed + rank * cfg.env.num_envs)[0]
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["rewards"] = np.zeros((1, num_envs, 1), dtype=np.float32)
    step_data["truncated"] = np.zeros((1, num_envs, 1), dtype=np.float32)
    step_data["terminated"] = np.zeros((1, num_envs, 1), dtype=np.float32)
    step_data["is_first"] = np.ones_like(step_data["terminated"])
    player.init_states()

    # The gradient steps of every process from the start of the run, also in the run it resumes
    cumulative_per_rank_gradient_steps = state.get("per_rank_gradient_steps", 0) if cfg.checkpoint.resume_from else 0
    for iter_num in range(start_iter, total_iters + 1):
        policy_step += policy_steps_per_iter

        with torch.inference_mode():
            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with phase_timer("Time/env_interaction_time"):
                # The policy plays from the first step: the latent states of the steps go in the replay buffer
                torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=num_envs)
                mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                actions = player.get_actions(torch_obs, mask=mask or None)
                latent_states = (
                    player.recurrent_state.half(),
                    player.stochastic_state.view(1, num_envs, stochastic_size, discrete_size).argmax(-1),
                )
                if is_continuous:
                    real_actions = torch.stack(actions, dim=-1).cpu().numpy()
                else:
                    real_actions = torch.stack([act.argmax(dim=-1) for act in actions], dim=-1).cpu().numpy()
                step_data["actions"] = torch.cat(actions, -1).cpu().numpy().reshape((1, num_envs, -1))
                step_data["deter"] = latent_states[0].cpu().numpy()
                step_data["stoch"] = latent_states[1].cpu().numpy().astype(stoch_dtype)
                step_data[STEP_ID_KEY] = np.stack((env_ids, counters), -1)[np.newaxis]
                rb.add(step_data, validate_args=cfg.buffer.validate_args)
                counters += 1

                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                dones = np.logical_or(terminated, truncated).astype(np.uint8)

            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:
                restarted_envs = []
                for i, agent_roe in enumerate(infos["restart_on_exception"]):
                    if agent_roe and not dones[i]:
                        # The last observation stored for the restarted environment ends its episode
                        last_inserted_idx = (counters[i] - 1) % rb.buffer[i].buffer_size
                        rb.buffer[i]["terminated"][last_inserted_idx] = 0
                        rb.buffer[i]["truncated"][last_inserted_idx] = 1
                        # The observation returned after the restart starts a new episode
                        step_data["is_first"][:, i] = 1
                        restarted_envs.append(i)
                if len(restarted_envs) > 0:
                    player.init_states(restarted_envs)

            if cfg.metric.log_level > 0:
                for i, ep_rew, ep_len in get_episode_stats(infos):
                    if aggregator and not aggregator.disabled:
                        aggregator.update("Rewards/rew_avg", ep_rew)
                        aggregator.update("Game/ep_len_avg", ep_len)
                    fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew}")

            # Save the real next observation
            real_next_obs = copy.deepcopy(next_obs)
            if "final_obs" in infos:
                for idx, final_obs in enumerate(infos["final_obs"]):
                    if final_obs is not None:
                        for k, v in final_obs.items():
                            real_next_obs[k][idx] = v

            for k in obs_keys:
                step_data[k] = next_obs[k][np.newaxis]

            # next_obs becomes the new obs
            obs = next_obs

            rewards = rewards.reshape((1, num_envs, -1)).astype(np.float32)
            step_data["terminated"] = terminated.reshape((1, num_envs, -1)).astype(np.float32)
            step_data["truncated"] = truncated.reshape((1, num_envs, -1)).astype(np.float32)
            step_data["rewards"] = clip_rewards_fn(rewards)

            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                # The last observations of the episodes, with no action and no latent state (the next steps start new
                # episodes, from zeros)
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
                reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                reset_data["actions"] = np.zeros((1, reset_envs, int(np.sum(actions_dim))), dtype=np.float32)
                reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                reset_data["deter"] = np.zeros((1, reset_envs, step_data["deter"].shape[-1]), dtype=np.float16)
                reset_data["stoch"] = np.zeros((1, reset_envs, stochastic_size), dtype=stoch_dtype)
                reset_data[STEP_ID_KEY] = np.stack((env_ids[dones_idxes], counters[dones_idxes]), -1)[np.newaxis]
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)
                counters[dones_idxes] += 1

                # Reset already inserted step data
                step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
                player.init_states(dones_idxes)

        # Train the agent
        if iter_num >= train_starts:
            ratio_steps = policy_step - (train_starts - 1) * policy_steps_per_iter
            per_rank_gradient_steps = ratio(ratio_steps / world_size)
            if iter_num == train_starts:
                per_rank_gradient_steps += pretrain_steps
            if per_rank_gradient_steps > 0:
                with training_timer(fabric.device):
                    # The batches are sampled a few at a time: the latent states computed on them are written back
                    # in the buffer before the next ones are sampled
                    remaining = per_rank_gradient_steps
                    while remaining > 0:
                        n_samples = min(MAX_SAMPLED_BATCHES, remaining)
                        sample, step_ids = sample_sequences(
                            rb,
                            cfg.algo.per_rank_batch_size,
                            sampled_length,
                            n_samples,
                            device,
                            from_numpy=cfg.buffer.from_numpy,
                        )
                        latent_updates = []
                        for i in range(n_samples):
                            latents = train(
                                fabric,
                                cfg,
                                world_model,
                                actor,
                                critic,
                                target_critic,
                                optimizer,
                                moments,
                                {k: v[i] for k, v in sample.items()},
                                aggregator,
                                actions_dim,
                            )
                            if (
                                cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                                == 0
                            ):
                                update_target_critic(critic, target_critic, cfg.algo.critic.tau)
                            cumulative_per_rank_gradient_steps += 1
                            if latents is not None:
                                latent_updates.append((step_ids[i, context:], *latents))
                        write_latent_states(rb, latent_updates, buffer_size)
                        remaining -= n_samples
                    # The gradient steps of all the processes
                    train_step += world_size * per_rank_gradient_steps

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or iter_num == total_iters):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
            )

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics and timer_metrics["Time/train_time"] > 0:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_timed_step) * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step
            last_timed_step = policy_step
            last_train = train_step

        # Checkpoint Model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            iter_num == total_iters and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "world_model": world_model.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "moments": moments.state_dict(),
                "ratio": ratio.state_dict(),
                "per_rank_gradient_steps": cumulative_per_rank_gradient_steps,
                "iter_num": iter_num * fabric.world_size,
                "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir, greedy=False, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "actor": actor,
            "critic": critic,
            "target_critic": target_critic,
            "moments": moments,
        }
        register_model(fabric, log_models, cfg, models_to_log)
