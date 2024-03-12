"""Dreamer-V3 implementation from [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
Adapted from the original implementation from https://github.com/danijar/dreamerv3
"""

from __future__ import annotations

import copy
import os
import warnings
from functools import partial
from typing import Any, Dict, Sequence

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.distributions import Distribution, Independent
from torch.optim import Optimizer
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_v3.agent import PlayerDV3, WorldModel, build_agent
from sheeprl.algos.dreamer_v3.loss import reconstruction_loss
from sheeprl.algos.dreamer_v3.utils import Moments, compute_lambda_values, test
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.utils.distribution import (
    BernoulliSafeMode,
    MSEDistribution,
    OneHotCategoricalValidateArgs,
    SymlogDistribution,
    TwoHotEncodingDistribution,
)
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import polynomial_decay, save_configs

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"


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
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
    moments: Moments,
) -> None:
    """Runs one-step update of the agent.

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
        aggregator (MetricAggregator, optional): the aggregator to print the metrics.
        cfg (DictConfig): the configs.
        is_continuous (bool): whether or not the environment is continuous.
        actions_dim (Sequence[int]): the actions dimension.
        moments (Moments): the moments for normalizing the lambda values.
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

    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    validate_args = cfg.distribution.validate_args
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    priors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)

    # Embed observations from the environment
    embedded_obs = world_model.encoder(batch_obs)

    if cfg.algo.decoupled_rssm:
        posteriors_logits, posteriors = world_model.rssm._representation(embedded_obs)
        for i in range(0, sequence_length):
            if i == 0:
                posterior = torch.zeros_like(posteriors[:1])
            else:
                posterior = posteriors[i - 1 : i]
            recurrent_state, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
    else:
        posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
        posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device)
        posteriors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)
        for i in range(0, sequence_length):
            recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                embedded_obs[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

    # Compute predictions for the observations
    reconstructed_obs: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)

    # Compute the distribution over the reconstructed observations
    po = {
        k: MSEDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
        for k in cfg.algo.cnn_keys.decoder
    }
    po.update(
        {
            k: SymlogDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
            for k in cfg.algo.mlp_keys.decoder
        }
    )

    # Compute the distribution over the rewards
    pr = TwoHotEncodingDistribution(world_model.reward_model(latent_states), dims=1)

    # Compute the distribution over the terminal steps, if required
    pc = Independent(
        BernoulliSafeMode(logits=world_model.continue_model(latent_states), validate_args=validate_args),
        1,
        validate_args=validate_args,
    )
    continue_targets = 1 - data["dones"]

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

    # World model optimization step. Eq. 4 in the paper
    world_optimizer.zero_grad(set_to_none=True)
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        po,
        batch_obs,
        pr,
        data["rewards"],
        priors_logits,
        posteriors_logits,
        cfg.algo.world_model.kl_dynamic,
        cfg.algo.world_model.kl_representation,
        cfg.algo.world_model.kl_free_nats,
        cfg.algo.world_model.kl_regularizer,
        pc,
        continue_targets,
        cfg.algo.world_model.continue_scale_factor,
        validate_args=validate_args,
    )
    fabric.backward(rec_loss)
    world_model_grads = None
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_model_grads = fabric.clip_gradients(
            module=world_model,
            optimizer=world_optimizer,
            max_norm=cfg.algo.world_model.clip_gradients,
            error_if_nonfinite=False,
        )
    world_optimizer.step()

    # Behaviour Learning
    imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    imagined_trajectories = torch.empty(
        cfg.algo.horizon + 1,
        batch_size * sequence_length,
        stoch_state_size + recurrent_state_size,
        device=device,
    )
    imagined_trajectories[0] = imagined_latent_state
    imagined_actions = torch.empty(
        cfg.algo.horizon + 1,
        batch_size * sequence_length,
        data["actions"].shape[-1],
        device=device,
    )
    actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    imagined_actions[0] = actions

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
    for i in range(1, cfg.algo.horizon + 1):
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions

    # Predict values, rewards and continues
    predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean
    predicted_rewards = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean
    continues = Independent(
        BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories), validate_args=validate_args),
        1,
        validate_args=validate_args,
    ).mode
    true_done = (1 - data["dones"]).flatten().reshape(1, -1, 1)
    continues = torch.cat((true_done, continues[1:]))

    # Estimate lambda-values
    lambda_values = compute_lambda_values(
        predicted_rewards[1:],
        predicted_values[1:],
        continues[1:] * cfg.algo.gamma,
        lmbda=cfg.algo.lmbda,
    )

    # Compute the discounts to multiply the lambda values to
    with torch.no_grad():
        discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

    # Actor optimization step. Eq. 11 from the paper
    # Given the following diagram, with H=3
    # Actions:          [a'0]    [a'1]    [a'2]    a'3
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:       [z0] -> [z'1] -> [z'2] ->  z'3
    # Values:       [v'0]   [v'1]    [v'2]     v'3
    # Lambda-values:        [l'1]    [l'2]    [l'3]
    # Entropies:    [e'0]   [e'1]    [e'2]
    actor_optimizer.zero_grad(set_to_none=True)
    policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]

    baseline = predicted_values[:-1]
    offset, invscale = moments(lambda_values, fabric)
    normed_lambda_values = (lambda_values - offset) / invscale
    normed_baseline = (baseline - offset) / invscale
    advantage = normed_lambda_values - normed_baseline
    if is_continuous:
        objective = advantage
    else:
        objective = (
            torch.stack(
                [
                    p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                    for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                ],
                dim=-1,
            ).sum(dim=-1)
            * advantage.detach()
        )
    try:
        entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
    except NotImplementedError:
        entropy = torch.zeros_like(objective)
    policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
    fabric.backward(policy_loss)
    actor_grads = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads = fabric.clip_gradients(
            module=actor, optimizer=actor_optimizer, max_norm=cfg.algo.actor.clip_gradients, error_if_nonfinite=False
        )
    actor_optimizer.step()

    # Predict the values
    qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
    predicted_target_values = TwoHotEncodingDistribution(
        target_critic(imagined_trajectories.detach()[:-1]), dims=1
    ).mean

    # Critic optimization. Eq. 10 in the paper
    critic_optimizer.zero_grad(set_to_none=True)
    value_loss = -qv.log_prob(lambda_values.detach())
    value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

    fabric.backward(value_loss)
    critic_grads = None
    if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
        critic_grads = fabric.clip_gradients(
            module=critic, optimizer=critic_optimizer, max_norm=cfg.algo.critic.clip_gradients, error_if_nonfinite=False
        )
    critic_optimizer.step()

    # Log metrics
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/world_model_loss", rec_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.detach())
        aggregator.update("Loss/reward_loss", reward_loss.detach())
        aggregator.update("Loss/state_loss", state_loss.detach())
        aggregator.update("Loss/continue_loss", continue_loss.detach())
        aggregator.update("State/kl", kl.mean().detach())
        aggregator.update(
            "State/post_entropy",
            Independent(
                OneHotCategoricalValidateArgs(logits=posteriors_logits.detach(), validate_args=validate_args),
                1,
                validate_args=validate_args,
            )
            .entropy()
            .mean()
            .detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(
                OneHotCategoricalValidateArgs(logits=priors_logits.detach(), validate_args=validate_args),
                1,
                validate_args=validate_args,
            )
            .entropy()
            .mean()
            .detach(),
        )
        aggregator.update("Loss/policy_loss", policy_loss.detach())
        aggregator.update("Loss/value_loss", value_loss.detach())
        if world_model_grads:
            aggregator.update("Grads/world_model", world_model_grads.mean().detach())
        if actor_grads:
            aggregator.update("Grads/actor", actor_grads.mean().detach())
        if critic_grads:
            aggregator.update("Grads/critic", critic_grads.mean().detach())

    # Reset everything
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # These arguments cannot be changed
    cfg.env.frame_stack = -1
    if 2 ** int(np.log2(cfg.env.screen_size)) != cfg.env.screen_size:
        raise ValueError(f"The screen size must be a power of 2, got: {cfg.env.screen_size}")

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
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

    world_model, actor, critic, target_critic = build_agent(
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
    player = PlayerDV3(
        world_model.encoder.module,
        world_model.rssm,
        actor.module,
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        fabric.device,
        discrete_size=cfg.algo.world_model.discrete_size,
        decoupled_rssm=cfg.algo.decoupled_rssm,
    )

    # Optimizers
    world_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
    )
    actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters(), _convert_="all")
    critic_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic.parameters(), _convert_="all")
    if cfg.checkpoint.resume_from:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
        world_optimizer, actor_optimizer, critic_optimizer
    )
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

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 2
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=SequentialReplayBuffer,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], EnvIndependentReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")
    expl_decay_steps = state["expl_decay_steps"] if cfg.checkpoint.resume_from else 0

    # Global variables
    train_step = 0
    last_train = 0
    start_step = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["update"] // fabric.world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * fabric.world_size)
    updates_before_training = cfg.algo.train_every // policy_steps_per_update
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    max_step_expl_decay = cfg.algo.actor.max_step_expl_decay // (cfg.algo.per_rank_gradient_steps * fabric.world_size)
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
        actor.expl_amount = polynomial_decay(
            expl_decay_steps,
            initial=cfg.algo.actor.expl_amount,
            final=cfg.algo.actor.expl_min,
            max_decay_steps=max_step_expl_decay,
        )
        if not cfg.buffer.checkpoint:
            learning_starts += start_step

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    # Get the first environment observation and start the optimization
    step_data = {}
    obs = envs.reset(seed=cfg.seed)[0]
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["dones"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["dones"])
    player.init_states()

    per_rank_gradient_steps = 0
    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * world_size

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
            # Sample an action given the observation received by the environment
            if (
                update <= learning_starts
                and cfg.checkpoint.resume_from is None
                and "minedojo" not in cfg.env.wrapper._target_.lower()
            ):
                real_actions = actions = np.array(envs.action_space.sample())
                if not is_continuous:
                    actions = np.concatenate(
                        [
                            F.one_hot(torch.as_tensor(act), act_dim).numpy()
                            for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
                        ],
                        axis=-1,
                    )
            else:
                with torch.no_grad():
                    preprocessed_obs = {}
                    for k, v in obs.items():
                        preprocessed_obs[k] = torch.as_tensor(v[np.newaxis], dtype=torch.float32, device=device)
                        if k in cfg.algo.cnn_keys.encoder:
                            preprocessed_obs[k] = preprocessed_obs[k] / 255.0 - 0.5
                    mask = {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None
                    real_actions = actions = player.get_exploration_action(preprocessed_obs, mask)
                    actions = torch.cat(actions, -1).cpu().numpy()
                    if is_continuous:
                        real_actions = torch.cat(real_actions, dim=-1).cpu().numpy()
                    else:
                        real_actions = (
                            torch.cat([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                        )

            step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
            rb.add(step_data, validate_args=cfg.buffer.validate_args)

            next_obs, rewards, dones, truncated, infos = envs.step(real_actions.reshape(envs.action_space.shape))
            dones = np.logical_or(dones, truncated).astype(np.uint8)

        step_data["is_first"] = np.zeros_like(step_data["dones"])
        if "restart_on_exception" in infos:
            for i, agent_roe in enumerate(infos["restart_on_exception"]):
                if agent_roe and not dones[i]:
                    last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                    rb.buffer[i]["dones"][last_inserted_idx] = np.ones_like(rb.buffer[i]["dones"][last_inserted_idx])
                    rb.buffer[i]["is_first"][last_inserted_idx] = np.zeros_like(
                        rb.buffer[i]["is_first"][last_inserted_idx]
                    )
                    step_data["is_first"][i] = np.ones_like(step_data["is_first"][i])

        if cfg.metric.log_level > 0 and "final_info" in infos:
            for i, agent_ep_info in enumerate(infos["final_info"]):
                if agent_ep_info is not None:
                    ep_rew = agent_ep_info["episode"]["r"]
                    ep_len = agent_ep_info["episode"]["l"]
                    if aggregator and not aggregator.disabled:
                        aggregator.update("Rewards/rew_avg", ep_rew)
                        aggregator.update("Game/ep_len_avg", ep_len)
                    fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Save the real next observation
        real_next_obs = copy.deepcopy(next_obs)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v

        for k in obs_keys:
            step_data[k] = next_obs[k][np.newaxis]

        # next_obs becomes the new obs
        obs = next_obs

        rewards = rewards.reshape((1, cfg.env.num_envs, -1))
        step_data["dones"] = dones.reshape((1, cfg.env.num_envs, -1))
        step_data["rewards"] = clip_rewards_fn(rewards)

        dones_idxes = dones.nonzero()[0].tolist()
        reset_envs = len(dones_idxes)
        if reset_envs > 0:
            reset_data = {}
            for k in obs_keys:
                reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
            reset_data["dones"] = np.ones((1, reset_envs, 1))
            reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
            reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
            reset_data["is_first"] = np.zeros_like(reset_data["dones"])
            rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

            # Reset already inserted step data
            step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
            step_data["dones"][:, dones_idxes] = np.zeros_like(step_data["dones"][:, dones_idxes])
            step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
            player.init_states(dones_idxes)

        updates_before_training -= 1

        # Train the agent
        if update >= learning_starts and updates_before_training <= 0:
            local_data = rb.sample_tensors(
                cfg.algo.per_rank_batch_size,
                sequence_length=cfg.algo.per_rank_sequence_length,
                n_samples=(
                    cfg.algo.per_rank_pretrain_steps if update == learning_starts else cfg.algo.per_rank_gradient_steps
                ),
                dtype=None,
                device=fabric.device,
                from_numpy=cfg.buffer.from_numpy,
            )
            with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                for i in range(next(iter(local_data.values())).shape[0]):
                    if per_rank_gradient_steps % cfg.algo.critic.target_network_update_freq == 0:
                        tau = 1 if per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                        for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
                            tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                    batch = {k: v[i].float() for k, v in local_data.items()}
                    train(
                        fabric,
                        world_model,
                        actor,
                        critic,
                        target_critic,
                        world_optimizer,
                        actor_optimizer,
                        critic_optimizer,
                        batch,
                        aggregator,
                        cfg,
                        is_continuous,
                        actions_dim,
                        moments,
                    )
                    per_rank_gradient_steps += 1
                train_step += world_size
            updates_before_training = cfg.algo.train_every // policy_steps_per_update
            if cfg.algo.actor.expl_decay:
                expl_decay_steps += 1
                actor.expl_amount = polynomial_decay(
                    expl_decay_steps,
                    initial=cfg.algo.actor.expl_amount,
                    final=cfg.algo.actor.expl_min,
                    max_decay_steps=max_step_expl_decay,
                )
            if aggregator and not aggregator.disabled:
                aggregator.update("Params/exploration_amount", actor.expl_amount)

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or update == num_updates):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint Model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            update == num_updates and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "world_model": world_model.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "expl_decay_steps": expl_decay_steps,
                "moments": moments.state_dict(),
                "update": update * fabric.world_size,
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
        test(player, fabric, cfg, log_dir, sample_actions=True)

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
