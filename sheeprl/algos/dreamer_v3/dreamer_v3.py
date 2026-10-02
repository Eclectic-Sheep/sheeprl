"""Dreamer-V3 implementation from [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
Adapted from the original implementation from https://github.com/danijar/dreamerv3

Written on the shared training loop of `sheeprl.core`: `DreamerV3` says how to build, play and train;
`sheeprl.core.loop.run` does the rest. Plan2Explore (`sheeprl.algos.p2e_dv3`) reuses the player
(`SequencePlayer`) and the two phases of a gradient step (`world_model_learning`, `behaviour_learning`).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, OneHotCategorical
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v3.agent import Actor, MinedojoActor, PlayerDV3, build_models
from sheeprl.algos.dreamer_v3.loss import reconstruction_loss
from sheeprl.algos.dreamer_v3.utils import Moments, compute_lambda_values, prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.distribution import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    TwoHotEncodingDistribution,
)
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
        The posteriors and the recurrent states of the batch, the starting points of the imagination, and the metrics.
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

    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    with autocast(fabric):
        # Embed observations from the environment
        embedded_obs = world_model.encoder(batch_obs)

        # The outputs of every step are concatenated at the end of the unroll: writing them in place into
        # preallocated tensors makes the backward pass copy the gradient of the whole tensor at every step
        recurrent_states, priors_logits = [], []
        if cfg.algo.world_model.decoupled_rssm:
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
                recurrent_states.append(recurrent_state)
                priors_logits.append(prior_logits)
        else:
            posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
            posteriors, posteriors_logits = [], []
            for i in range(0, sequence_length):
                recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
                    posterior,
                    recurrent_state,
                    batch_actions[i : i + 1],
                    embedded_obs[i : i + 1],
                    data["is_first"][i : i + 1],
                )
                recurrent_states.append(recurrent_state)
                priors_logits.append(prior_logits)
                posteriors.append(posterior)
                posteriors_logits.append(posterior_logits)
            posteriors = torch.cat(posteriors, dim=0)
            posteriors_logits = torch.cat(posteriors_logits, dim=0)
        recurrent_states = torch.cat(recurrent_states, dim=0)
        priors_logits = torch.cat(priors_logits, dim=0)
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
            cfg.algo.world_model.kl_dynamic,
            cfg.algo.world_model.kl_representation,
            cfg.algo.world_model.kl_free_nats,
            cfg.algo.world_model.kl_regularizer,
            pc,
            continues_targets,
            cfg.algo.world_model.continue_scale_factor,
        )
    grads = update(
        fabric,
        rec_loss,
        world_optimizer,
        max_grad_norm=cfg.algo.world_model.clip_gradients or 0.0,
        error_if_nonfinite=False,
    )

    metrics = {
        "Loss/world_model_loss": rec_loss.detach(),
        "Loss/observation_loss": observation_loss.detach(),
        "Loss/reward_loss": reward_loss.detach(),
        "Loss/state_loss": state_loss.detach(),
        "Loss/continue_loss": continue_loss.detach(),
        "State/kl": kl.mean().detach(),
    }
    if not MetricAggregator.disabled:
        metrics["State/post_entropy"] = (
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach()
        )
        metrics["State/prior_entropy"] = (
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach()
        )
    if grads is not None:
        metrics["Grads/world_model"] = grads.mean().detach()
    return posteriors, recurrent_states, metrics


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
) -> Dict[str, Optional[Tensor]]:
    """One update of the actor and one of the critic, on trajectories imagined from the latent states of the batch
    (behaviour learning, Eq. 10 and 11 in the paper).

    Returns:
        The losses (`policy_loss`, `value_loss`) and the norms of the gradients before clipping (`actor_grads`,
        `critic_grads`, `None` without clipping).
    """
    stoch_state_size = cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    with autocast(fabric):
        imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
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
        for i in range(1, cfg.algo.horizon + 1):
            imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
            imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
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
    actor_grads = update(
        fabric,
        policy_loss,
        actor_optimizer,
        max_grad_norm=cfg.algo.actor.clip_gradients or 0.0,
        error_if_nonfinite=False,
    )

    with autocast(fabric):
        # Predict the values
        qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
        predicted_target_values = TwoHotEncodingDistribution(
            target_critic(imagined_trajectories.detach()[:-1]), dims=1
        ).mean

        # Critic optimization. Eq. 10 in the paper
        value_loss = -qv.log_prob(lambda_values.detach())
        value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
        value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))
    critic_grads = update(
        fabric,
        value_loss,
        critic_optimizer,
        max_grad_norm=cfg.algo.critic.clip_gradients or 0.0,
        error_if_nonfinite=False,
    )
    return {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "actor_grads": None if actor_grads is None else actor_grads.mean().detach(),
        "critic_grads": None if critic_grads is None else critic_grads.mean().detach(),
    }


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

        world_model, actor, critic = build_models(fabric.device, self.actions_dim, self.is_continuous, cfg, obs_space)
        world_model.encoder = setup_module(fabric, world_model.encoder)
        world_model.observation_model = setup_module(fabric, world_model.observation_model)
        world_model.reward_model = setup_module(fabric, world_model.reward_model)
        world_model.rssm.recurrent_model = setup_module(fabric, world_model.rssm.recurrent_model)
        world_model.rssm.representation_model = setup_module(fabric, world_model.rssm.representation_model)
        world_model.rssm.transition_model = setup_module(fabric, world_model.rssm.transition_model)
        if world_model.continue_model:
            world_model.continue_model = setup_module(fabric, world_model.continue_model)
        actor = setup_module(fabric, actor)
        critic = setup_module(fabric, critic)
        target_critic = setup_module(fabric, copy.deepcopy(critic.module))

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
            cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 2,
            n_envs=cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            buffer_cls=SequentialReplayBuffer,
            seed=cfg.seed + fabric.global_rank,
        )
        self.schedule = schedule
        return state, buffer

    def policy(self, state: DreamerV3State) -> PlayerDV3:
        """The policy to play with: it shares its modules (and so its weights) with the trained agent."""
        cfg = self.cfg
        return PlayerDV3(
            state.world_model.encoder,
            state.world_model.rssm,
            state.actor,
            self.actions_dim,
            cfg.env.num_envs,
            cfg.algo.world_model.stochastic_size,
            cfg.algo.world_model.recurrent_model.recurrent_state_size,
            self.fabric.device,
            discrete_size=cfg.algo.world_model.discrete_size,
        )

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
        cfg = self.cfg
        sample = buffer.sample_tensors(
            cfg.algo.per_rank_batch_size,
            sequence_length=cfg.algo.per_rank_sequence_length,
            n_samples=n_steps,
            dtype=None,
            device=self.fabric.device,
            from_numpy=cfg.buffer.from_numpy,
        )  # [N_Steps, Sequence_Length, Batch_Size, ...]
        for i in range(n_steps):
            yield {k: v[i].float() for k, v in sample.items()}

    def train_step(self, state: DreamerV3State, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg
        # The target critic follows the critic: every `critic.per_rank_target_network_update_freq` gradient steps, an
        # exponential moving average with `critic.tau`; at the first gradient step, a copy
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            tau = 1 if step == 0 else cfg.algo.critic.tau
            for cp, tcp in zip(state.critic.module.parameters(), state.target_critic.parameters()):
                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)

        posteriors, recurrent_states, metrics = world_model_learning(
            self.fabric, cfg, state.world_model, state.world_optimizer, batch
        )
        behaviour = behaviour_learning(
            self.fabric,
            cfg,
            state.world_model,
            state.actor,
            state.critic,
            state.target_critic,
            state.actor_optimizer,
            state.critic_optimizer,
            state.moments,
            posteriors,
            recurrent_states,
            batch["terminated"],
            self.is_continuous,
            self.actions_dim,
        )
        metrics["Loss/policy_loss"] = behaviour["policy_loss"]
        metrics["Loss/value_loss"] = behaviour["value_loss"]
        if behaviour["actor_grads"] is not None:
            metrics["Grads/actor"] = behaviour["actor_grads"]
        if behaviour["critic_grads"] is not None:
            metrics["Grads/critic"] = behaviour["critic_grads"]
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = DreamerV3(fabric, cfg)
    state, log_dir = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir, greedy=False)

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
