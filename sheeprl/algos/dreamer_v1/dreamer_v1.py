"""Dreamer-V1 implementation from [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
Adapted from the original implementation from https://github.com/danijar/dreamer

Written on the shared training loop of `sheeprl.core`: `DreamerV1` says how to build, play and train;
`sheeprl.core.loop.run` does the rest. Plan2Explore (`sheeprl.algos.p2e_dv1`) reuses the player (`SequencePlayer`) and,
to finetune, the two phases of a gradient step (`world_model_learning`, `behaviour_learning`).
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
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch import Tensor, nn
from torch.distributions import Bernoulli, Independent, Normal
from torch.distributions.utils import logits_to_probs
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v1.agent import Actor, MinedojoActor, PlayerDV1, WorldModel, build_agent
from sheeprl.algos.dreamer_v1.loss import actor_loss, critic_loss, reconstruction_loss
from sheeprl.algos.dreamer_v1.utils import add_is_first, compute_lambda_values
from sheeprl.algos.dreamer_v2.dreamer_v2 import actions_dim_of, check_keys
from sheeprl.algos.dreamer_v2.utils import env_buffer_size, prepare_obs, sample_batches, test
from sheeprl.core import Algorithm, EnvRunner, TrainSchedule, TrainState, run
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.fabric import autocast_cache_scope, update
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"


@dataclass
class DreamerV1State(TrainState):
    # Encoder, RSSM, decoder, reward and (optional) continue models
    world_model: WorldModel
    actor: Actor | MinedojoActor
    critic: nn.Module
    world_optimizer: Optimizer
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer


class SequencePlayer:
    """Plays in the environments and writes in the replay buffer the sequences the world model learns from.

    Every row holds an observation, the action that led to it and the reward, `terminated`, `truncated` and `is_first`
    of that step. The first row of every environment holds its first observation, with a zero action and `is_first`.
    When an episode ends, its row holds the final observation, and a further row the first observation of the new
    episode (zero action and reward, `is_first`).

    With `random_warmup`, the actions are uniformly random until `algo.learning_starts`; otherwise they come from
    `policy` (`PlayerDV1`) with its exploration noise, and its recurrent state is reset at the start of every episode.
    """

    def __init__(
        self,
        fabric: Fabric,
        cfg: Dict[str, Any],
        policy: PlayerDV1,
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
        self.started = False

    def step(self, env: EnvRunner, buffer: EnvIndependentReplayBuffer) -> None:
        cfg = self.cfg
        num_envs = env.num_envs
        if not self.started:
            # The first observations start the episodes
            step_data = {k: env.obs[k][np.newaxis] for k in self.obs_keys}
            step_data["terminated"] = np.zeros((1, num_envs, 1))
            step_data["truncated"] = np.zeros((1, num_envs, 1))
            step_data["actions"] = np.zeros((1, num_envs, sum(self.actions_dim)))
            step_data["rewards"] = np.zeros((1, num_envs, 1))
            step_data["is_first"] = np.ones((1, num_envs, 1))
            buffer.add(step_data, validate_args=cfg.buffer.validate_args)
            self.policy.init_states()
            self.started = True

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
            # The exploration noise decays with the policy steps played at the end of this step
            policy_step = env.policy_step + num_envs * self.fabric.world_size
            real_actions = actions = self.policy.get_exploration_actions(
                torch_obs, mask=mask if len(mask) > 0 else None, step=policy_step
            )
            actions = torch.cat(actions, -1).view(num_envs, -1).cpu().numpy()
            if self.is_continuous:
                real_actions = torch.stack(real_actions, -1).cpu().numpy()
            else:
                real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

        step = env.step(real_actions)
        dones = np.logical_or(step.terminated, step.truncated).astype(np.uint8)

        # The observations that follow the actions: for the episodes that have just ended, their last observation
        real_next_obs = copy.deepcopy(step.next_obs)
        if "final_obs" in step.info:
            for idx, final_obs in enumerate(step.info["final_obs"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v
        step_data = {k: real_next_obs[k][np.newaxis] for k in self.obs_keys}
        step_data["terminated"] = step.terminated.reshape((1, num_envs, -1))
        step_data["truncated"] = step.truncated.reshape((1, num_envs, -1))
        step_data["actions"] = actions.reshape((1, num_envs, -1))
        rewards = np.tanh(step.rewards) if cfg.env.clip_rewards else step.rewards
        step_data["rewards"] = rewards.reshape((1, num_envs, -1))
        step_data["is_first"] = np.zeros((1, num_envs, 1))
        buffer.add(step_data, validate_args=cfg.buffer.validate_args)

        # The episodes that have just ended get a row with the first observation of the new episode
        dones_idxes = dones.nonzero()[0].tolist()
        reset_envs = len(dones_idxes)
        if reset_envs > 0:
            reset_data = {k: (step.next_obs[k][dones_idxes])[np.newaxis] for k in self.obs_keys}
            reset_data["terminated"] = np.zeros((1, reset_envs, 1))
            reset_data["truncated"] = np.zeros((1, reset_envs, 1))
            reset_data["actions"] = np.zeros((1, reset_envs, np.sum(self.actions_dim)))
            reset_data["rewards"] = np.zeros((1, reset_envs, 1))
            reset_data["is_first"] = np.ones((1, reset_envs, 1))
            buffer.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)
            self.policy.init_states(reset_envs=dones_idxes)


def world_model_loss(
    world_model: WorldModel,
    data: Dict[str, Tensor],
    *,
    cnn_keys: Tuple[str, ...],
    mlp_keys: Tuple[str, ...],
    stochastic_size: int,
    recurrent_state_size: int,
    use_continues: bool,
    gamma: float,
    kl_free_nats: float,
    kl_regularizer: float,
    continue_scale_factor: float,
    entropies: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """The loss of the world model on a batch of sequences (dynamic learning).

    Returns:
        The loss, the posteriors and the recurrent states of the batch (the starting points of the imagination) and the
        metrics: the terms of the loss, the KL and, with `entropies`, the entropies of the posteriors and of the priors.
    """
    sequence_length, batch_size = data["actions"].shape[:2]
    device = data["actions"].device
    batch_obs = {k: data[k] / 255 - 0.5 for k in cnn_keys}
    batch_obs.update({k: data[k] for k in mlp_keys})

    # initialize the recurrent_state that must be a tuple of tensors (one for GRU or RNN).
    # the dimension of each vector must be (1, batch_size, recurrent_state_size)
    # the recurrent state is the deterministic state (or ht) from the Figure 2c in
    # [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)

    # initialize the posterior that must be of dimension (1, batch_size, stochastic_size)
    # the stochastic state is the stochastic state (or st) from the Figure 2c in
    # [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
    posterior = torch.zeros(1, batch_size, stochastic_size, device=device)

    # The outputs of every step are concatenated at the end of the unroll: writing them in place into
    # preallocated tensors makes the backward pass copy the gradient of the whole tensor at every step
    # recurrent_states will contain all the recurrent states computed during the dynamic learning phase,
    # and its dimension is (sequence_length, batch_size, recurrent_state_size)
    recurrent_states = []
    # posteriors will contain all the posterior states computed during the dynamic learning phase,
    # and its dimension is (sequence_length, batch_size, stochastic_size)
    posteriors = []

    # posteriors_mean and posteriors_std will contain all
    # the actual means and stds of the posterior states respectively,
    # their dimension is (sequence_length, batch_size, stochastic_size)
    posteriors_mean = []
    posteriors_std = []

    # priors_mean and priors_std will contain all
    # the predicted means and stds of the prior states respectively,
    # their dimension is (sequence_length, batch_size, stochastic_size)
    priors_mean = []
    priors_std = []

    embedded_obs = world_model.encoder(batch_obs)

    for i in range(0, sequence_length):
        # one step of dynamic learning, take the posterior state, the recurrent state, the action,
        # and the observation; compute the mean and std of both the posterior and prior state,
        # the new recurrent state and the new posterior state
        recurrent_state, posterior, _, posterior_mean_std, prior_state_mean_std = world_model.rssm.dynamic(
            posterior,
            recurrent_state,
            data["actions"][i : i + 1],
            embedded_obs[i : i + 1],
            data["is_first"][i : i + 1],
        )
        recurrent_states.append(recurrent_state)
        posteriors.append(posterior)
        posteriors_mean.append(posterior_mean_std[0])
        posteriors_std.append(posterior_mean_std[1])
        priors_mean.append(prior_state_mean_std[0])
        priors_std.append(prior_state_mean_std[1])
    recurrent_states = torch.cat(recurrent_states, dim=0)
    posteriors = torch.cat(posteriors, dim=0)
    posteriors_mean = torch.cat(posteriors_mean, dim=0)
    posteriors_std = torch.cat(posteriors_std, dim=0)
    priors_mean = torch.cat(priors_mean, dim=0)
    priors_std = torch.cat(priors_std, dim=0)

    # concatenate the posterior states with the recurrent states on the last dimension
    # latent_states tensor has dimension (sequence_length, batch_size, recurrent_state_size + stochastic_size)
    latent_states = torch.cat((posteriors, recurrent_states), -1)

    # compute predictions for the observations
    decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)
    # compute the distribution of the reconstructed observations
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size observations.shape
    qo = {k: Independent(Normal(rec_obs, 1), len(rec_obs.shape[2:])) for k, rec_obs in decoded_information.items()}

    # compute predictions for the rewards
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the number of rewards
    qr = Independent(Normal(world_model.reward_model(latent_states), 1), 1)

    # compute predictions for terminal steps, if required
    if use_continues:
        qc = Independent(Bernoulli(logits=world_model.continue_model(latent_states)), 1)
        continues_targets = (1 - data["terminated"]) * gamma
    else:
        qc = continues_targets = None

    # compute the distributions of the states (posteriors and priors)
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the stochastic size
    posteriors_dist = Independent(Normal(posteriors_mean, posteriors_std), 1)
    priors_dist = Independent(Normal(priors_mean, priors_std), 1)

    # world model optimization step
    # compute the overall loss of the world model
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        qo,
        batch_obs,
        qr,
        data["rewards"],
        posteriors_dist,
        priors_dist,
        kl_free_nats,
        kl_regularizer,
        qc,
        continues_targets,
        continue_scale_factor,
    )
    metrics = {
        "observation_loss": observation_loss.detach(),
        "reward_loss": reward_loss.detach(),
        "state_loss": state_loss.detach(),
        "continue_loss": continue_loss.detach(),
        "kl": kl.detach(),
    }
    if entropies:
        metrics["post_entropy"] = posteriors_dist.entropy().mean().detach()
        metrics["prior_entropy"] = priors_dist.entropy().mean().detach()
    return rec_loss, posteriors, recurrent_states, metrics


def imagine(
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    posteriors: Tensor,
    recurrent_states: Tensor,
    *,
    horizon: int,
    gamma: float,
    lmbda: float,
    use_continues: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """The trajectories imagined from the latent states of the batch (behaviour learning), with the lambda-values and
    the discounts of the actor and critic losses."""
    stochastic_size = posteriors.shape[-1]
    recurrent_state_size = recurrent_states.shape[-1]
    # Unflatten first 2 dimensions of recurrent and posterior states in order
    # to have all the states on the first dimension.
    # The 1 in the second dimension is needed for the recurrent model in the imagination step,
    # 1 because the agent imagines one state at a time.
    # (1, batch_size * sequence_length, stochastic_size)
    imagined_prior = posteriors.reshape(1, -1, stochastic_size)

    # initialize the recurrent state of the recurrent model with the recurrent states computed
    # during the dynamic learning phase, its shape is (1, batch_size * sequence_length, recurrent_state_size).
    recurrent_state = recurrent_states.reshape(1, -1, recurrent_state_size)

    # starting states for the imagination phase.
    # (1, batch_size * sequence_length, determinisitic_size + stochastic_size)
    imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)

    # the imagined states are concatenated at the end of the imagination,
    # obtaining a tensor of shape (horizon, batch_size * sequence_length, stochastic_size + recurrent_state_size)
    imagined_trajectories = []

    # imagine trajectories in the latent space
    for i in range(horizon):
        # actions tensor has dimension (1, batch_size * sequence_length, num_actions)
        actions = torch.cat(actor(imagined_latent_states.detach())[0], dim=-1)

        # imagination step
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)

        # update current state
        imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories.append(imagined_latent_states)
    imagined_trajectories = torch.cat(imagined_trajectories, dim=0)

    # predict values and rewards
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the number of values/rewards
    predicted_values = critic(imagined_trajectories)
    predicted_rewards = world_model.reward_model(imagined_trajectories)

    # predict the probability that the episode will continue in the imagined states
    if use_continues:
        predicted_continues = logits_to_probs(logits=world_model.continue_model(imagined_trajectories), is_binary=True)
    else:
        predicted_continues = torch.ones_like(predicted_rewards.detach()) * gamma

    # compute the lambda_values, by passing as last values the values of the last imagined state
    # the dimensions of the lambda_values tensor are
    # (horizon, batch_size * sequence_length, recurrent_state_size + stochastic_size)
    lambda_values = compute_lambda_values(
        predicted_rewards,
        predicted_values,
        predicted_continues,
        last_values=predicted_values[-1],
        horizon=horizon,
        lmbda=lmbda,
    )

    # compute the discounts to multiply to the lambda values
    with torch.no_grad():
        # the time steps in Eq. 7 and Eq. 8 of the paper are weighted by the cumulative product of the predicted
        # discount factors, estimated by the continue model, so terms are wighted down based on how likely
        # the imagined trajectory would have ended.
        # Ref. subsection "Learning objectives" of paragraph 3 (Learning Behaviors by Latent Imagination)
        # in [https://doi.org/10.48550/arXiv.1912.01603](https://doi.org/10.48550/arXiv.1912.01603)
        #
        # Suppose the case in which the continue model is not used and gamma = .99
        # predicted_continues.shape = (15, 2500, 1)
        # predicted_continues = [
        #   [ [.99], ..., [.99] ], (2500 columns)
        #   ...
        # ] (15 rows)
        # torch.ones_like(predicted_continues[:1]) = [
        #   [ [1.], ..., [1.] ]
        # ] (1 row and 2500 columns), the discount of the time step 0 is 1.
        # predicted_continues[:-2] = [
        #   [ [.99], ..., [.99] ], (2500 columns)
        #   ...
        # ] (13 rows)
        # torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0) = [
        #   [ [1.], ..., [1.] ], (2500 columns)
        #   [ [.99], ..., [.99] ],
        #   ...,
        #   [ [.99], ..., [.99] ],
        # ] (14 rows), the total number of imagined steps is 15, but one is lost because of the values computation
        # torch.cumprod(torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0) = [
        #   [ [1.], ..., [1.] ], (2500 columns)
        #   [ [.99], ..., [.99] ],
        #   [ [.9801], ..., [.9801] ],
        #   ...,
        #   [ [.8775], ..., [.8775] ],
        # ] (14 rows)
        discount = torch.cumprod(torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0)
    return imagined_trajectories, lambda_values, discount


def value_loss_fn(
    critic: _FabricModule, imagined_trajectories: Tensor, lambda_values: Tensor, discount: Tensor
) -> Tensor:
    """The loss of the critic, on the first H (horizon) imagined states."""
    # Predict the values distribution only for the first H (horizon) imagined states
    # (to match the dimension with the lambda values),
    # it removes the last imagined state in the trajectory
    # because it is used only for computing correclty the lambda values
    qv = Independent(Normal(critic(imagined_trajectories.detach())[:-1], 1), 1)

    # critic optimization step
    # compute the value loss
    # the discount has shape (horizon, seuqence_length * batch_size, 1), so,
    # it is necessary to remove the last dimension to properly match the shapes
    # for the log prob
    return critic_loss(qv, lambda_values.detach(), discount[..., 0])


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    world_optimizer: _FabricOptimizer,
    actor_optimizer: _FabricOptimizer,
    critic_optimizer: _FabricOptimizer,
    data: Dict[str, Tensor],
    cfg: Dict[str, Any],
) -> None:
    """Runs one-step update of the agent.

    The follwing designations are used:
        - recurrent_state: is what is called ht or deterministic state from Figure 2c in
        [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        - stochastic_state: is what is called st or stochastic state from Figure 2c in
        [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            It can be both posterior or prior.
        - latent state: the concatenation of the stochastic and recurrent states on the last dimension.
        - p: the output of the representation model, from Eq. 9 in
        [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        - q: the output of the transition model, from Eq. 9 in
        [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        - qo: the output of the observation model, from Eq. 9 in
        [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        - qr: the output of the reward model, from Eq. 9 in
        [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        - qc: the output of the continue model.
        - qv: the output of the value model (critic), from Eq. 2 in
        [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    In particular, it updates the agent as specified by Algorithm 1 in
    [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    1. Dynamic Learning:
        - Encoder: encode the observations.
        - Recurrent Model: compute the recurrent state from the previous recurrent state,
            the previous stochastic state, and from the previous actions.
        - Transition Model: predict the posterior state from the recurrent state, i.e., the deterministic state or ht.
        - Representation Model: compute the posterior state from the recurrent state and
            from the embedded observations provided by the environment.
        - Observation Model: reconstructs observations from latent states.
        - Reward Model: estimate rewards from the latent states.
        - Update the models
    2. Behaviour Learning:
        - Imagine trajectories in the latent space from each latent state s_t up
        to the horizon H: s'_(t+1), ..., s'_(t+H).
        - Predict rewards and values in the imagined trajectories.
        - Compute lambda targets (Eq. 6 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603))
        - Update the actor and the critic

    Args:
        fabric (Fabric): the fabric instance.
        world_model (WorldModel): the world model wrapped with Fabric.
        actor (_FabricModule): the actor model wrapped with Fabric.
        critic (_FabricModule): the critic model wrapped with Fabric.
        world_optimizer (_FabricOptimizer): the world optimizer.
        actor_optimizer (_FabricOptimizer): the actor optimizer.
        critic_optimizer (_FabricOptimizer): the critic optimizer.
        data (Dict[str, Tensor]): the batch of data to use for training.
        cfg (DictConfig): the configs.
    """
    metrics: Dict[str, Tensor] = {}
    # Every sequence starts from the zero state, as an episode does: its first step is treated as the first one of an
    # episode (its action, which comes from before the sequence, is not seen)
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])
    # The losses are compiled when `algo.compile.enabled` is set
    mark_gradient_step(fabric, cfg)

    # Dynamic Learning
    world_model_cfg = cfg.algo.world_model
    use_continues = bool(world_model_cfg.use_continues and world_model.continue_model)
    # Cast the weights to low precision once for the whole forward pass, not at every step of the unroll
    with autocast_cache_scope(fabric):
        rec_loss, posteriors, recurrent_states, losses = compiled(world_model_loss, fabric, cfg)(
            world_model,
            data,
            cnn_keys=tuple(cfg.algo.cnn_keys.encoder),
            mlp_keys=tuple(cfg.algo.mlp_keys.encoder),
            stochastic_size=world_model_cfg.stochastic_size,
            recurrent_state_size=world_model_cfg.recurrent_model.recurrent_state_size,
            use_continues=use_continues,
            gamma=cfg.algo.gamma,
            kl_free_nats=world_model_cfg.kl_free_nats,
            kl_regularizer=world_model_cfg.kl_regularizer,
            continue_scale_factor=world_model_cfg.continue_scale_factor,
            entropies=not MetricAggregator.disabled,
        )
    world_model_grads = update(
        fabric, rec_loss, world_optimizer, world_model_cfg.clip_gradients, error_if_nonfinite=False
    )

    # Behaviour Learning
    if use_continues:
        # The last step of the sequences could be terminal: the imagination starts from the other ones, as in the
        # official implementation (`Dreamer._imagine_ahead`)
        posteriors, recurrent_states = posteriors[:-1], recurrent_states[:-1]
    with autocast_cache_scope(fabric):
        imagined_trajectories, lambda_values, discount = compiled(imagine, fabric, cfg)(
            world_model,
            actor,
            critic,
            posteriors.detach(),
            recurrent_states.detach(),
            horizon=cfg.algo.horizon,
            gamma=cfg.algo.gamma,
            lmbda=cfg.algo.lmbda,
            use_continues=use_continues,
        )
        # actor optimization step
        # compute the policy loss
        policy_loss = actor_loss(discount * lambda_values)
    actor_grads = update(fabric, policy_loss, actor_optimizer, cfg.algo.actor.clip_gradients, error_if_nonfinite=False)

    with autocast_cache_scope(fabric):
        value_loss = compiled(value_loss_fn, fabric, cfg)(critic, imagined_trajectories, lambda_values, discount)
    critic_grads = update(
        fabric, value_loss, critic_optimizer, cfg.algo.critic.clip_gradients, error_if_nonfinite=False
    )

    # Log metrics
    metrics["Loss/world_model_loss"] = rec_loss.detach()
    metrics["Loss/observation_loss"] = losses["observation_loss"]
    metrics["Loss/reward_loss"] = losses["reward_loss"]
    metrics["Loss/state_loss"] = losses["state_loss"]
    metrics["Loss/continue_loss"] = losses["continue_loss"]
    metrics["State/kl"] = losses["kl"]
    if "post_entropy" in losses:
        metrics["State/post_entropy"] = losses["post_entropy"]
        metrics["State/prior_entropy"] = losses["prior_entropy"]
    metrics["Loss/policy_loss"] = policy_loss.detach()
    metrics["Loss/value_loss"] = value_loss.detach()
    if world_model_grads is not None:
        metrics["Grads/world_model"] = world_model_grads.mean().detach()
    if actor_grads is not None:
        metrics["Grads/actor"] = actor_grads.mean().detach()
    if critic_grads is not None:
        metrics["Grads/critic"] = critic_grads.mean().detach()

    # Reset everything
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)
    return metrics


def build_buffer(fabric: Fabric, cfg: Dict[str, Any], log_dir: str, dry_run_size: int) -> EnvIndependentReplayBuffer:
    """One buffer of sequences per environment."""
    return EnvIndependentReplayBuffer(
        env_buffer_size(fabric, cfg, dry_run_size),
        n_envs=cfg.env.num_envs,
        obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=SequentialReplayBuffer,
        seed=cfg.seed + fabric.global_rank,
    )


def sample_batches_of_iteration(
    algo: Algorithm, buffer: EnvIndependentReplayBuffer, n_steps: int, iteration: int
) -> Iterator[Dict[str, Tensor]]:
    """The batches of the `n_steps` gradient steps of an iteration. Before the last one, `algo.exploration_step` is
    set to the policy steps played at the end of the iteration (`None` before the others): the training steps log the
    amount of exploration noise once per iteration, after its gradient steps."""
    algo.exploration_step = None
    for i, batch in enumerate(sample_batches(algo.fabric, algo.cfg, buffer, n_steps)):
        if i == n_steps - 1:
            algo.exploration_step = iteration * algo.schedule.policy_steps_per_iter
        yield batch


class DreamerV1(Algorithm):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each on its own batch of sequences: the world model, then the
    actor and the critic on trajectories imagined from the batch."""

    off_policy = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        # These arguments cannot be changed
        cfg.env.screen_size = 64
        cfg.env.frame_stack = 1
        # The policy steps at the end of the iteration, set for its last gradient step (see `batches`)
        self.exploration_step: Optional[int] = None

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[DreamerV1State, EnvIndependentReplayBuffer]:
        cfg = self.cfg
        fabric = self.fabric
        self.actions_dim, self.is_continuous = actions_dim_of(action_space)
        check_keys(fabric, cfg, obs_space)

        world_model, actor, critic, self._policy = build_agent(
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
        state = DreamerV1State(
            world_model=world_model,
            actor=actor,
            critic=critic,
            world_optimizer=world_optimizer,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )
        self.schedule = schedule
        return state, build_buffer(fabric, cfg, log_dir, dry_run_size=2)

    def policy(self, state: DreamerV1State) -> PlayerDV1:
        """The policy to play with: it shares its weights with the trained agent (`build_agent`)."""
        return self._policy

    def load_store(self, saved: Any, store: EnvIndependentReplayBuffer) -> EnvIndependentReplayBuffer:
        # A buffer saved before `is_first` was stored
        return add_is_first(super().load_store(saved, store))

    def player(self, state: DreamerV1State) -> SequencePlayer:
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
        self, state: DreamerV1State, buffer: EnvIndependentReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        yield from sample_batches_of_iteration(self, buffer, n_steps, iteration)

    def train_step(self, state: DreamerV1State, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        metrics = train(
            self.fabric,
            state.world_model,
            state.actor,
            state.critic,
            state.world_optimizer,
            state.actor_optimizer,
            state.critic_optimizer,
            batch,
            self.cfg,
        )
        if self.exploration_step is not None:
            metrics["Params/exploration_amount"] = state.actor._get_expl_amount(self.exploration_step)
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = DreamerV1(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"world_model": state.world_model, "actor": state.actor, "critic": state.critic}
        register_model(fabric, log_models, cfg, models_to_log)
