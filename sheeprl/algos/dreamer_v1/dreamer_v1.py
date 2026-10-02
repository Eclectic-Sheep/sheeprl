"""Dreamer-V1 implementation from [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
Adapted from the original implementation from https://github.com/danijar/dreamer

Written on the shared training loop of `sheeprl.core`: `DreamerV1` says how to build, play and train;
`sheeprl.core.loop.run` does the rest. Plan2Explore (`sheeprl.algos.p2e_dv1`) reuses the player (`SequencePlayer`) and
the two phases of a gradient step (`world_model_learning`, `behaviour_learning`).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.distributions import Bernoulli, Independent, Normal
from torch.distributions.utils import logits_to_probs
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v1.agent import Actor, MinedojoActor, PlayerDV1, WorldModel, build_models
from sheeprl.algos.dreamer_v1.loss import actor_loss, critic_loss, reconstruction_loss
from sheeprl.algos.dreamer_v1.utils import compute_lambda_values
from sheeprl.algos.dreamer_v2.dreamer_v2 import (
    actions_dim_of,
    check_keys,
    env_buffer_size,
    sample_batches,
    setup_world_model,
)
from sheeprl.algos.dreamer_v2.utils import prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
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


def world_model_learning(
    fabric: Fabric,
    cfg: Dict[str, Any],
    world_model: WorldModel,
    world_optimizer: Optimizer,
    data: Dict[str, Tensor],
    detach_heads: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """One update of the world model on a batch of sequences (dynamic learning, Eq. 10 in the paper).

    The following designations are used:
        - recurrent_state: the deterministic state (ht) of Figure 2c in
          [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        - stochastic_state: the stochastic state (st) of Figure 2c, posterior or prior.
        - latent state: the concatenation of the stochastic and recurrent states on the last dimension.

    Args:
        detach_heads: the reward and continue models learn from the latent states without changing them (P2E).

    Returns:
        The posteriors and the recurrent states of the batch, the starting points of the imagination, the embedded
        observations and the metrics.
    """
    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    device = fabric.device
    batch_obs = {k: data[k] / 255 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})

    # Every sequence starts from the zero state, as an episode does: its first step is treated as the first one of an
    # episode (its action, which comes from before the sequence, is not seen)
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # The states start from zero at the beginning of every sequence: (1, batch_size, state_size)
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, device=device)
    with autocast(fabric):
        # The outputs of every step are concatenated at the end of the unroll: writing them in place into
        # preallocated tensors makes the backward pass copy the gradient of the whole tensor at every step
        recurrent_states, posteriors = [], []
        # The means and the standard deviations of the posteriors and of the priors
        posteriors_mean, posteriors_std, priors_mean, priors_std = [], [], [], []
        embedded_obs = world_model.encoder(batch_obs)
        for i in range(0, sequence_length):
            # One step of dynamic learning, which takes the posterior state, the recurrent state, the action and the
            # observation and computes the next recurrent and posterior states, and the distributions of the
            # posterior and of the prior. The states are reset at the start of every episode
            recurrent_state, posterior, _, posterior_mean_std, prior_mean_std = world_model.rssm.dynamic(
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
            priors_mean.append(prior_mean_std[0])
            priors_std.append(prior_mean_std[1])
        recurrent_states = torch.cat(recurrent_states, dim=0)
        posteriors = torch.cat(posteriors, dim=0)
        posteriors_mean = torch.cat(posteriors_mean, dim=0)
        posteriors_std = torch.cat(posteriors_std, dim=0)
        priors_mean = torch.cat(priors_mean, dim=0)
        priors_std = torch.cat(priors_std, dim=0)

        # (sequence_length, batch_size, recurrent_state_size + stochastic_size)
        latent_states = torch.cat((posteriors, recurrent_states), -1)

        # The distributions over the reconstructed observations, the rewards and, if required, the terminal steps
        decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)
        qo = {k: Independent(Normal(rec_obs, 1), len(rec_obs.shape[2:])) for k, rec_obs in decoded_information.items()}
        heads_input = latent_states.detach() if detach_heads else latent_states
        qr = Independent(Normal(world_model.reward_model(heads_input), 1), 1)
        if cfg.algo.world_model.use_continues and world_model.continue_model:
            qc = Independent(Bernoulli(logits=world_model.continue_model(heads_input)), 1)
            continues_targets = (1 - data["terminated"]) * cfg.algo.gamma
        else:
            qc = continues_targets = None

        # The distributions of the stochastic states (posteriors and priors)
        posteriors_dist = Independent(Normal(posteriors_mean, posteriors_std), 1)
        priors_dist = Independent(Normal(priors_mean, priors_std), 1)

        # World model optimization step
        rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
            qo,
            batch_obs,
            qr,
            data["rewards"],
            posteriors_dist,
            priors_dist,
            cfg.algo.world_model.kl_free_nats,
            cfg.algo.world_model.kl_regularizer,
            qc,
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
        "State/kl": kl.detach(),
    }
    if not MetricAggregator.disabled:
        metrics["State/post_entropy"] = posteriors_dist.entropy().mean().detach()
        metrics["State/prior_entropy"] = priors_dist.entropy().mean().detach()
    if grads is not None:
        metrics["Grads/world_model"] = grads.mean().detach()
    return posteriors, recurrent_states, embedded_obs, metrics


def behaviour_learning(
    fabric: Fabric,
    cfg: Dict[str, Any],
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    posteriors: Tensor,
    recurrent_states: Tensor,
    reward_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> Dict[str, Optional[Tensor]]:
    """One update of the actor and one of the critic, on trajectories imagined from the latent states of the batch
    (behaviour learning, Algorithm 1 in the paper).

    Args:
        reward_fn: the rewards of the imagined trajectories (from the states and the actions that led to them);
            default: the ones predicted by the reward model.

    Returns:
        The losses (`policy_loss`, `value_loss`), the norms of the gradients before clipping (`actor_grads`,
        `critic_grads`, `None` without clipping), the predicted values (`values`), the lambda-values and the rewards
        of the imagined trajectories.
    """
    stochastic_size = cfg.algo.world_model.stochastic_size
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    with autocast(fabric):
        # The imagination starts from every latent state of the batch, one state at a time:
        # (1, batch_size * sequence_length, stochastic_size + recurrent_state_size)
        imagined_prior = posteriors.detach().reshape(1, -1, stochastic_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
        imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)

        # The imagined states s'_(t+1), ..., s'_(t+H) and the actions that led to them:
        # (horizon, batch_size * sequence_length, ...)
        imagined_trajectories, imagined_actions = [], []
        for i in range(cfg.algo.horizon):
            actions = torch.cat(actor(imagined_latent_states.detach())[0], dim=-1)
            imagined_actions.append(actions)
            imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
            imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories.append(imagined_latent_states)
        imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
        imagined_actions = torch.cat(imagined_actions, dim=0)

        # Predict values, rewards and the probability that the imagined episodes continue
        predicted_values = critic(imagined_trajectories)
        if reward_fn is None:
            predicted_rewards = world_model.reward_model(imagined_trajectories)
        else:
            predicted_rewards = reward_fn(imagined_trajectories, imagined_actions)
        if cfg.algo.world_model.use_continues and world_model.continue_model:
            predicted_continues = logits_to_probs(
                logits=world_model.continue_model(imagined_trajectories), is_binary=True
            )
        else:
            predicted_continues = torch.ones_like(predicted_rewards.detach()) * cfg.algo.gamma

        # The lambda-values (Eq. 6 in the paper), bootstrapped by the value of the last imagined state:
        # (horizon - 1, batch_size * sequence_length, 1)
        lambda_values = compute_lambda_values(
            predicted_rewards,
            predicted_values,
            predicted_continues,
            last_values=predicted_values[-1],
            horizon=cfg.algo.horizon,
            lmbda=cfg.algo.lmbda,
        )

        # The steps of the objectives (Eq. 7 and 8 in the paper) are weighted by the cumulative product of the
        # predicted discounts, so that the steps of the imagined trajectories that likely ended count less: the
        # first step has discount 1, and the last imagined state is lost in the lambda-values
        with torch.no_grad():
            discount = torch.cumprod(
                torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0
            )

        # Actor optimization step
        policy_loss = actor_loss(discount * lambda_values)
    actor_grads = update(
        fabric,
        policy_loss,
        actor_optimizer,
        max_grad_norm=cfg.algo.actor.clip_gradients or 0.0,
        error_if_nonfinite=False,
    )

    with autocast(fabric):
        # The values of the first H - 1 imagined states, to match the lambda-values: the last one bootstraps them
        qv = Independent(Normal(critic(imagined_trajectories.detach())[:-1], 1), 1)

        # Critic optimization step: the log-probabilities remove the last dimension of the discount
        value_loss = critic_loss(qv, lambda_values.detach(), discount[..., 0])
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
        "values": predicted_values.detach(),
        "lambda_values": lambda_values.detach(),
        "rewards": predicted_rewards.detach(),
    }


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

        world_model, actor, critic = build_models(self.actions_dim, self.is_continuous, cfg, obs_space)
        world_model = setup_world_model(fabric, world_model)
        actor = setup_module(fabric, actor)
        critic = setup_module(fabric, critic)

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
        """The policy to play with: it shares its modules (and so its weights) with the trained agent."""
        cfg = self.cfg
        return PlayerDV1(
            state.world_model.encoder,
            state.world_model.rssm.recurrent_model,
            state.world_model.rssm.representation_model,
            state.actor,
            self.actions_dim,
            cfg.env.num_envs,
            cfg.algo.world_model.stochastic_size,
            cfg.algo.world_model.recurrent_model.recurrent_state_size,
            self.fabric.device,
            min_std=cfg.algo.world_model.min_std,
        )

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
        cfg = self.cfg
        posteriors, recurrent_states, _, metrics = world_model_learning(
            self.fabric, cfg, state.world_model, state.world_optimizer, batch
        )
        behaviour = behaviour_learning(
            self.fabric,
            cfg,
            state.world_model,
            state.actor,
            state.critic,
            state.actor_optimizer,
            state.critic_optimizer,
            posteriors,
            recurrent_states,
        )
        metrics["Loss/policy_loss"] = behaviour["policy_loss"]
        metrics["Loss/value_loss"] = behaviour["value_loss"]
        if behaviour["actor_grads"] is not None:
            metrics["Grads/actor"] = behaviour["actor_grads"]
        if behaviour["critic_grads"] is not None:
            metrics["Grads/critic"] = behaviour["critic_grads"]
        if self.exploration_step is not None:
            metrics["Params/exploration_amount"] = state.actor._get_expl_amount(self.exploration_step)
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = DreamerV1(fabric, cfg)
    state, log_dir = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"world_model": state.world_model, "actor": state.actor, "critic": state.critic}
        register_model(fabric, log_models, cfg, models_to_log)
