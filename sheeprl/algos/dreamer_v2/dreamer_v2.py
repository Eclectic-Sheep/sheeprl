"""Dreamer-V2 implementation from [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
Adapted from the original implementation from https://github.com/danijar/dreamerv2

Written on the shared training loop of `sheeprl.core`: `DreamerV2` says how to build, play and train;
`sheeprl.core.loop.run` does the rest. Plan2Explore (`sheeprl.algos.p2e_dv2`) reuses the player (`SequencePlayer`) and
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
from torch.distributions import Bernoulli, Distribution, Independent, Normal, OneHotCategorical
from torch.distributions.utils import logits_to_probs
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import Actor, MinedojoActor, PlayerDV2, WorldModel, build_models
from sheeprl.algos.dreamer_v2.loss import reconstruction_loss
from sheeprl.algos.dreamer_v2.utils import compute_lambda_values, prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer, SequentialReplayBuffer
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"


@dataclass
class DreamerV2State(TrainState):
    # Encoder, RSSM, decoder, reward and (optional) continue models
    world_model: WorldModel
    actor: Actor | MinedojoActor
    critic: nn.Module
    # Copy of the critic, which estimates the values of the imagined trajectories
    target_critic: nn.Module
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
    `policy` (`PlayerDV2`), whose recurrent state is reset at the start of every episode.
    """

    def __init__(
        self,
        fabric: Fabric,
        cfg: Dict[str, Any],
        policy: PlayerDV2,
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
        # The row written at the last step; created, and written, from the first observations of the environments
        self.step_data: Optional[Dict[str, np.ndarray]] = None

    def step(self, env: EnvRunner, buffer: EnvIndependentReplayBuffer | EpisodeBuffer) -> None:
        cfg = self.cfg
        num_envs = env.num_envs
        if self.step_data is None:
            # The first observations start the episodes (in a dry run they also end them, for the episode buffer)
            self.step_data = {k: env.obs[k][np.newaxis] for k in self.obs_keys}
            self.step_data["terminated"] = np.zeros((1, num_envs, 1))
            self.step_data["truncated"] = np.zeros((1, num_envs, 1))
            if cfg.dry_run:
                self.step_data["truncated"] = self.step_data["truncated"] + 1
                self.step_data["terminated"] = self.step_data["terminated"] + 1
            self.step_data["actions"] = np.zeros((1, num_envs, sum(self.actions_dim)))
            self.step_data["rewards"] = np.zeros((1, num_envs, 1))
            self.step_data["is_first"] = np.ones_like(self.step_data["terminated"])
            buffer.add(self.step_data, validate_args=cfg.buffer.validate_args)
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
            actions = torch.cat(actions, -1).view(num_envs, -1).cpu().numpy()
            if self.is_continuous:
                real_actions = torch.stack(real_actions, -1).cpu().numpy()
            else:
                real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

        # The row of this step starts an episode when the previous one ended it
        step_data["is_first"] = copy.deepcopy(np.logical_or(step_data["terminated"], step_data["truncated"]))
        step = env.step(real_actions)
        dones = np.logical_or(step.terminated, step.truncated).astype(np.uint8)
        if cfg.dry_run and cfg.buffer.type.lower() == "episode":
            dones = np.ones_like(dones)

        # The observations that follow the actions: for the episodes that have just ended, their last observation
        real_next_obs = copy.deepcopy(step.next_obs)
        if "final_obs" in step.info:
            for idx, final_obs in enumerate(step.info["final_obs"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v
        for k in self.obs_keys:
            step_data[k] = real_next_obs[k][np.newaxis]
        step_data["terminated"] = step.terminated.reshape((1, num_envs, -1))
        step_data["truncated"] = step.truncated.reshape((1, num_envs, -1))
        step_data["actions"] = actions.reshape((1, num_envs, -1))
        rewards = np.tanh(step.rewards) if cfg.env.clip_rewards else step.rewards
        step_data["rewards"] = rewards.reshape((1, num_envs, -1))
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
            reset_data["is_first"] = np.ones_like(reset_data["terminated"])
            buffer.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)
            # The next row of those envs doesn't start an episode: the reset row did
            for d in dones_idxes:
                step_data["terminated"][0, d] = np.zeros_like(step_data["terminated"][0, d])
                step_data["truncated"][0, d] = np.zeros_like(step_data["truncated"][0, d])
            self.policy.init_states(dones_idxes)


def world_model_learning(
    fabric: Fabric,
    cfg: Dict[str, Any],
    world_model: WorldModel,
    world_optimizer: Optimizer,
    data: Dict[str, Tensor],
    detach_heads: bool = False,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """One update of the world model on a batch of sequences (dynamic learning, Eq. 2 in the paper).

    Args:
        detach_heads: the reward and continue models learn from the latent states without changing them (P2E).

    Returns:
        The posteriors and the recurrent states of the batch, the starting points of the imagination, and the metrics.
    """
    # The environment interaction goes like this:
    # Actions:       0   a1       a2       a3
    #                    ^ \      ^ \      ^ \
    #                   /   \    /   \    /   \
    #                  /     v  /     v  /     v
    # Observations:  o0       o1       o2      o3
    # Rewards:       0        r1       r2      r3
    # Dones:         0        d1       d2      d3
    # Is-first       1        i1       i2      i3
    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device
    batch_obs = {k: data[k] / 255 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})

    # Given how the environment interaction works, we assume that the first element in a sequence
    # is the first one, as if the environment has been reset
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
    with autocast(fabric):
        # The outputs of every step are concatenated at the end of the unroll: writing them in place into
        # preallocated tensors makes the backward pass copy the gradient of the whole tensor at every step
        recurrent_states, priors_logits, posteriors, posteriors_logits = [], [], [], []
        embedded_obs = world_model.encoder(batch_obs)
        for i in range(0, sequence_length):
            # One step of dynamic learning, which take the posterior state, the recurrent state, the action
            # and the observation and compute the next recurrent, prior and posterior states
            recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                data["actions"][i : i + 1],
                embedded_obs[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states.append(recurrent_state)
            priors_logits.append(prior_logits)
            posteriors.append(posterior)
            posteriors_logits.append(posterior_logits)
        recurrent_states = torch.cat(recurrent_states, dim=0)
        priors_logits = torch.cat(priors_logits, dim=0)
        posteriors = torch.cat(posteriors, dim=0)
        posteriors_logits = torch.cat(posteriors_logits, dim=0)

        # (sequence_length, batch_size, recurrent_state_size + stochastic_size * discrete_size)
        latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

        # The distributions over the reconstructed observations, the rewards and, if required, the terminal steps
        decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)
        po = {k: Independent(Normal(rec_obs, 1), len(rec_obs.shape[2:])) for k, rec_obs in decoded_information.items()}
        heads_input = latent_states.detach() if detach_heads else latent_states
        pr = Independent(Normal(world_model.reward_model(heads_input), 1), 1)
        if cfg.algo.world_model.use_continues and world_model.continue_model:
            pc = Independent(Bernoulli(logits=world_model.continue_model(heads_input)), 1)
            continues_targets = (1 - data["terminated"]) * cfg.algo.gamma
        else:
            pc = continues_targets = None

        # Reshape posterior and prior logits to shape [T, B, 32, 32]
        priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
        posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

        # World model optimization step
        rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
            po,
            batch_obs,
            pr,
            data["rewards"],
            priors_logits,
            posteriors_logits,
            cfg.algo.world_model.kl_balancing_alpha,
            cfg.algo.world_model.kl_free_nats,
            cfg.algo.world_model.kl_free_avg,
            cfg.algo.world_model.kl_regularizer,
            pc,
            continues_targets,
            cfg.algo.world_model.discount_scale_factor,
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
    posteriors: Tensor,
    recurrent_states: Tensor,
    terminated: Tensor,
    is_continuous: bool,
    actions_dim: Sequence[int],
    objective_mix: Optional[float],
    reward_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> Dict[str, Optional[Tensor]]:
    """One update of the actor and one of the critic, on trajectories imagined from the latent states of the batch
    (behaviour learning, Eq. 5 and 6 in the paper).

    Args:
        objective_mix: the weight of the REINFORCE gradients in the objective of the actor, the rest being the
            gradients of the dynamics (`algo.actor.objective_mix` of DreamerV2). `None` for Plan2Explore: the
            dynamics for continuous actions, REINFORCE for discrete ones, and the values of the critic predicted
            for the whole trajectories.
        reward_fn: the rewards of the imagined trajectories (from the states and the actions that led to them);
            default: the ones predicted by the reward model.

    Returns:
        The losses (`policy_loss`, `value_loss`), the norms of the gradients before clipping (`actor_grads`,
        `critic_grads`, `None` without clipping), the predicted values (`values`), the lambda-values and the rewards
        of the imagined trajectories.
    """
    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    stoch_state_size = cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    with autocast(fabric):
        # (1, batch_size * sequence_length, stochastic_size * discrete_size)
        imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
        # (1, batch_size * sequence_length, recurrent_state_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories = [imagined_latent_state]
        imagined_actions = [torch.zeros(1, batch_size * sequence_length, sum(actions_dim), device=fabric.device)]

        # The imagination goes like this, with H=3:
        # Actions:       0   a'1      a'2     a'3
        #                    ^ \      ^ \      ^ \
        #                   /   \    /   \    /   \
        #                  /     v  /     v  /     v
        # States:        z0 ---> z'1 ---> z'2 ---> z'3
        # Rewards:       r'0     r'1      r'2      r'3
        # Values:        v'0     v'1      v'2      v'3
        # Lambda-values: l'0     l'1      l'2
        # Continues:     c0      c'1      c'2      c'3
        # where z0 comes from the posterior, while z'i is the imagined states (prior)
        for i in range(1, cfg.algo.horizon + 1):
            actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
            imagined_actions.append(actions)
            imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
            imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories.append(imagined_latent_state)
        imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
        imagined_actions = torch.cat(imagined_actions, dim=0)

        # Predict values, rewards and continues
        predicted_target_values = target_critic(imagined_trajectories)
        if reward_fn is None:
            predicted_rewards = world_model.reward_model(imagined_trajectories)
        else:
            predicted_rewards = reward_fn(imagined_trajectories, imagined_actions)
        if cfg.algo.world_model.use_continues and world_model.continue_model:
            continues = logits_to_probs(world_model.continue_model(imagined_trajectories), is_binary=True)
            true_continue = (1 - terminated).reshape(1, -1, 1) * cfg.algo.gamma
            continues = torch.cat((true_continue, continues[1:]))
        else:
            continues = torch.ones_like(predicted_rewards.detach()) * cfg.algo.gamma

        # The lambda-values, bootstrapped by the value of the last imagined state: (horizon, batch * sequence, 1)
        lambda_values = compute_lambda_values(
            predicted_rewards[:-1],
            predicted_target_values[:-1],
            continues[:-1],
            bootstrap=predicted_target_values[-1:],
            horizon=cfg.algo.horizon,
            lmbda=cfg.algo.lmbda,
        )

        # The discounts to multiply the lambda values to
        with torch.no_grad():
            discount = torch.cumprod(torch.cat((torch.ones_like(continues[:1]), continues[:-1]), 0), 0)

        # Actor optimization step. Eq. 6 from the paper
        # Given the following diagram, with H=3:
        # Actions:       0  [a'1]    [a'2]     a'3
        #                    ^ \      ^ \      ^ \
        #                   /   \    /   \    /   \
        #                  /     v  /     v  /     v
        # States:       [z0] -> [z'1] ->  z'2 ->   z'3
        # Values:       [v'0]   [v'1]     v'2      v'3
        # Lambda-values: l'0    [l'1]    [l'2]
        # Entropies:            [e'1]    [e'2]
        # The quantities wrapped into `[]` are the ones used for the actor optimization.
        # From Hafner (https://github.com/danijar/dreamerv2/blob/main/dreamerv2/agent.py#L253):
        # `Two states are lost at the end of the trajectory, one for the boostrap
        #  value prediction and one because the corresponding action does not lead
        #  anywhere anymore. One target is lost at the start of the trajectory
        #  because the initial state comes from the replay buffer.`
        policies: Sequence[Distribution] = actor(imagined_trajectories[:-2].detach())[1]

        def reinforce(baseline: Tensor) -> Tensor:
            advantage = (lambda_values[1:] - baseline[:-2]).detach()
            logprobs = [
                p.log_prob(imgnd_act[1:-1].detach()).unsqueeze(-1)
                for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, -1))
            ]
            return torch.stack(logprobs, -1).sum(-1) * advantage

        if objective_mix is not None:
            # Dynamics backpropagation and REINFORCE
            dynamics = lambda_values[1:]
            objective = objective_mix * reinforce(predicted_target_values) + (1 - objective_mix) * dynamics
        elif is_continuous:
            objective = lambda_values[1:]
        else:
            objective = reinforce(target_critic(imagined_trajectories))
        try:
            entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
        except NotImplementedError:
            entropy = torch.zeros_like(objective)
        policy_loss = -torch.mean(discount[:-2].detach() * (objective + entropy.unsqueeze(-1)))
    actor_grads = update(
        fabric,
        policy_loss,
        actor_optimizer,
        max_grad_norm=cfg.algo.actor.clip_gradients or 0.0,
        error_if_nonfinite=False,
    )

    with autocast(fabric):
        # The values of the first H imagined states, to match the lambda-values: the last one bootstraps them
        if objective_mix is not None:
            qv = Independent(Normal(critic(imagined_trajectories.detach()[:-1]), 1), 1)
        else:
            qv = Independent(Normal(critic(imagined_trajectories.detach())[:-1], 1), 1)

        # Critic optimization step. Eq. 5 from the paper.
        value_loss = -torch.mean(discount[:-1, ..., 0] * qv.log_prob(lambda_values.detach()))
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
        "values": predicted_target_values.detach(),
        "lambda_values": lambda_values.detach(),
        "rewards": predicted_rewards.detach(),
    }


def build_buffer(
    fabric: Fabric, cfg: Dict[str, Any], log_dir: str, dry_run_size: int
) -> EnvIndependentReplayBuffer | EpisodeBuffer:
    """The replay buffer of `buffer.type`: one buffer of sequences per environment (`sequential`), or a buffer of
    whole episodes (`episode`), sampled with their ends prioritized with `buffer.prioritize_ends`."""
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else dry_run_size
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
    memmap_dir = os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}")
    buffer_type = cfg.buffer.type.lower()
    if buffer_type == "sequential":
        return EnvIndependentReplayBuffer(
            buffer_size,
            n_envs=cfg.env.num_envs,
            obs_keys=obs_keys,
            memmap=cfg.buffer.memmap,
            memmap_dir=memmap_dir,
            buffer_cls=SequentialReplayBuffer,
            seed=cfg.seed + fabric.global_rank,
        )
    elif buffer_type == "episode":
        return EpisodeBuffer(
            buffer_size,
            minimum_episode_length=1 if cfg.dry_run else cfg.algo.per_rank_sequence_length,
            n_envs=cfg.env.num_envs,
            obs_keys=obs_keys,
            prioritize_ends=cfg.buffer.prioritize_ends,
            memmap=cfg.buffer.memmap,
            memmap_dir=memmap_dir,
        )
    raise ValueError(f"Unrecognized buffer type: must be one of `sequential` or `episode`, received: {buffer_type}")


def check_keys(fabric: Fabric, cfg: Dict[str, Any], obs_space: gym.spaces.Dict) -> None:
    """The observations must be a dictionary, and the keys of the decoder must be among the ones of the encoder."""
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


def setup_world_model(fabric: Fabric, world_model: WorldModel) -> WorldModel:
    """Set up the modules of the world model (device, precision)."""
    world_model.encoder = setup_module(fabric, world_model.encoder)
    world_model.observation_model = setup_module(fabric, world_model.observation_model)
    world_model.reward_model = setup_module(fabric, world_model.reward_model)
    world_model.rssm.recurrent_model = setup_module(fabric, world_model.rssm.recurrent_model)
    world_model.rssm.representation_model = setup_module(fabric, world_model.rssm.representation_model)
    world_model.rssm.transition_model = setup_module(fabric, world_model.rssm.transition_model)
    if world_model.continue_model:
        world_model.continue_model = setup_module(fabric, world_model.continue_model)
    return world_model


def actions_dim_of(action_space: gym.Space) -> Tuple[Tuple[int, ...], bool]:
    """The dimensions of the actions (one per discrete action) and whether they are continuous."""
    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    return actions_dim, is_continuous


class DreamerV2(Algorithm):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each on its own batch of sequences: the world model, then the
    actor and the critic on trajectories imagined from the batch."""

    off_policy = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        # These arguments cannot be changed
        cfg.env.screen_size = 64
        cfg.env.frame_stack = 1

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[DreamerV2State, EnvIndependentReplayBuffer | EpisodeBuffer]:
        cfg = self.cfg
        fabric = self.fabric
        self.actions_dim, self.is_continuous = actions_dim_of(action_space)
        check_keys(fabric, cfg, obs_space)

        world_model, actor, critic = build_models(self.actions_dim, self.is_continuous, cfg, obs_space)
        world_model = setup_world_model(fabric, world_model)
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
        state = DreamerV2State(
            world_model=world_model,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            world_optimizer=world_optimizer,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )
        self.schedule = schedule
        return state, build_buffer(fabric, cfg, log_dir, dry_run_size=2)

    def policy(self, state: DreamerV2State) -> PlayerDV2:
        """The policy to play with: it shares its modules (and so its weights) with the trained agent."""
        cfg = self.cfg
        return PlayerDV2(
            state.world_model.encoder,
            state.world_model.rssm.recurrent_model,
            state.world_model.rssm.representation_model,
            state.actor,
            self.actions_dim,
            cfg.env.num_envs,
            cfg.algo.world_model.stochastic_size,
            cfg.algo.world_model.recurrent_model.recurrent_state_size,
            self.fabric.device,
            discrete_size=cfg.algo.world_model.discrete_size,
        )

    def player(self, state: DreamerV2State) -> SequencePlayer:
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
        self, state: DreamerV2State, buffer: EnvIndependentReplayBuffer | EpisodeBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        yield from sample_batches(self.fabric, self.cfg, buffer, n_steps)

    def train_step(self, state: DreamerV2State, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg
        # The target critic is a copy of the critic, every `critic.per_rank_target_network_update_freq` gradient steps
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            for cp, tcp in zip(state.critic.module.parameters(), state.target_critic.parameters()):
                tcp.data.copy_(cp.data)

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
            posteriors,
            recurrent_states,
            batch["terminated"],
            self.is_continuous,
            self.actions_dim,
            objective_mix=cfg.algo.actor.objective_mix,
        )
        metrics["Loss/policy_loss"] = behaviour["policy_loss"]
        metrics["Loss/value_loss"] = behaviour["value_loss"]
        if behaviour["actor_grads"] is not None:
            metrics["Grads/actor"] = behaviour["actor_grads"]
        if behaviour["critic_grads"] is not None:
            metrics["Grads/critic"] = behaviour["critic_grads"]
        return metrics


# The most batches sampled (and moved to the device) at once: the first training can do many gradient steps
# (`algo.per_rank_pretrain_steps`)
MAX_SAMPLED_BATCHES = 16


def sample_batches(
    fabric: Fabric, cfg: Dict[str, Any], buffer: EnvIndependentReplayBuffer | EpisodeBuffer, n_steps: int
) -> Iterator[Dict[str, Tensor]]:
    """The batches of sequences of the `n_steps` gradient steps of an iteration, sampled `MAX_SAMPLED_BATCHES` at a
    time."""
    for first in range(0, n_steps, MAX_SAMPLED_BATCHES):
        n_samples = min(MAX_SAMPLED_BATCHES, n_steps - first)
        sample = buffer.sample_tensors(
            batch_size=cfg.algo.per_rank_batch_size,
            sequence_length=cfg.algo.per_rank_sequence_length,
            n_samples=n_samples,
            dtype=None,
            device=fabric.device,
            from_numpy=cfg.buffer.from_numpy,
        )  # [N_Samples, Sequence_Length, Batch_Size, ...]
        for i in range(n_samples):
            yield {k: v[i].float() for k, v in sample.items()}


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = DreamerV2(fabric, cfg)
    state, log_dir = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": state.world_model,
            "actor": state.actor,
            "critic": state.critic,
            "target_critic": state.target_critic,
        }
        register_model(fabric, log_models, cfg, models_to_log)
