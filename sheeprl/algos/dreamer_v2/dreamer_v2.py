"""Dreamer-V2 implementation from [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
Adapted from the original implementation from https://github.com/danijar/dreamerv2
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Sequence

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Independent, Normal, OneHotCategorical
from torch.distributions.utils import logits_to_probs
from torch.optim import Optimizer
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_v2.agent import WorldModel, build_agent
from sheeprl.algos.dreamer_v2.loss import reconstruction_loss
from sheeprl.algos.dreamer_v2.utils import actor_objective, build_buffer, compute_lambda_values, prepare_obs, test
from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer
from sheeprl.utils.distribution import entropy as policy_entropy
from sheeprl.utils.env import get_episode_stats, get_vector_env_cls, make_env
from sheeprl.utils.fabric import autocast_cache_scope
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import off_policy_schedule, save_configs

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
    actions_dim: Sequence[int],
) -> None:
    """Runs one-step update of the agent.

    The follwing designations are used:
        - recurrent_state: is what is called ht or deterministic state from Figure 2 in .
        - prior: the stochastic state coming out from the transition model, depicted as z-hat_t in Figure 2.
        - posterior: the stochastic state coming out from the representation model, depicted as z_t in Figure 2.
        - latent state: the concatenation of the stochastic (can be both the prior or the posterior one)
        and recurrent states on the last dimension.
        - p: the output of the transition model, from Eq. 1.
        - q: the output of the representation model, from Eq. 1.
        - po: the output of the observation model (decoder), from Eq. 1.
        - pr: the output of the reward model, from Eq. 1.
        - pc: the output of the continue model (discout predictor), from Eq. 1.
        - pv: the output of the value model (critic), from Eq. 3.

    In particular, the agent is updated as following:

    1. Dynamic Learning:
        - Encoder: encode the observations.
        - Recurrent Model: compute the recurrent state from the previous recurrent state,
            the previous posterior state, and from the previous actions.
        - Transition Model: predict the stochastic state from the recurrent state, i.e., the deterministic state or ht.
        - Representation Model: compute the actual stochastic state from the recurrent state and
            from the embedded observations provided by the environment.
        - Observation Model: reconstructs observations from latent states.
        - Reward Model: estimate rewards from the latent states.
        - Update the models
    2. Behaviour Learning:
        - Imagine trajectories in the latent space from each latent state
        s_t up to the horizon H: s'_(t+1), ..., s'_(t+H).
        - Predict rewards and values in the imagined trajectories.
        - Compute lambda targets (Eq. 4 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193))
        - Update the actor and the critic

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
        actions_dim (Sequence[int]): the actions dimension.
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

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)

    # Cast the weights to low precision once for the whole forward pass, not at every step of the unroll
    with autocast_cache_scope(fabric):
        # The outputs of every step are concatenated at the end of the unroll: writing them in place into
        # preallocated tensors makes the backward pass copy the gradient of the whole tensor at every step
        # Initialize the recurrent_states, which will contain all the recurrent states
        # computed during the dynamic learning phase
        recurrent_states = []

        # Initialize all the lists to collect priors and posteriors states with their associated logits
        priors_logits = []
        posteriors = []
        posteriors_logits = []

        # Embed observations from the environment
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

        # Concatenate the posteriors with the recurrent states on the last dimension.
        # Latent_states has dimension
        # (sequence_length, batch_size, recurrent_state_size + stochastic_size * discrete_size)
        latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

        # Compute predictions for the observations
        decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)

        # Compute the distribution over the reconstructed observations
        po = {k: Independent(Normal(rec_obs, 1), len(rec_obs.shape[2:])) for k, rec_obs in decoded_information.items()}

        # Compute the distribution over the rewards
        pr = Independent(Normal(world_model.reward_model(latent_states), 1), 1)

        # Compute the distribution over the terminal steps, if required
        if cfg.algo.world_model.use_continues and world_model.continue_model:
            pc = Independent(Bernoulli(logits=world_model.continue_model(latent_states)), 1)
            continues_targets = (1 - data["terminated"]) * cfg.algo.gamma
        else:
            pc = continues_targets = None

        # Reshape posterior and prior logits to shape [T, B, 32, 32]
        priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
        posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

        # World model optimization step
        world_optimizer.zero_grad(set_to_none=True)
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
    with autocast_cache_scope(fabric):
        # (1, batch_size * sequence_length, stochastic_size * discrete_size)
        imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)

        # (1, batch_size * sequence_length, recurrent_state_size).
        recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)

        # (1, batch_size * sequence_length, recurrent_state_size + stochastic_size * discrete_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)

        # Initialize the list of the imagined trajectories, concatenated at the end of the imagination
        imagined_trajectories = [imagined_latent_state]

        # Initialize the list of the imagined actions, concatenated at the end of the imagination
        imagined_actions = [torch.zeros(1, batch_size * sequence_length, data["actions"].shape[-1], device=device)]

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
        # where z0 comes from the posterior
        # (is initialized as the concatenation of the posteriors and the recurrent states)
        # while z'i is the imagined states (prior)

        # Imagine trajectories in the latent space
        for i in range(1, cfg.algo.horizon + 1):
            # (1, batch_size * sequence_length, sum(actions_dim))
            actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
            imagined_actions.append(actions)

            # Imagination step
            imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)

            # Update current state
            imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories.append(imagined_latent_state)
        imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
        imagined_actions = torch.cat(imagined_actions, dim=0)

        # Predict values and rewards
        predicted_target_values = target_critic(imagined_trajectories)
        predicted_rewards = world_model.reward_model(imagined_trajectories)
        if cfg.algo.world_model.use_continues and world_model.continue_model:
            continues = logits_to_probs(world_model.continue_model(imagined_trajectories), is_binary=True)
            true_continue = (1 - data["terminated"]).reshape(1, -1, 1) * cfg.algo.gamma
            continues = torch.cat((true_continue, continues[1:]))
        else:
            continues = torch.ones_like(predicted_rewards.detach()) * cfg.algo.gamma

        # Compute the lambda_values, by passing as last value the value of the last imagined state
        # (horizon, batch_size * sequence_length, 1)
        lambda_values = compute_lambda_values(
            predicted_rewards[:-1],
            predicted_target_values[:-1],
            continues[:-1],
            bootstrap=predicted_target_values[-1:],
            horizon=cfg.algo.horizon,
            lmbda=cfg.algo.lmbda,
        )

        # Compute the discounts to multiply the lambda values
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
        actor_optimizer.zero_grad(set_to_none=True)
        policies: Sequence[Distribution] = actor(imagined_trajectories[:-2].detach())[1]

        def reinforce() -> Tensor:
            advantage = (lambda_values[1:] - predicted_target_values[:-2]).detach()
            logprobs = [
                p.log_prob(imgnd_act[1:-1].detach()).unsqueeze(-1)
                for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, -1))
            ]
            return torch.stack(logprobs, -1).sum(-1) * advantage

        # Dynamics backpropagation (the lambda-values) and REINFORCE
        objective = actor_objective(cfg.algo.actor.objective_mix, actor.is_continuous, lambda_values[1:], reinforce)
        # The tanh-normal policies have no analytic entropy: it is estimated from samples
        entropy = cfg.algo.actor.ent_coef * torch.stack([policy_entropy(p) for p in policies], -1).sum(dim=-1)
        policy_loss = -torch.mean(discount[:-2].detach() * (objective + entropy.unsqueeze(-1)))
    fabric.backward(policy_loss)
    actor_grads = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads = fabric.clip_gradients(
            module=actor, optimizer=actor_optimizer, max_norm=cfg.algo.actor.clip_gradients, error_if_nonfinite=False
        )
    actor_optimizer.step()

    with autocast_cache_scope(fabric):
        # Predict the values distribution only for the first H (horizon)
        # imagined states (to match the dimension with the lambda values),
        # It removes the last imagined state in the trajectory because it is used for bootstrapping
        qv = Independent(Normal(critic(imagined_trajectories.detach()[:-1]), 1), 1)

        # Critic optimization step. Eq. 5 from the paper.
        critic_optimizer.zero_grad(set_to_none=True)
        value_loss = -torch.mean(discount[:-1, ..., 0] * qv.log_prob(lambda_values.detach()))
    fabric.backward(value_loss)
    critic_grads = None
    if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
        critic_grads = fabric.clip_gradients(
            module=critic,
            optimizer=critic_optimizer,
            max_norm=cfg.algo.critic.clip_gradients,
            error_if_nonfinite=False,
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
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
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
        state = fabric.load(cfg.checkpoint.resume_from, weights_only=False)

    # These arguments cannot be changed
    cfg.env.screen_size = 64
    cfg.env.frame_stack = 1

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
            make_env(
                cfg,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank * cfg.env.num_envs,
                log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    # Seed the random actions played before the training starts
    envs.action_space.seed(cfg.seed + rank)
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

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    rb = build_buffer(fabric, cfg, log_dir, dry_run_size=2)
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], (EnvIndependentReplayBuffer, EpisodeBuffer)):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {world_size} processes are instantiated")

    # Global variables
    train_step = 0
    last_train = 0
    start_iter = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["iter_num"] // world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_iter = int(cfg.env.num_envs * world_size)
    total_iters = cfg.algo.total_steps // policy_steps_per_iter if not cfg.dry_run else 1
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // world_size
    # Random actions in the iterations up to `learning_starts`, training from `train_starts`
    learning_starts, train_starts, ratio = off_policy_schedule(
        cfg, state if cfg.checkpoint.resume_from else None, start_iter, policy_steps_per_iter, fabric.world_size
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
    step_data = {}
    obs = envs.reset(seed=cfg.seed + rank * cfg.env.num_envs)[0]
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    if cfg.dry_run:
        step_data["truncated"] = step_data["truncated"] + 1
        step_data["terminated"] = step_data["terminated"] + 1
    step_data["actions"] = np.zeros((1, cfg.env.num_envs, sum(actions_dim)))
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["terminated"])
    rb.add(step_data, validate_args=cfg.buffer.validate_args)
    player.init_states()

    # The gradient steps of every process from the start of the run, also in the run it resumes (the older
    # checkpoints don't have them)
    cumulative_per_rank_gradient_steps = state.get("per_rank_gradient_steps", 0) if cfg.checkpoint.resume_from else 0
    for iter_num in range(start_iter, total_iters + 1):
        policy_step += policy_steps_per_iter

        with torch.inference_mode():
            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                # Sample an action given the observation received by the environment
                if iter_num <= learning_starts and "minedojo" not in cfg.env.wrapper._target_.lower():
                    real_actions = actions = np.array(envs.action_space.sample())
                    if not is_continuous:
                        # One row per environment and one column per discrete action: the one-hots of every discrete
                        # action of every environment (they were mixed between the environments)
                        per_action = actions.reshape(cfg.env.num_envs, len(actions_dim)).T
                        actions = np.concatenate(
                            [
                                F.one_hot(torch.as_tensor(act), act_dim).numpy()
                                for act, act_dim in zip(per_action, actions_dim)
                            ],
                            axis=-1,
                        )
                else:
                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None
                    real_actions = actions = player.get_actions(torch_obs, mask=mask)
                    actions = torch.cat(actions, -1).view(cfg.env.num_envs, -1).cpu().numpy()
                    if is_continuous:
                        real_actions = torch.stack(real_actions, -1).cpu().numpy()
                    else:
                        real_actions = (
                            torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                        )

                step_data["is_first"] = copy.deepcopy(np.logical_or(step_data["terminated"], step_data["truncated"]))
                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                dones = np.logical_or(terminated, truncated).astype(np.uint8)
                if cfg.dry_run and isinstance(rb, EpisodeBuffer):
                    dones = np.ones_like(dones)

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

            for k in obs_keys:  # [N_envs, N_obs]
                step_data[k] = real_next_obs[k][np.newaxis]

            # Next_obs becomes the new obs
            obs = next_obs

            step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
            step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
            step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
            step_data["rewards"] = clip_rewards_fn(rewards).reshape((1, cfg.env.num_envs, -1))
            rb.add(step_data, validate_args=cfg.buffer.validate_args)

            # Reset and save the observation coming from the automatic reset
            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (next_obs[k][dones_idxes])[np.newaxis]
                reset_data["terminated"] = np.zeros((1, reset_envs, 1))
                reset_data["truncated"] = np.zeros((1, reset_envs, 1))
                reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                reset_data["rewards"] = np.zeros((1, reset_envs, 1))
                reset_data["is_first"] = np.ones_like(reset_data["terminated"])
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)
                # Reset dones so that `is_first` is updated
                for d in dones_idxes:
                    step_data["terminated"][0, d] = np.zeros_like(step_data["terminated"][0, d])
                    step_data["truncated"][0, d] = np.zeros_like(step_data["truncated"][0, d])
                # Reset internal agent states
                player.init_states(dones_idxes)

        # Train the agent
        if iter_num >= train_starts:
            ratio_steps = policy_step - (train_starts - 1) * policy_steps_per_iter
            per_rank_gradient_steps = ratio(ratio_steps / world_size)
            if per_rank_gradient_steps > 0:
                local_data = rb.sample_tensors(
                    batch_size=cfg.algo.per_rank_batch_size,
                    sequence_length=cfg.algo.per_rank_sequence_length,
                    n_samples=per_rank_gradient_steps,
                    dtype=None,
                    device=fabric.device,
                    from_numpy=cfg.buffer.from_numpy,
                )
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    for i in range(per_rank_gradient_steps):
                        if (
                            cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                            == 0
                        ):
                            for cp, tcp in zip(critic.module.parameters(), target_critic.module.parameters()):
                                tcp.data.copy_(cp.data)
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
                            actions_dim,
                        )
                        cumulative_per_rank_gradient_steps += 1
                    train_step += world_size

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
            iter_num == total_iters and cfg.checkpoint.save_last
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
                "ratio": ratio.state_dict(),
                "per_rank_gradient_steps": cumulative_per_rank_gradient_steps,
                "iter_num": iter_num * world_size,
                "batch_size": cfg.algo.per_rank_batch_size * world_size,
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
        test(player, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"world_model": world_model, "actor": actor, "critic": critic, "target_critic": target_critic}
        register_model(fabric, log_models, cfg, models_to_log)
