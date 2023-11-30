from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict

import gymnasium as gym
import hydra
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from mlflow.models.model import ModelInfo
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch.distributions import Bernoulli, Independent, Normal
from torch.utils.data import BatchSampler
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_v1.agent import PlayerDV1, WorldModel, build_agent
from sheeprl.algos.dreamer_v1.loss import actor_loss, critic_loss, reconstruction_loss
from sheeprl.algos.dreamer_v1.utils import compute_lambda_values
from sheeprl.algos.dreamer_v2.utils import test
from sheeprl.data.buffers import AsyncReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import polynomial_decay, register_model, unwrap_fabric

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    world_optimizer: _FabricOptimizer,
    actor_optimizer: _FabricOptimizer,
    critic_optimizer: _FabricOptimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator | None,
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
        data (TensorDictBase): the batch of data to use for training.
        aggregator (MetricAggregator, optional): the aggregator to print the metrics.
        cfg (DictConfig): the configs.
    """
    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    validate_args = cfg.distribution.validate_args
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    device = fabric.device
    batch_obs = {k: data[k] / 255 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})

    # Dynamic Learning
    # initialize the recurrent_state that must be a tuple of tensors (one for GRU or RNN).
    # the dimension of each vector must be (1, batch_size, recurrent_state_size)
    # the recurrent state is the deterministic state (or ht) from the Figure 2c in
    # [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)

    # initialize the posterior that must be of dimension (1, batch_size, stochastic_size)
    # the stochastic state is the stochastic state (or st) from the Figure 2c in
    # [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
    posterior = torch.zeros(1, batch_size, stochastic_size, device=device)

    # initialize the tensors for dynamic learning
    # recurrent_states will contain all the recurrent states computed during the dynamic learning phase,
    # and its dimension is (sequence_length, batch_size, recurrent_state_size)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    # posteriors will contain all the posterior states computed during the dynamic learning phase,
    # and its dimension is (sequence_length, batch_size, stochastic_size)
    posteriors = torch.empty(sequence_length, batch_size, stochastic_size, device=device)

    # posteriors_mean and posteriors_std will contain all
    # the actual means and stds of the posterior states respectively,
    # their dimension is (sequence_length, batch_size, stochastic_size)
    posteriors_mean = torch.empty(sequence_length, batch_size, stochastic_size, device=device)
    posteriors_std = torch.empty(sequence_length, batch_size, stochastic_size, device=device)

    # priors_mean and priors_std will contain all
    # the predicted means and stds of the prior states respectively,
    # their dimension is (sequence_length, batch_size, stochastic_size)
    priors_mean = torch.empty(sequence_length, batch_size, stochastic_size, device=device)
    priors_std = torch.empty(sequence_length, batch_size, stochastic_size, device=device)

    embedded_obs = world_model.encoder(batch_obs)

    for i in range(0, sequence_length):
        # one step of dynamic learning, take the posterior state, the recurrent state, the action, and the observation
        # compute the mean and std of both the posterior and prior state, the new recurrent state
        # and the new posterior state
        recurrent_state, posterior, _, posterior_mean_std, prior_state_mean_std = world_model.rssm.dynamic(
            posterior, recurrent_state, data["actions"][i : i + 1], embedded_obs[i : i + 1]
        )
        recurrent_states[i] = recurrent_state
        posteriors[i] = posterior
        posteriors_mean[i] = posterior_mean_std[0]
        posteriors_std[i] = posterior_mean_std[1]
        priors_mean[i] = prior_state_mean_std[0]
        priors_std[i] = prior_state_mean_std[1]

    # concatenate the posterior states with the recurrent states on the last dimension
    # latent_states tensor has dimension (sequence_length, batch_size, recurrent_state_size + stochastic_size)
    latent_states = torch.cat((posteriors, recurrent_states), -1)

    # compute predictions for the observations
    decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)
    # compute the distribution of the reconstructed observations
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size observations.shape
    qo = {
        k: Independent(
            Normal(rec_obs, 1, validate_args=validate_args),
            len(rec_obs.shape[2:]),
            validate_args=validate_args,
        )
        for k, rec_obs in decoded_information.items()
    }

    # compute predictions for the rewards
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the number of rewards
    qr = Independent(
        Normal(world_model.reward_model(latent_states), 1, validate_args=validate_args),
        1,
        validate_args=validate_args,
    )

    # compute predictions for terminal steps, if required
    if cfg.algo.world_model.use_continues and world_model.continue_model:
        qc = Independent(
            Bernoulli(logits=world_model.continue_model(latent_states), validate_args=validate_args),
            1,
            validate_args=validate_args,
        )
        continue_targets = (1 - data["dones"]) * cfg.algo.gamma
    else:
        qc = continue_targets = None

    # compute the distributions of the states (posteriors and priors)
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the stochastic size
    posteriors_dist = Independent(
        Normal(posteriors_mean, posteriors_std, validate_args=validate_args),
        1,
        validate_args=validate_args,
    )
    priors_dist = Independent(
        Normal(priors_mean, priors_std, validate_args=validate_args),
        1,
        validate_args=validate_args,
    )

    # world model optimization step
    world_optimizer.zero_grad(set_to_none=True)
    # compute the overall loss of the world model
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
        continue_targets,
        cfg.algo.world_model.continue_scale_factor,
    )
    fabric.backward(rec_loss)
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_model_grads = fabric.clip_gradients(
            module=world_model, optimizer=world_optimizer, max_norm=cfg.algo.world_model.clip_gradients
        )
    world_optimizer.step()
    if aggregator and not aggregator.disabled:
        aggregator.update("Grads/world_model", world_model_grads.mean().detach())
        aggregator.update("Loss/world_model_loss", rec_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.detach())
        aggregator.update("Loss/reward_loss", reward_loss.detach())
        aggregator.update("Loss/state_loss", state_loss.detach())
        aggregator.update("Loss/continue_loss", continue_loss.detach())
        aggregator.update("State/kl", kl.detach())
        aggregator.update("State/post_entropy", posteriors_dist.entropy().mean().detach())
        aggregator.update("State/prior_entropy", priors_dist.entropy().mean().detach())

    # Behaviour Learning
    # Unflatten first 2 dimensions of recurrent and posterior states in order
    # to have all the states on the first dimension.
    # The 1 in the second dimension is needed for the recurrent model in the imagination step,
    # 1 because the agent imagines one state at a time.
    # (1, batch_size * sequence_length, stochastic_size)
    imagined_prior = posteriors.detach().reshape(1, -1, stochastic_size)

    # initialize the recurrent state of the recurrent model with the recurrent states computed
    # during the dynamic learning phase, its shape is (1, batch_size * sequence_length, recurrent_state_size).
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)

    # starting states for the imagination phase.
    # (1, batch_size * sequence_length, determinisitic_size + stochastic_size)
    imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)

    # initialize the tensor of the imagined states
    imagined_trajectories = torch.empty(
        cfg.algo.horizon, batch_size * sequence_length, stochastic_size + recurrent_state_size, device=device
    )

    # imagine trajectories in the latent space
    for i in range(cfg.algo.horizon):
        # actions tensor has dimension (1, batch_size * sequence_length, num_actions)
        actions = torch.cat(actor(imagined_latent_states.detach())[0], dim=-1)

        # imagination step
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)

        # update current state
        imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_states

    # predict values and rewards
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the number of values/rewards
    predicted_values = Independent(
        Normal(critic(imagined_trajectories), 1, validate_args=validate_args),
        1,
        validate_args=validate_args,
    ).mean
    predicted_rewards = Independent(
        Normal(world_model.reward_model(imagined_trajectories), 1, validate_args=validate_args),
        1,
        validate_args=validate_args,
    ).mean

    # predict the probability that the episode will continue in the imagined states
    if cfg.algo.world_model.use_continues and world_model.continue_model:
        predicted_continues = Independent(
            Bernoulli(logits=world_model.continue_model(imagined_trajectories), validate_args=validate_args),
            1,
            validate_args=validate_args,
        ).mean
    else:
        predicted_continues = torch.ones_like(predicted_rewards.detach()) * cfg.algo.gamma

    # compute the lambda_values, by passing as last values the values of the last imagined state
    # the dimensions of the lambda_values tensor are
    # (horizon, batch_size * sequence_length, recurrent_state_size + stochastic_size)
    lambda_values = compute_lambda_values(
        predicted_rewards,
        predicted_values,
        predicted_continues,
        last_values=predicted_values[-1],
        horizon=cfg.algo.horizon,
        lmbda=cfg.algo.lmbda,
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

    # actor optimization step
    actor_optimizer.zero_grad(set_to_none=True)
    # compute the policy loss
    policy_loss = actor_loss(discount * lambda_values)
    fabric.backward(policy_loss)
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads = fabric.clip_gradients(
            module=actor, optimizer=actor_optimizer, max_norm=cfg.algo.actor.clip_gradients
        )
    actor_optimizer.step()
    if aggregator and not aggregator.disabled:
        aggregator.update("Grads/actor", actor_grads.mean().detach())
        aggregator.update("Loss/policy_loss", policy_loss.detach())

    # Predict the values distribution only for the first H (horizon) imagined states
    # (to match the dimension with the lambda values),
    # it removes the last imagined state in the trajectory
    # because it is used only for computing correclty the lambda values
    qv = Independent(
        Normal(critic(imagined_trajectories.detach())[:-1], 1, validate_args=validate_args),
        1,
        validate_args=validate_args,
    )

    # critic optimization step
    critic_optimizer.zero_grad(set_to_none=True)
    # compute the value loss
    # the discount has shape (horizon, seuqence_length * batch_size, 1), so,
    # it is necessary to remove the last dimension to properly match the shapes
    # for the log prob
    value_loss = critic_loss(qv, lambda_values.detach(), discount[..., 0])
    fabric.backward(value_loss)
    if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
        critic_grads = fabric.clip_gradients(
            module=critic, optimizer=critic_optimizer, max_norm=cfg.algo.critic.clip_gradients
        )
    critic_optimizer.step()
    if aggregator and not aggregator.disabled:
        aggregator.update("Grads/critic", critic_grads.mean().detach())
        aggregator.update("Loss/value_loss", value_loss.detach())

    # Reset everything
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)
        cfg.algo.per_rank_batch_size = state["batch_size"] // world_size

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

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
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
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    clip_rewards_fn = lambda r: torch.tanh(r) if cfg.env.clip_rewards else r
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

    world_model, actor, critic = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"] if cfg.checkpoint.resume_from else None,
        state["actor"] if cfg.checkpoint.resume_from else None,
        state["critic"] if cfg.checkpoint.resume_from else None,
    )
    player = PlayerDV1(
        world_model.encoder.module,
        world_model.rssm.recurrent_model.module,
        world_model.rssm.representation_model.module,
        actor.module,
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        fabric.device,
    )

    # Optimizers
    world_optimizer = hydra.utils.instantiate(cfg.algo.world_model.optimizer, params=world_model.parameters())
    actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters())
    critic_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic.parameters())
    if cfg.checkpoint.resume_from:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
        world_optimizer, actor_optimizer, critic_optimizer
    )

    local_vars = locals()

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator).to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * world_size) if not cfg.dry_run else 2
    rb = AsyncReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        device=fabric.device if cfg.buffer.memmap else "cpu",
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        sequential=True,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], AsyncReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {world_size} processes are instantiated")
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=fabric.device if cfg.buffer.memmap else "cpu")
    expl_decay_steps = state["expl_decay_steps"] if cfg.checkpoint.resume_from else 0

    # Global variables
    train_step = 0
    last_train = 0
    start_step = state["update"] // world_size if cfg.checkpoint.resume_from else 1
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * world_size)
    updates_before_training = cfg.algo.train_every // policy_steps_per_update if not cfg.dry_run else 0
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = (cfg.algo.learning_starts // policy_steps_per_update) if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from and not cfg.buffer.checkpoint:
        learning_starts += start_step
    max_step_expl_decay = cfg.algo.actor.max_step_expl_decay // (cfg.algo.per_rank_gradient_steps * world_size)
    if cfg.checkpoint.resume_from:
        actor.expl_amount = polynomial_decay(
            expl_decay_steps,
            initial=cfg.algo.actor.expl_amount,
            final=cfg.algo.actor.expl_min,
            max_decay_steps=max_step_expl_decay,
        )

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
    o = envs.reset(seed=cfg.seed)[0]
    obs = {k: torch.from_numpy(v).view(cfg.env.num_envs, *v.shape[1:]) for k, v in o.items() if k.startswith("mask")}
    for k in obs_keys:
        torch_obs = torch.from_numpy(o[k]).view(cfg.env.num_envs, *o[k].shape[1:])
        if k in cfg.algo.mlp_keys.encoder:
            torch_obs = torch_obs.float()
        step_data[k] = torch_obs
        obs[k] = torch_obs
    step_data["dones"] = torch.zeros(cfg.env.num_envs, 1)
    step_data["actions"] = torch.zeros(cfg.env.num_envs, sum(actions_dim))
    step_data["rewards"] = torch.zeros(cfg.env.num_envs, 1)
    rb.add(step_data[None, ...])
    player.init_states()

    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * world_size

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
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
                            F.one_hot(torch.tensor(act), act_dim).numpy()
                            for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
                        ],
                        axis=-1,
                    )
            else:
                with torch.no_grad():
                    preprocessed_obs = {}
                    for k, v in obs.items():
                        if k in cfg.algo.cnn_keys.encoder:
                            preprocessed_obs[k] = v[None, ...].to(device) / 255 - 0.5
                        else:
                            preprocessed_obs[k] = v[None, ...].to(device)
                    mask = {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None
                    real_actions = actions = player.get_exploration_action(preprocessed_obs, mask)
                    actions = torch.cat(actions, -1).cpu().numpy()
                    if is_continuous:
                        real_actions = torch.cat(real_actions, -1).cpu().numpy()
                    else:
                        real_actions = np.array([real_act.cpu().argmax(dim=-1).numpy() for real_act in real_actions])
            o, rewards, dones, truncated, infos = envs.step(real_actions.reshape(envs.action_space.shape))
            dones = np.logical_or(dones, truncated)

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
        real_next_obs = copy.deepcopy(o)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v

        next_obs = {
            k: torch.from_numpy(v).view(cfg.env.num_envs, *v.shape[1:]) for k, v in o.items() if k.startswith("mask")
        }
        for k in obs_keys:  # [N_envs, N_obs]
            next_obs[k] = torch.from_numpy(o[k]).view(cfg.env.num_envs, *o[k].shape[1:])
            step_data[k] = torch.from_numpy(real_next_obs[k]).view(cfg.env.num_envs, *real_next_obs[k].shape[1:])
            if k in cfg.algo.mlp_keys.encoder:
                next_obs[k] = next_obs[k].float()
                step_data[k] = step_data[k].float()
        actions = torch.from_numpy(actions).view(cfg.env.num_envs, -1).float()
        rewards = torch.from_numpy(rewards).view(cfg.env.num_envs, -1).float()
        dones = torch.from_numpy(dones).view(cfg.env.num_envs, -1).float()

        # next_obs becomes the new obs
        obs = next_obs

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["rewards"] = clip_rewards_fn(rewards)
        rb.add(step_data[None, ...])

        # Reset and save the observation coming from the automatic reset
        dones_idxes = dones.nonzero(as_tuple=True)[0].tolist()
        reset_envs = len(dones_idxes)
        if reset_envs > 0:
            reset_data = TensorDict({}, batch_size=[reset_envs], device="cpu")
            for k in obs_keys:
                reset_data[k] = next_obs[k][dones_idxes]
            reset_data["dones"] = torch.zeros(reset_envs, 1)
            reset_data["actions"] = torch.zeros(reset_envs, np.sum(actions_dim))
            reset_data["rewards"] = torch.zeros(reset_envs, 1)
            rb.add(reset_data[None, ...], dones_idxes)
            # Reset dones so that `is_first` is updated
            for d in dones_idxes:
                step_data["dones"][d] = torch.zeros_like(step_data["dones"][d])
            # Reset internal agent states
            player.init_states(dones_idxes)

        updates_before_training -= 1

        # Train the agent
        if update > learning_starts and updates_before_training <= 0:
            local_data = rb.sample(
                cfg.algo.per_rank_batch_size,
                sequence_length=cfg.algo.per_rank_sequence_length,
                n_samples=cfg.algo.per_rank_gradient_steps,
            ).to(device)
            distributed_sampler = BatchSampler(range(local_data.shape[0]), batch_size=1, drop_last=False)
            # Start training
            with timer("Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute)):
                for i in distributed_sampler:
                    train(
                        fabric,
                        world_model,
                        actor,
                        critic,
                        world_optimizer,
                        actor_optimizer,
                        critic_optimizer,
                        local_data[i].view(cfg.algo.per_rank_sequence_length, cfg.algo.per_rank_batch_size),
                        aggregator,
                        cfg,
                    )
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
            if aggregator:
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
                "world_optimizer": world_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "expl_decay_steps": expl_decay_steps,
                "update": update * world_size,
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
    if fabric.is_global_zero:
        test(player, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:

        def log_models(
            run_id: str, experiment_id: str | None = None, run_name: str | None = None
        ) -> Dict[str, ModelInfo]:
            with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=True) as _:
                model_info = {}
                unwrapped_models = {}
                for k in cfg.model_manager.models.keys():
                    unwrapped_models[k] = unwrap_fabric(local_vars[k])
                    model_info[k] = mlflow.pytorch.log_model(unwrapped_models[k], artifact_path=k)
                mlflow.log_dict(cfg, "config.json")
            return model_info

        register_model(fabric, log_models, cfg)
