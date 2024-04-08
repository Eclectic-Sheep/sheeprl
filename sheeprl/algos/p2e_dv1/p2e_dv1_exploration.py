from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch import Tensor
from torch.distributions import Bernoulli, Independent, Normal
from torch.distributions.utils import logits_to_probs
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_v1.agent import WorldModel
from sheeprl.algos.dreamer_v1.loss import actor_loss, critic_loss, reconstruction_loss
from sheeprl.algos.dreamer_v1.utils import compute_lambda_values
from sheeprl.algos.dreamer_v2.utils import test
from sheeprl.algos.p2e_dv1.agent import build_agent
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs, unwrap_fabric

# Decomment the following line if you are using MineDojo on an headless machine
# os.environ["MINEDOJO_HEADLESS"] = "1"


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor_task: _FabricModule,
    critic_task: _FabricModule,
    world_optimizer: _FabricOptimizer,
    actor_task_optimizer: _FabricOptimizer,
    critic_task_optimizer: _FabricOptimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    ensembles: _FabricModule,
    ensemble_optimizer: _FabricOptimizer,
    actor_exploration: _FabricModule,
    critic_exploration: _FabricModule,
    actor_exploration_optimizer: _FabricOptimizer,
    critic_exploration_optimizer: _FabricOptimizer,
) -> None:
    """Runs one-step update of the agent.

    In particular, it updates the agent as specified by Algorithm 1 in
    [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960).

    The algorithm is made by different phases:
    1. Dynamic Learning: see Algorithm 1 in
    [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
    2. Ensemble Learning: learn the ensemble models as described in
    [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960).
        The ensemble models give the novelty of the state visited by the agent.
    3. Behaviour Learning Exploration: the agent learns to explore the environment,
    having as reward only the intrinsic reward, computed from the ensembles.
    4. Behaviour Learning Task (zero-shot): the agent learns to solve the task,
    the experiences it uses to learn it are the ones collected during the exploration:
        - Imagine trajectories in the latent space from each latent state
        s_t up to the horizon H: s'_(t+1), ..., s'_(t+H).
        - Predict rewards and values in the imagined trajectories.
        - Compute lambda targets (Eq. 6 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603))
        - Update the actor and the critic

    This method is based on [sheeprl.algos.dreamer_v1.dreamer_v1](sheeprl.algos.dreamer_v1.dreamer_v1) algorithm,
    extending it to implement the
    [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960).

    Args:
        fabric (Fabric): the fabric instance.
        world_model (WorldModel): the world model wrapped with Fabric.
        actor_task (_FabricModule): the actor for solving the task.
        critic_task (_FabricModule): the critic for solving the task.
        world_optimizer (_FabricOptimizer): the world optimizer.
        actor_task_optimizer (_FabricOptimizer): the actor optimizer for solving the task.
        critic_task_optimizer (_FabricOptimizer): the critic optimizer for solving the task.
        data (Dict[str, Tensor]): the batch of data to use for training.
        aggregator (MetricAggregator, optional): the aggregator to print the metrics.
        cfg (DictConfig): the configs.
        ensembles (_FabricModule): the ensemble models.
        ensemble_optimizer (_FabricOptimizer): the optimizer of the ensemble models.
        actor_exploration (_FabricModule): the actor for exploration.
        critic_exploration (_FabricModule): the critic for exploration.
        actor_exploration_optimizer (_FabricOptimizer): the optimizer of the actor for exploration.
        critic_exploration_optimizer (_FabricOptimizer): the optimizer of the critic for exploration.
    """
    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    device = fabric.device
    batch_obs = {k: data[k] / 255 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})

    # Dynamic Learning
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    posteriors = torch.empty(sequence_length, batch_size, stochastic_size, device=device)
    priors = torch.empty(sequence_length, batch_size, stochastic_size, device=device)
    posteriors_mean = torch.empty(sequence_length, batch_size, stochastic_size, device=device)
    posteriors_std = torch.empty(sequence_length, batch_size, stochastic_size, device=device)
    priors_mean = torch.empty(sequence_length, batch_size, stochastic_size, device=device)
    priors_std = torch.empty(sequence_length, batch_size, stochastic_size, device=device)
    embedded_obs = world_model.encoder(batch_obs)

    for i in range(0, sequence_length):
        recurrent_state, posterior, prior, posterior_mean_std, prior_mean_std = world_model.rssm.dynamic(
            posterior, recurrent_state, data["actions"][i : i + 1], embedded_obs[i : i + 1]
        )
        recurrent_states[i] = recurrent_state
        posteriors[i] = posterior
        posteriors_mean[i] = posterior_mean_std[0]
        posteriors_std[i] = posterior_mean_std[1]
        priors_mean[i] = prior_mean_std[0]
        priors_std[i] = prior_mean_std[1]
        priors[i] = prior
    latent_states = torch.cat((posteriors, recurrent_states), -1)

    decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)
    qo = {k: Independent(Normal(rec_obs, 1), len(rec_obs.shape[2:])) for k, rec_obs in decoded_information.items()}
    qr = Independent(Normal(world_model.reward_model(latent_states.detach()), 1), 1)
    if cfg.algo.world_model.use_continues and world_model.continue_model:
        qc = Independent(Bernoulli(logits=world_model.continue_model(latent_states.detach())), 1)
        continues_targets = (1 - data["terminated"]) * cfg.algo.gamma
    else:
        qc = continues_targets = None
    posteriors_dist = Independent(Normal(posteriors_mean, posteriors_std), 1)
    priors_dist = Independent(Normal(priors_mean, priors_std), 1)

    world_optimizer.zero_grad(set_to_none=True)
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
    fabric.backward(rec_loss)
    world_grad = None
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_grad = fabric.clip_gradients(
            module=world_model,
            optimizer=world_optimizer,
            max_norm=cfg.algo.world_model.clip_gradients,
            error_if_nonfinite=False,
        )
    world_optimizer.step()

    # Ensemble Learning
    loss = 0.0
    ensemble_optimizer.zero_grad(set_to_none=True)
    for ens in ensembles:
        out = ens(torch.cat((posteriors.detach(), recurrent_states.detach(), data["actions"].detach()), -1))[:-1]
        next_obs_embedding_dist = Independent(Normal(out, 1), 1)
        loss -= next_obs_embedding_dist.log_prob(embedded_obs.detach()[1:]).mean()
    loss.backward()
    ensemble_grad = None
    if cfg.algo.ensembles.clip_gradients is not None and cfg.algo.ensembles.clip_gradients > 0:
        ensemble_grad = fabric.clip_gradients(
            module=ens,
            optimizer=ensemble_optimizer,
            max_norm=cfg.algo.ensembles.clip_gradients,
            error_if_nonfinite=False,
        )
    ensemble_optimizer.step()

    # Behaviour Learning Exploration
    imagined_prior = posteriors.detach().reshape(1, -1, stochastic_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    imagined_trajectories = torch.empty(
        cfg.algo.horizon, batch_size * sequence_length, stochastic_size + recurrent_state_size, device=device
    )
    # initialize the tensor of imagined actions, they are used to compute the intrinsic reward
    imagined_actions = torch.zeros(
        cfg.algo.horizon, batch_size * sequence_length, data["actions"].shape[-1], device=device
    )

    # imagine trajectories in the latent space
    for i in range(cfg.algo.horizon):
        actions = torch.cat(actor_exploration(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state
    predicted_values_exploration = critic_exploration(imagined_trajectories)

    # Predict intrinsic reward
    next_obs_embedding = torch.zeros(
        len(ensembles),
        cfg.algo.horizon,
        batch_size * sequence_length,
        embedded_obs.shape[-1],
        device=device,
    )
    for i, ens in enumerate(ensembles):
        next_obs_embedding[i] = ens(torch.cat((imagined_trajectories.detach(), imagined_actions.detach()), -1))

    # next_obs_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
    intrinsic_reward = next_obs_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier

    if cfg.algo.world_model.use_continues and world_model.continue_model:
        predicted_continues = logits_to_probs(logits=world_model.continue_model(imagined_trajectories), is_binary=True)
    else:
        predicted_continues = torch.ones_like(intrinsic_reward.detach()) * cfg.algo.gamma

    lambda_values_exploration = compute_lambda_values(
        intrinsic_reward,
        predicted_values_exploration,
        predicted_continues,
        last_values=predicted_values_exploration[-1],
        horizon=cfg.algo.horizon,
        lmbda=cfg.algo.lmbda,
    )

    with torch.no_grad():
        discount = torch.cumprod(torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0)

    actor_exploration_optimizer.zero_grad(set_to_none=True)
    policy_loss_exploration = actor_loss(discount * lambda_values_exploration)
    fabric.backward(policy_loss_exploration)
    actor_exploration_grad = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_exploration_grad = fabric.clip_gradients(
            module=actor_exploration,
            optimizer=actor_exploration_optimizer,
            max_norm=cfg.algo.actor.clip_gradients,
            error_if_nonfinite=False,
        )
    actor_exploration_optimizer.step()

    qv = Independent(Normal(critic_exploration(imagined_trajectories.detach())[:-1], 1), 1)
    critic_exploration_optimizer.zero_grad(set_to_none=True)
    value_loss_exploration = critic_loss(qv, lambda_values_exploration.detach(), discount[..., 0])
    fabric.backward(value_loss_exploration)
    critic_exploration_grad = None
    if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
        critic_exploration_grad = fabric.clip_gradients(
            module=critic_exploration,
            optimizer=critic_exploration_optimizer,
            max_norm=cfg.algo.critic.clip_gradients,
            error_if_nonfinite=False,
        )
    critic_exploration_optimizer.step()

    # reset the world_model gradients, to avoid interferences with task learning
    world_optimizer.zero_grad(set_to_none=True)

    # Behaviour Learning Task
    imagined_prior = posteriors.detach().reshape(1, -1, stochastic_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    imagined_trajectories = torch.empty(
        cfg.algo.horizon, batch_size * sequence_length, stochastic_size + recurrent_state_size, device=device
    )
    for i in range(cfg.algo.horizon):
        actions = torch.cat(actor_task(imagined_latent_state.detach())[0], dim=-1)
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state

    predicted_values_task = critic_task(imagined_trajectories)
    predicted_rewards = world_model.reward_model(imagined_trajectories)
    if cfg.algo.world_model.use_continues and world_model.continue_model:
        predicted_continues = logits_to_probs(logits=world_model.continue_model(imagined_trajectories), is_binary=True)
    else:
        predicted_continues = torch.ones_like(predicted_rewards.detach()) * cfg.algo.gamma

    lambda_values_task = compute_lambda_values(
        predicted_rewards,
        predicted_values_task,
        predicted_continues,
        last_values=predicted_values_task[-1],
        horizon=cfg.algo.horizon,
        lmbda=cfg.algo.lmbda,
    )

    with torch.no_grad():
        discount = torch.cumprod(torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0)

    actor_task_optimizer.zero_grad(set_to_none=True)
    policy_loss_task = actor_loss(discount * lambda_values_task)
    fabric.backward(policy_loss_task)
    actor_task_grad = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_task_grad = fabric.clip_gradients(
            module=actor_task,
            optimizer=actor_task_optimizer,
            max_norm=cfg.algo.actor.clip_gradients,
            error_if_nonfinite=False,
        )
    actor_task_optimizer.step()

    qv = Independent(Normal(critic_task(imagined_trajectories.detach())[:-1], 1), 1)
    critic_task_optimizer.zero_grad(set_to_none=True)
    value_loss_task = critic_loss(qv, lambda_values_task.detach(), discount[..., 0])
    fabric.backward(value_loss_task)
    critic_task_grad = None
    if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
        critic_task_grad = fabric.clip_gradients(
            module=critic_task,
            optimizer=critic_task_optimizer,
            max_norm=cfg.algo.critic.clip_gradients,
            error_if_nonfinite=False,
        )
    critic_task_optimizer.step()
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/world_model_loss", rec_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.detach())
        aggregator.update("Loss/reward_loss", reward_loss.detach())
        aggregator.update("Loss/state_loss", state_loss.detach())
        aggregator.update("Loss/continue_loss", continue_loss.detach())
        aggregator.update("State/kl", kl.mean().detach())
        aggregator.update("State/post_entropy", posteriors_dist.entropy().mean().detach())
        aggregator.update("State/prior_entropy", priors_dist.entropy().mean().detach())
        aggregator.update("Loss/ensemble_loss", loss.detach().cpu())
        aggregator.update("Values_exploration/predicted_values", predicted_values_exploration.detach().cpu().mean())
        aggregator.update("Values_exploration/lambda_values", lambda_values_exploration.detach().cpu().mean())
        aggregator.update("Rewards/intrinsic", intrinsic_reward.detach().cpu().mean())
        aggregator.update("Loss/policy_loss_exploration", policy_loss_exploration.detach())
        aggregator.update("Loss/value_loss_exploration", value_loss_exploration.detach())
        aggregator.update("Loss/policy_loss_task", policy_loss_task.detach())
        aggregator.update("Loss/value_loss_task", value_loss_task.detach())
        if world_grad:
            aggregator.update("Grads/world_model", world_grad.detach())
        if ensemble_grad:
            aggregator.update("Grads/ensemble", ensemble_grad.detach())
        if actor_exploration_grad:
            aggregator.update("Grads/actor_exploration", actor_exploration_grad.detach())
        if critic_exploration_grad:
            aggregator.update("Grads/critic_exploration", critic_exploration_grad.detach())
        if actor_task_grad:
            aggregator.update("Grads/actor_task", actor_task_grad.detach())
        if critic_task_grad:
            aggregator.update("Grads/critic_task", critic_task_grad.detach())

    # Reset everything
    actor_exploration_optimizer.zero_grad(set_to_none=True)
    critic_exploration_optimizer.zero_grad(set_to_none=True)
    actor_task_optimizer.zero_grad(set_to_none=True)
    critic_task_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)
    ensemble_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # These arguments cannot be changed
    cfg.env.screen_size = 64
    cfg.env.frame_stack = 1
    cfg.algo.player.actor_type = "exploration"

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

    world_model, ensembles, actor_task, critic_task, actor_exploration, critic_exploration, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"] if cfg.checkpoint.resume_from else None,
        state["ensembles"] if cfg.checkpoint.resume_from else None,
        state["actor_task"] if cfg.checkpoint.resume_from else None,
        state["critic_task"] if cfg.checkpoint.resume_from else None,
        state["actor_exploration"] if cfg.checkpoint.resume_from else None,
        state["critic_exploration"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers
    world_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
    )
    actor_task_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer, params=actor_task.parameters(), _convert_="all"
    )
    critic_task_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer, params=critic_task.parameters(), _convert_="all"
    )
    actor_exploration_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer, params=actor_exploration.parameters(), _convert_="all"
    )
    critic_exploration_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer, params=critic_exploration.parameters(), _convert_="all"
    )

    ensemble_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=ensembles.parameters(), _convert_="all"
    )
    if cfg.checkpoint.resume_from:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_task_optimizer.load_state_dict(state["actor_task_optimizer"])
        critic_task_optimizer.load_state_dict(state["critic_task_optimizer"])
        ensemble_optimizer.load_state_dict(state["ensemble_optimizer"])
        actor_exploration_optimizer.load_state_dict(state["actor_exploration_optimizer"])
        critic_exploration_optimizer.load_state_dict(state["critic_exploration_optimizer"])
    (
        world_optimizer,
        actor_task_optimizer,
        critic_task_optimizer,
        ensemble_optimizer,
        actor_exploration_optimizer,
        critic_exploration_optimizer,
    ) = fabric.setup_optimizers(
        world_optimizer,
        actor_task_optimizer,
        critic_task_optimizer,
        ensemble_optimizer,
        actor_exploration_optimizer,
        critic_exploration_optimizer,
    )

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * world_size) if not cfg.dry_run else 2
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        obs_keys=obs_keys,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=SequentialReplayBuffer,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], EnvIndependentReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {world_size} processes are instantiated")
    step_data = {}

    # Global variables
    train_step = 0
    last_train = 0
    start_step = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["update"] // world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * world_size)
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = (cfg.algo.learning_starts // policy_steps_per_update) if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // world_size
        if not cfg.buffer.checkpoint:
            learning_starts += start_step

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

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
        if k in cfg.algo.cnn_keys.encoder:
            obs[k] = obs[k].reshape(cfg.env.num_envs, -1, *obs[k].shape[-2:])
        step_data[k] = obs[k][np.newaxis]
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["actions"] = np.zeros((1, cfg.env.num_envs, sum(actions_dim)))
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    rb.add(step_data, validate_args=cfg.buffer.validate_args)
    player.init_states()

    cumulative_per_rank_gradient_steps = 0
    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * world_size

        with torch.inference_mode():
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
                    normalized_obs = {}
                    for k in obs_keys:
                        torch_obs = torch.as_tensor(obs[k][np.newaxis], dtype=torch.float32, device=device)
                        if k in cfg.algo.cnn_keys.encoder:
                            torch_obs = torch_obs / 255 - 0.5
                        normalized_obs[k] = torch_obs
                    mask = {k: v for k, v in normalized_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None
                    real_actions = actions = player.get_exploration_actions(normalized_obs, mask=mask, step=policy_step)
                    actions = torch.cat(actions, -1).view(cfg.env.num_envs, -1).cpu().numpy()
                    if is_continuous:
                        real_actions = torch.cat(real_actions, -1).cpu().numpy()
                    else:
                        real_actions = (
                            torch.cat([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                        )
                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                dones = np.logical_or(terminated, truncated).astype(np.uint8)

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
                if k in cfg.algo.cnn_keys.encoder:
                    next_obs[k] = next_obs[k].reshape(cfg.env.num_envs, -1, *next_obs[k].shape[-2:])
                    real_next_obs[k] = real_next_obs[k].reshape(cfg.env.num_envs, -1, *real_next_obs[k].shape[-2:])
                step_data[k] = real_next_obs[k][np.newaxis]

            # next_obs becomes the new obs
            obs = next_obs

            step_data["terminated"] = terminated[np.newaxis]
            step_data["truncated"] = truncated[np.newaxis]
            step_data["actions"] = actions[np.newaxis]
            step_data["rewards"] = clip_rewards_fn(rewards)[np.newaxis]
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
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)
                for d in dones_idxes:
                    step_data["terminated"][0, d] = np.zeros_like(step_data["terminated"][0, d])
                    step_data["truncated"][0, d] = np.zeros_like(step_data["truncated"][0, d])
                # Reset internal agent states
                player.init_states(reset_envs=dones_idxes)

        # Train the agent
        if update >= learning_starts:
            per_rank_gradient_steps = ratio(policy_step / world_size)
            if per_rank_gradient_steps > 0:
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    sample = rb.sample_tensors(
                        batch_size=cfg.algo.per_rank_batch_size,
                        sequence_length=cfg.algo.per_rank_sequence_length,
                        n_samples=per_rank_gradient_steps,
                        dtype=None,
                        device=device,
                        from_numpy=cfg.buffer.from_numpy,
                    )  # [N_samples, Seq_len, Batch_size, ...]
                    for i in range(per_rank_gradient_steps):
                        batch = {k: v[i].float() for k, v in sample.items()}
                        train(
                            fabric,
                            world_model,
                            actor_task,
                            critic_task,
                            world_optimizer,
                            actor_task_optimizer,
                            critic_task_optimizer,
                            batch,
                            aggregator,
                            cfg,
                            ensembles=ensembles,
                            ensemble_optimizer=ensemble_optimizer,
                            actor_exploration=actor_exploration,
                            critic_exploration=critic_exploration,
                            actor_exploration_optimizer=actor_exploration_optimizer,
                            critic_exploration_optimizer=critic_exploration_optimizer,
                        )
                        cumulative_per_rank_gradient_steps += 1
                    train_step += world_size

                if aggregator and not aggregator.disabled:
                    aggregator.update("Params/exploration_amount_task", actor_task._get_expl_amount(policy_step))
                    aggregator.update(
                        "Params/exploration_amount_exploration", actor_exploration._get_expl_amount(policy_step)
                    )

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or update == num_updates):
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
                "actor_task": actor_task.state_dict(),
                "critic_task": critic_task.state_dict(),
                "ensembles": ensembles.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_task_optimizer": actor_task_optimizer.state_dict(),
                "critic_task_optimizer": critic_task_optimizer.state_dict(),
                "ensemble_optimizer": ensemble_optimizer.state_dict(),
                "ratio": ratio.state_dict(),
                "update": update * world_size,
                "batch_size": cfg.algo.per_rank_batch_size * world_size,
                "actor_exploration": actor_exploration.state_dict(),
                "critic_exploration": critic_exploration.state_dict(),
                "actor_exploration_optimizer": actor_exploration_optimizer.state_dict(),
                "critic_exploration_optimizer": critic_exploration_optimizer.state_dict(),
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
    # task test zero-shot
    if fabric.is_global_zero and cfg.algo.run_test:
        player.actor_type = "task"
        fabric_player = get_single_device_fabric(fabric)
        player.actor = fabric_player.setup_module(unwrap_fabric(actor_task))
        test(player, fabric, cfg, log_dir, "zero-shot")

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "ensembles": ensembles,
            "actor_exploration": actor_exploration,
            "critic_exploration": critic_exploration,
            "actor_task": actor_task,
            "critic_task": critic_task,
        }
        register_model(fabric, log_models, cfg, models_to_log)
