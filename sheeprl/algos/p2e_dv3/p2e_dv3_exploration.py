import copy
import os
import warnings
from typing import Any, Dict, Sequence

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, OneHotCategorical
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_v3.agent import WorldModel
from sheeprl.algos.dreamer_v3.loss import reconstruction_loss
from sheeprl.algos.dreamer_v3.utils import Moments, compute_lambda_values, prepare_obs, test
from sheeprl.algos.p2e_dv3.agent import build_agent
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.distribution import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    TwoHotEncodingDistribution,
)
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
    target_critic_task: nn.Module,
    world_optimizer: _FabricOptimizer,
    actor_task_optimizer: _FabricOptimizer,
    critic_task_optimizer: _FabricOptimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator,
    cfg: DictConfig,
    ensembles: _FabricModule,
    ensemble_optimizer: _FabricOptimizer,
    actor_exploration: _FabricModule,
    critics_exploration: Dict[str, Dict[str, Any]],
    actor_exploration_optimizer: _FabricOptimizer,
    moments_exploration: Dict[str, Moments],
    moments_task: Moments,
    is_continuous: bool,
    actions_dim: Sequence[int],
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

    This method is based on [sheeprl.algos.dreamer_v3.dreamer_v3](sheeprl.algos.dreamer_v3.dreamer_v3) algorithm,
    extending it to implement the
    [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960).

    Args:
        fabric (Fabric): the fabric instance.
        world_model (WorldModel): the world model wrapped with Fabric.
        actor_task (_FabricModule): the actor for solving the task.
        critic_task (_FabricModule): the critic for solving the task.
        target_critic_task (nn.Module): the target critic for solving the task.
        world_optimizer (_FabricOptimizer): the world optimizer.
        actor_task_optimizer (_FabricOptimizer): the actor optimizer for solving the task.
        critic_task_optimizer (_FabricOptimizer): the critic optimizer for solving the task.
        data (Dict[str, Tensor]): the batch of data to use for training.
        aggregator (MetricAggregator): the aggregator to print the metrics.
        cfg (DictConfig): the configs.
        ensembles (_FabricModule): the ensemble models.
        ensemble_optimizer (_FabricOptimizer): the optimizer of the ensemble models.
        actor_exploration (_FabricModule): the actor for exploration.
        critics_exploration (Dict[str, Dict[str, Any]]): the critic for exploration.
        actor_exploration_optimizer (_FabricOptimizer): the optimizer of the actor for exploration.
        is_continuous (bool): whether or not are continuous actions.
        actions_dim (Sequence[int]): the actions dimension.
    """
    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device
    data = {k: data[k] for k in data.keys()}
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    priors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)
    posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device)
    posteriors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)

    # embedded observations from the environment
    embedded_obs = world_model.encoder(batch_obs)

    for i in range(0, sequence_length):
        recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
            posterior, recurrent_state, batch_actions[i : i + 1], embedded_obs[i : i + 1], data["is_first"][i : i + 1]
        )
        recurrent_states[i] = recurrent_state
        priors_logits[i] = prior_logits
        posteriors[i] = posterior
        posteriors_logits[i] = posterior_logits
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

    # compute predictions for the observations
    reconstructed_obs: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)

    # compute the distribution over the reconstructed observations
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
    pr = TwoHotEncodingDistribution(world_model.reward_model(latent_states.detach()), dims=1)

    # Compute the distribution over the terminal steps, if required
    pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states.detach())), 1)
    continues_targets = 1 - data["terminated"]

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

    # world model optimization step
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
        continues_targets,
        cfg.algo.world_model.continue_scale_factor,
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

    # Free up space
    del posterior
    del prior_logits
    del recurrent_state
    del posterior_logits
    world_optimizer.zero_grad(set_to_none=True)

    # Ensemble Learning
    loss = 0.0
    ensemble_optimizer.zero_grad(set_to_none=True)
    for ens in ensembles:
        out = ens(
            torch.cat(
                (
                    posteriors.view(*posteriors.shape[:-2], -1).detach(),
                    recurrent_states.detach(),
                    data["actions"].detach(),
                ),
                -1,
            )
        )[:-1]
        next_state_embedding_dist = MSEDistribution(out, 1)
        loss -= next_state_embedding_dist.log_prob(posteriors.view(sequence_length, batch_size, -1).detach()[1:]).mean()
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
    actions = torch.cat(actor_exploration(imagined_latent_state.detach())[0], dim=-1)
    imagined_actions[0] = actions

    # imagine trajectories in the latent space
    for i in range(1, cfg.algo.horizon + 1):
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state
        actions = torch.cat(actor_exploration(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions

    advantages = []
    weights_sum = sum([c["weight"] for c in critics_exploration.values()])
    for k, critic in critics_exploration.items():
        # Predict values and continues
        predicted_values = TwoHotEncodingDistribution(critic["module"](imagined_trajectories), dims=1).mean
        continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
        true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
        continues = torch.cat((true_continue, continues[1:]))

        if critic["reward_type"] == "intrinsic":
            # Predict intrinsic reward
            next_state_embedding = torch.empty(
                len(ensembles),
                cfg.algo.horizon + 1,
                batch_size * sequence_length,
                stochastic_size * discrete_size,
                device=device,
            )
            for i, ens in enumerate(ensembles):
                next_state_embedding[i] = ens(
                    torch.cat((imagined_trajectories.detach(), imagined_actions.detach()), -1)
                )

            # next_state_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
            reward = next_state_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier
            if aggregator and not aggregator.disabled:
                aggregator.update(f"Rewards/intrinsic_{k}", reward.detach().cpu().mean())
        else:
            reward = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean

        lambda_values = compute_lambda_values(
            reward[1:],
            predicted_values[1:],
            continues[1:] * cfg.algo.gamma,
            lmbda=cfg.algo.lmbda,
        )
        critic["lambda_values"] = lambda_values
        baseline = predicted_values[:-1]
        offset, invscale = moments_exploration[k](lambda_values, fabric)
        normed_lambda_values = (lambda_values - offset) / invscale
        normed_baseline = (baseline - offset) / invscale
        advantages.append((normed_lambda_values - normed_baseline) * critic["weight"] / weights_sum)

        if aggregator and not aggregator.disabled:
            aggregator.update(f"Values_exploration/predicted_values_{k}", predicted_values.detach().cpu().mean())
            aggregator.update(f"Values_exploration/lambda_values_{k}", lambda_values.detach().cpu().mean())

    advantage = torch.stack(advantages, dim=0).sum(dim=0)
    with torch.no_grad():
        discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

    actor_exploration_optimizer.zero_grad(set_to_none=True)
    policies: Sequence[Distribution] = actor_exploration(imagined_trajectories.detach())[1]
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

    policy_loss_exploration = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
    fabric.backward(policy_loss_exploration)
    actor_grads_exploration = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads_exploration = fabric.clip_gradients(
            module=actor_exploration,
            optimizer=actor_exploration_optimizer,
            max_norm=cfg.algo.actor.clip_gradients,
            error_if_nonfinite=False,
        )
    actor_exploration_optimizer.step()

    for k, critic in critics_exploration.items():
        qv = TwoHotEncodingDistribution(critic["module"](imagined_trajectories.detach()[:-1]), dims=1)
        with torch.no_grad():
            predicted_target_values_expl = TwoHotEncodingDistribution(
                critic["target_module"](imagined_trajectories.detach()[:-1]), dims=1
            ).mean
        # Critic optimization. Eq. 10 in the paper
        critic["optimizer"].zero_grad(set_to_none=True)
        value_loss = -qv.log_prob(critic["lambda_values"].detach())
        value_loss = value_loss - qv.log_prob(predicted_target_values_expl.detach())
        value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

        fabric.backward(value_loss)
        critic_grads_exploration = None
        if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
            critic_grads_exploration = fabric.clip_gradients(
                module=critic["module"],
                optimizer=critic["optimizer"],
                max_norm=cfg.algo.critic.clip_gradients,
                error_if_nonfinite=False,
            )
        critic["optimizer"].step()
        if aggregator and not aggregator.disabled:
            if critic_grads_exploration:
                aggregator.update(f"Grads/critic_exploration_{k}", critic_grads_exploration.mean().detach())
            aggregator.update(f"Loss/value_loss_exploration_{k}", value_loss.detach())

    # reset the world_model gradients, to avoid interferences with task learning
    world_optimizer.zero_grad(set_to_none=True)

    # Behaviour Learning Task
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
    actions = torch.cat(actor_task(imagined_latent_state.detach())[0], dim=-1)
    imagined_actions[0] = actions

    # imagine trajectories in the latent space
    for i in range(1, cfg.algo.horizon + 1):
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state
        actions = torch.cat(actor_task(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions

    # Predict values, rewards and continues
    predicted_values = TwoHotEncodingDistribution(critic_task(imagined_trajectories), dims=1).mean
    predicted_rewards = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean
    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
    true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
    continues = torch.cat((true_continue, continues[1:]))

    lambda_values = compute_lambda_values(
        predicted_rewards[1:],
        predicted_values[1:],
        continues[1:] * cfg.algo.gamma,
        lmbda=cfg.algo.lmbda,
    )

    # Compute the discounts to multiply the lambda values to
    with torch.no_grad():
        discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

    actor_task_optimizer.zero_grad(set_to_none=True)
    policies: Sequence[Distribution] = actor_task(imagined_trajectories.detach())[1]

    baseline = predicted_values[:-1]
    offset, invscale = moments_task(lambda_values, fabric)
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
    policy_loss_task = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
    fabric.backward(policy_loss_task)
    actor_grads_task = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads_task = fabric.clip_gradients(
            module=actor_task,
            optimizer=actor_task_optimizer,
            max_norm=cfg.algo.actor.clip_gradients,
            error_if_nonfinite=False,
        )
    actor_task_optimizer.step()

    # Predict the values
    qv = TwoHotEncodingDistribution(critic_task(imagined_trajectories.detach()[:-1]), dims=1)
    with torch.no_grad():
        predicted_target_values_tsk = TwoHotEncodingDistribution(
            target_critic_task(imagined_trajectories.detach()[:-1]), dims=1
        ).mean

    # Critic optimization. Eq. 10 in the paper
    critic_task_optimizer.zero_grad(set_to_none=True)
    value_loss_task = -qv.log_prob(lambda_values.detach())
    value_loss_task = value_loss_task - qv.log_prob(predicted_target_values_tsk.detach())
    value_loss_task = torch.mean(value_loss_task * discount[:-1].squeeze(-1))

    fabric.backward(value_loss_task)
    critic_grads_task = None
    if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
        critic_grads_task = fabric.clip_gradients(
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
        aggregator.update(
            "State/post_entropy",
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update("Loss/ensemble_loss", loss.detach().cpu())
        aggregator.update("Loss/policy_loss_exploration", policy_loss_exploration.detach())
        aggregator.update("Loss/policy_loss_task", policy_loss_task.detach())
        aggregator.update("Loss/value_loss_task", value_loss_task.detach())
        if world_model_grads:
            aggregator.update("Grads/world_model", world_model_grads.mean().detach())
        if ensemble_grad:
            aggregator.update("Grads/ensemble", ensemble_grad.detach())
        if actor_grads_exploration:
            aggregator.update("Grads/actor_exploration", actor_grads_exploration.mean().detach())
        if actor_grads_task:
            aggregator.update("Grads/actor_task", actor_grads_task.mean().detach())
        if critic_grads_task:
            aggregator.update("Grads/critic_task", critic_grads_task.mean().detach())

    # Reset everything
    actor_exploration_optimizer.zero_grad(set_to_none=True)
    actor_task_optimizer.zero_grad(set_to_none=True)
    critic_task_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)
    ensemble_optimizer.zero_grad(set_to_none=True)
    for c in critics_exploration.values():
        c["optimizer"].zero_grad(set_to_none=True)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # These arguments cannot be changed
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

    (
        world_model,
        ensembles,
        actor_task,
        critic_task,
        target_critic_task,
        actor_exploration,
        critics_exploration,
        player,
    ) = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"] if cfg.checkpoint.resume_from else None,
        state["ensembles"] if cfg.checkpoint.resume_from else None,
        state["actor_task"] if cfg.checkpoint.resume_from else None,
        state["critic_task"] if cfg.checkpoint.resume_from else None,
        state["target_critic_task"] if cfg.checkpoint.resume_from else None,
        state["actor_exploration"] if cfg.checkpoint.resume_from else None,
        state["critics_exploration"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers
    world_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
    )
    actor_exploration_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer, params=actor_exploration.parameters(), _convert_="all"
    )
    for k, critic in critics_exploration.items():
        critic["optimizer"] = hydra.utils.instantiate(
            cfg.algo.critic.optimizer, params=critic["module"].parameters(), _convert_="all"
        )
    actor_task_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer, params=actor_task.parameters(), _convert_="all"
    )
    critic_task_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer, params=critic_task.parameters(), _convert_="all"
    )
    ensemble_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer, params=ensembles.parameters(), _convert_="all"
    )
    if cfg.checkpoint.resume_from:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_task_optimizer.load_state_dict(state["actor_task_optimizer"])
        critic_task_optimizer.load_state_dict(state["critic_task_optimizer"])
        ensemble_optimizer.load_state_dict(state["ensemble_optimizer"])
        actor_exploration_optimizer.load_state_dict(state["actor_exploration_optimizer"])
        for k, c in critics_exploration.items():
            c["optimizer"].load_state_dict(state[f"critic_exploration_optimizer_{k}"])
    (
        world_optimizer,
        actor_task_optimizer,
        critic_task_optimizer,
        ensemble_optimizer,
        actor_exploration_optimizer,
    ) = fabric.setup_optimizers(
        world_optimizer,
        actor_task_optimizer,
        critic_task_optimizer,
        ensemble_optimizer,
        actor_exploration_optimizer,
    )
    for k, critic in critics_exploration.items():
        critic["optimizer"] = fabric.setup_optimizers(critic["optimizer"])

    moments_exploration = {
        k: Moments(
            cfg.algo.actor.moments.decay,
            cfg.algo.actor.moments.max,
            cfg.algo.actor.moments.percentile.low,
            cfg.algo.actor.moments.percentile.high,
        )
        for k in critics_exploration.keys()
    }
    moments_task = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    if cfg.checkpoint.resume_from:
        for k, m in moments_exploration.items():
            m.load_state_dict(state[f"moments_exploration_{k}"])
        moments_task.load_state_dict(state["moments_task"])

    # Metrics
    # Since there could be more exploration critics, the key of the critic is added
    # to the metrics that the user has selected.
    for k, c in critics_exploration.items():
        if "Loss/value_loss_exploration" in cfg.metric.aggregator.metrics:
            cfg.metric.aggregator.metrics[f"Loss/value_loss_exploration_{k}"] = cfg.metric.aggregator.metrics[
                "Loss/value_loss_exploration"
            ]
        if "Values_exploration/predicted_values" in cfg.metric.aggregator.metrics:
            cfg.metric.aggregator.metrics[f"Values_exploration/predicted_values_{k}"] = cfg.metric.aggregator.metrics[
                "Values_exploration/predicted_values"
            ]
        if "Values_exploration/lambda_values" in cfg.metric.aggregator.metrics:
            cfg.metric.aggregator.metrics[f"Values_exploration/lambda_values_{k}"] = cfg.metric.aggregator.metrics[
                "Values_exploration/lambda_values"
            ]
        if "Grads/critic_exploration" in cfg.metric.aggregator.metrics:
            cfg.metric.aggregator.metrics[f"Grads/critic_exploration_{k}"] = cfg.metric.aggregator.metrics[
                "Grads/critic_exploration"
            ]
        if c["reward_type"] == "intrinsic" and "Rewards/intrinsic" in cfg.metric.aggregator.metrics:
            cfg.metric.aggregator.metrics[f"Rewards/intrinsic_{k}"] = cfg.metric.aggregator.metrics["Rewards/intrinsic"]
    # Remove general log keys from the aggregator
    cfg.metric.aggregator.metrics.pop("Loss/value_loss_exploration", None)
    cfg.metric.aggregator.metrics.pop("Values_exploration/predicted_values", None)
    cfg.metric.aggregator.metrics.pop("Values_exploration/lambda_values", None)
    cfg.metric.aggregator.metrics.pop("Grads/critic_exploration", None)
    cfg.metric.aggregator.metrics.pop("Rewards/intrinsic", None)
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * world_size) if not cfg.dry_run else 4
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
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
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
    total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
    prefill_steps = learning_starts - int(learning_starts > 0)
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // world_size
        learning_starts += start_iter
        prefill_steps += start_iter

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

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
    obs = envs.reset(seed=cfg.seed)[0]
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["terminated"])
    player.init_states()

    cumulative_per_rank_gradient_steps = 0
    for iter_num in range(start_iter, total_iters + 1):
        policy_step += policy_steps_per_iter

        with torch.inference_mode():
            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                # Sample an action given the observation received by the environment
                if (
                    iter_num <= learning_starts
                    and cfg.checkpoint.resume_from is None
                    and "minedojo" not in cfg.algo.actor.cls.lower()
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
                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None
                    real_actions = actions = player.get_actions(torch_obs, mask=mask)
                    actions = torch.cat(actions, -1).cpu().numpy()
                    if is_continuous:
                        real_actions = torch.stack(real_actions, dim=-1).cpu().numpy()
                    else:
                        real_actions = (
                            torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                        )

                step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
                rb.add(step_data, validate_args=cfg.buffer.validate_args)

                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                dones = np.logical_or(terminated, truncated).astype(np.uint8)

            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:
                for i, agent_roe in enumerate(infos["restart_on_exception"]):
                    if agent_roe and not dones[i]:
                        last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                        rb.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["terminated"][last_inserted_idx]
                        )
                        rb.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                            rb.buffer[i]["truncated"][last_inserted_idx]
                        )
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
            step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
            step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
            step_data["rewards"] = clip_rewards_fn(rewards)

            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
                reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

                # Reset already inserted step data
                step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
                player.init_states(dones_idxes)

        # Train the agent
        if iter_num >= learning_starts:
            ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
            per_rank_gradient_steps = ratio(ratio_steps / world_size)
            if per_rank_gradient_steps > 0:
                local_data = rb.sample_tensors(
                    cfg.algo.per_rank_batch_size,
                    sequence_length=cfg.algo.per_rank_sequence_length,
                    n_samples=per_rank_gradient_steps,
                    dtype=None,
                    device=fabric.device,
                    from_numpy=cfg.buffer.from_numpy,
                )
                # Start training
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    for i in range(per_rank_gradient_steps):
                        if (
                            cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                            == 0
                        ):
                            tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                            for cp, tcp in zip(critic_task.module.parameters(), target_critic_task.parameters()):
                                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                            for k in critics_exploration.keys():
                                for cp, tcp in zip(
                                    critics_exploration[k]["module"].module.parameters(),
                                    critics_exploration[k]["target_module"].parameters(),
                                ):
                                    tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                        batch = {k: v[i].float() for k, v in local_data.items()}
                        train(
                            fabric,
                            world_model,
                            actor_task,
                            critic_task,
                            target_critic_task,
                            world_optimizer,
                            actor_task_optimizer,
                            critic_task_optimizer,
                            batch,
                            aggregator,
                            cfg,
                            ensembles=ensembles,
                            ensemble_optimizer=ensemble_optimizer,
                            actor_exploration=actor_exploration,
                            critics_exploration=critics_exploration,
                            actor_exploration_optimizer=actor_exploration_optimizer,
                            is_continuous=is_continuous,
                            actions_dim=actions_dim,
                            moments_exploration=moments_exploration,
                            moments_task=moments_task,
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
            critics_exploration_state = {"critics_exploration": {}}
            for k, c in critics_exploration.items():
                critics_exploration_state["critics_exploration"][k] = {
                    "module": c["module"].state_dict(),
                    "target_module": c["target_module"].state_dict(),
                }
                critics_exploration_state[f"critic_exploration_optimizer_{k}"] = c["optimizer"].state_dict()
                critics_exploration_state[f"moments_exploration_{k}"] = moments_exploration[k].state_dict()
            state = {
                "world_model": world_model.state_dict(),
                "actor_task": actor_task.state_dict(),
                "critic_task": critic_task.state_dict(),
                "target_critic_task": target_critic_task.state_dict(),
                "ensembles": ensembles.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_task_optimizer": actor_task_optimizer.state_dict(),
                "critic_task_optimizer": critic_task_optimizer.state_dict(),
                "ensemble_optimizer": ensemble_optimizer.state_dict(),
                "ratio": ratio.state_dict(),
                "iter_num": iter_num * world_size,
                "batch_size": cfg.algo.per_rank_batch_size * world_size,
                "actor_exploration": actor_exploration.state_dict(),
                "actor_exploration_optimizer": actor_exploration_optimizer.state_dict(),
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
                "moments_task": moments_task.state_dict(),
                **critics_exploration_state,
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
        test(player, fabric, cfg, log_dir, "zero-shot", greedy=False)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "ensembles": ensembles,
            "actor_exploration": actor_exploration,
            "actor_task": actor_task,
            "critic_task": critic_task,
            "target_critic_task": target_critic_task,
            "moments_task": moments_task,
        }
        critics_to_log = {}
        for k, v in critics_exploration.items():
            critics_to_log["critic_exploration_" + k] = v["module"]
            critics_to_log["target_critic_exploration_" + k] = v["target_module"]
        critics_moments_to_log = {}
        for k, v in moments_exploration.items():
            critics_moments_to_log["moments_exploration_" + k] = v
        models_to_log.update(critics_to_log)
        models_to_log.update(critics_moments_to_log)
        register_model(fabric, log_models, cfg, models_to_log)
