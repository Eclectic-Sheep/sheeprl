"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V1: the exploration phase.

Written on the shared training loop of `sheeprl.core`. The agent plays with an exploration actor, rewarded by the
disagreement of an ensemble of models of the dynamics (the novelty of the states); a task actor learns the task from
the same experience (zero-shot). The writer is the one of Dreamer-V1 (`sheeprl.algos.dreamer_v1.dreamer_v1`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
import torch
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch import Tensor, nn
from torch.distributions import Bernoulli, Independent, Normal
from torch.distributions.utils import logits_to_probs
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v1.agent import DreamerV1Policy, WorldModel
from sheeprl.algos.dreamer_v1.dreamer_v1 import DreamerV1Writer, imagine, sample_batches_of_iteration, value_loss_fn
from sheeprl.algos.dreamer_v1.loss import actor_loss, reconstruction_loss
from sheeprl.algos.dreamer_v1.utils import compute_lambda_values
from sheeprl.algos.dreamer_v2.dreamer_v2 import check_keys
from sheeprl.algos.p2e_dv1.agent import build_agent
from sheeprl.core import Algorithm, TrainSchedule, TrainState, env_buffer_size, run, sequence_store
from sheeprl.core.evaluation import run_test
from sheeprl.data.store import ReplayStore
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.env import actions_dim_of
from sheeprl.utils.fabric import autocast_cache_scope, get_single_device_fabric, update
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import unwrap_fabric

# Decomment the following line if you are using MineDojo on an headless machine
# os.environ["MINEDOJO_HEADLESS"] = "1"


@dataclass
class P2EDV1ExplorationState(TrainState):
    world_model: WorldModel
    # Predict the next embedded observation: their disagreement is the intrinsic reward
    ensembles: nn.ModuleList
    # Learn the task from the experience of the exploration (zero-shot)
    actor_task: nn.Module
    critic_task: nn.Module
    # Play in the environments, rewarded by the intrinsic reward
    actor_exploration: nn.Module
    critic_exploration: nn.Module
    world_optimizer: Optimizer
    actor_task_optimizer: Optimizer
    critic_task_optimizer: Optimizer
    ensemble_optimizer: Optimizer
    actor_exploration_optimizer: Optimizer
    critic_exploration_optimizer: Optimizer


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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """The loss of the world model of DreamerV1 on a batch of sequences, whose reward and continue models learn from
    the latent states without changing them: the latent states learn only to reconstruct the observations. Can be
    compiled (`algo.compile`).

    Returns:
        The loss, the posteriors and the recurrent states of the batch, its embedded observations (which the ensembles
        predict), and the terms of the loss with the means and the standard deviations of the posteriors and of the
        priors.
    """
    sequence_length, batch_size = data["actions"].shape[:2]
    device = data["actions"].device
    batch_obs = {k: data[k] / 255 - 0.5 for k in cnn_keys}
    batch_obs.update({k: data[k] for k in mlp_keys})
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, device=device)
    # The outputs of every step are concatenated at the end of the unroll: writing them in place into preallocated
    # tensors makes the backward pass copy the gradient of the whole tensor at every step
    recurrent_states = []
    posteriors = []
    posteriors_mean = []
    posteriors_std = []
    priors_mean = []
    priors_std = []
    embedded_obs = world_model.encoder(batch_obs)
    for i in range(0, sequence_length):
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
    latent_states = torch.cat((posteriors, recurrent_states), -1)

    decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)
    qo = {k: Independent(Normal(rec_obs, 1), len(rec_obs.shape[2:])) for k, rec_obs in decoded_information.items()}
    qr = Independent(Normal(world_model.reward_model(latent_states.detach()), 1), 1)
    if use_continues:
        qc = Independent(Bernoulli(logits=world_model.continue_model(latent_states.detach())), 1)
        continues_targets = (1 - data["terminated"]) * gamma
    else:
        qc = continues_targets = None
    posteriors_dist = Independent(Normal(posteriors_mean, posteriors_std), 1)
    priors_dist = Independent(Normal(priors_mean, priors_std), 1)

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
    losses = {
        "kl": kl,
        "state_loss": state_loss,
        "reward_loss": reward_loss,
        "observation_loss": observation_loss,
        "continue_loss": continue_loss,
        "posteriors_mean": posteriors_mean,
        "posteriors_std": posteriors_std,
        "priors_mean": priors_mean,
        "priors_std": priors_std,
    }
    return rec_loss, posteriors, recurrent_states, embedded_obs, losses


def ensemble_loss(
    ensembles: nn.ModuleList, posteriors: Tensor, recurrent_states: Tensor, actions: Tensor, embedded_obs: Tensor
) -> Tensor:
    """The loss of the ensembles, each predicting the next embedded observation from the latent state and the action.
    Can be compiled (`algo.compile`)."""
    loss = 0.0
    for ens in ensembles:
        out = ens(torch.cat((posteriors.detach(), recurrent_states.detach(), actions.detach()), -1))[:-1]
        next_obs_embedding_dist = Independent(Normal(out, 1), 1)
        loss -= next_obs_embedding_dist.log_prob(embedded_obs.detach()[1:]).mean()
    return loss


def exploration_imagine(
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    ensembles: nn.ModuleList,
    posteriors: Tensor,
    recurrent_states: Tensor,
    *,
    horizon: int,
    use_continues: bool,
    gamma: float,
    lmbda: float,
    intrinsic_reward_multiplier: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The trajectories imagined from the latent states of the batch with the exploration actor, rewarded by the
    disagreement of the ensembles. Can be compiled (`algo.compile`).

    Returns:
        The imagined trajectories, their lambda-values, the discounts of the actor and critic losses, the intrinsic
        rewards and the values predicted by the critic.
    """
    stochastic_size = posteriors.shape[-1]
    recurrent_state_size = recurrent_states.shape[-1]
    imagined_prior = posteriors.detach().reshape(1, -1, stochastic_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    # the imagined states and actions are concatenated at the end of the imagination
    imagined_trajectories = []
    # the imagined actions are used to compute the intrinsic reward
    imagined_actions = []
    for i in range(horizon):
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions.append(actions)
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories.append(imagined_latent_state)
    imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
    imagined_actions = torch.cat(imagined_actions, dim=0)
    predicted_values = critic(imagined_trajectories)

    # The intrinsic reward is not detached from the imagined trajectories, as in the reference implementation: with
    # continuous actions the exploration actor is trained by backpropagating the lambda-values, intrinsic rewards
    # included, through the dynamics
    next_obs_embedding = torch.stack(
        [ens(torch.cat((imagined_trajectories, imagined_actions), -1)) for ens in ensembles], dim=0
    )
    # next_obs_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
    intrinsic_reward = next_obs_embedding.var(0).mean(-1, keepdim=True) * intrinsic_reward_multiplier
    if use_continues:
        predicted_continues = logits_to_probs(logits=world_model.continue_model(imagined_trajectories), is_binary=True)
    else:
        predicted_continues = torch.ones_like(intrinsic_reward.detach()) * gamma
    lambda_values = compute_lambda_values(
        intrinsic_reward,
        predicted_values,
        predicted_continues,
        last_values=predicted_values[-1],
        horizon=horizon,
        lmbda=lmbda,
    )
    with torch.no_grad():
        discount = torch.cumprod(torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0)
    return imagined_trajectories, lambda_values, discount, intrinsic_reward, predicted_values


def exploration_value_loss(
    critic: _FabricModule, imagined_trajectories: Tensor, lambda_values: Tensor, discount: Tensor
) -> Tensor:
    """The loss of the exploration critic, the one of the task critic (`value_loss_fn` of DreamerV1) in a function of
    its own: compiled with CUDA graphs, the outputs of a function are overwritten when it runs again in the same step.
    """
    return value_loss_fn(critic, imagined_trajectories, lambda_values, discount)


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor_task: _FabricModule,
    critic_task: _FabricModule,
    world_optimizer: _FabricOptimizer,
    actor_task_optimizer: _FabricOptimizer,
    critic_task_optimizer: _FabricOptimizer,
    data: Dict[str, Tensor],
    cfg: Dict[str, Any],
    ensembles: _FabricModule,
    ensemble_optimizer: _FabricOptimizer,
    actor_exploration: _FabricModule,
    critic_exploration: _FabricModule,
    actor_exploration_optimizer: _FabricOptimizer,
    critic_exploration_optimizer: _FabricOptimizer,
) -> Dict[str, Tensor]:
    """One gradient step of Plan2Explore on DreamerV1 (Algorithm 1 of the paper): the world model, the ensembles, the
    exploration actor and critic (with the intrinsic rewards), the task actor and critic (zero-shot). The losses are
    compiled when `algo.compile.enabled` is set.

    Returns:
        The metrics of the step.
    """
    # Every sequence starts from the zero state, as an episode does: its first step is treated as the first one of an
    # episode (its action, which comes from before the sequence, is not seen)
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])
    mark_gradient_step(fabric, cfg)
    world_model_cfg = cfg.algo.world_model
    use_continues = bool(world_model_cfg.use_continues and world_model.continue_model)

    # Dynamic Learning
    # Cast the weights to low precision once for the whole forward pass, not at every step of the unroll
    with autocast_cache_scope(fabric):
        rec_loss, posteriors, recurrent_states, embedded_obs, losses = compiled(world_model_loss, fabric, cfg)(
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
        )
    world_grad = update(fabric, rec_loss, world_optimizer, world_model_cfg.clip_gradients, error_if_nonfinite=False)

    # Ensemble Learning
    with autocast_cache_scope(fabric):
        loss = compiled(ensemble_loss, fabric, cfg)(
            ensembles, posteriors, recurrent_states, data["actions"], embedded_obs
        )
    ensemble_grad = update(
        fabric, loss, ensemble_optimizer, cfg.algo.ensembles.clip_gradients, error_if_nonfinite=False
    )

    # Behaviour Learning Exploration
    with autocast_cache_scope(fabric):
        (
            imagined_trajectories,
            lambda_values_exploration,
            discount,
            intrinsic_reward,
            predicted_values_exploration,
        ) = compiled(exploration_imagine, fabric, cfg)(
            world_model,
            actor_exploration,
            critic_exploration,
            ensembles,
            posteriors,
            recurrent_states,
            horizon=cfg.algo.horizon,
            use_continues=use_continues,
            gamma=cfg.algo.gamma,
            lmbda=cfg.algo.lmbda,
            intrinsic_reward_multiplier=cfg.algo.intrinsic_reward_multiplier,
        )
        policy_loss_exploration = actor_loss(discount * lambda_values_exploration)
    actor_exploration_grad = update(
        fabric,
        policy_loss_exploration,
        actor_exploration_optimizer,
        cfg.algo.actor.clip_gradients,
        error_if_nonfinite=False,
    )
    with autocast_cache_scope(fabric):
        value_loss_exploration = compiled(exploration_value_loss, fabric, cfg)(
            critic_exploration, imagined_trajectories, lambda_values_exploration, discount
        )
    critic_exploration_grad = update(
        fabric,
        value_loss_exploration,
        critic_exploration_optimizer,
        cfg.algo.critic.clip_gradients,
        error_if_nonfinite=False,
    )

    # reset the world_model gradients, to avoid interferences with task learning
    world_optimizer.zero_grad(set_to_none=True)

    # Behaviour Learning Task: the one of DreamerV1, from every step of the sequences
    with autocast_cache_scope(fabric):
        imagined_trajectories, lambda_values_task, discount = compiled(imagine, fabric, cfg)(
            world_model,
            actor_task,
            critic_task,
            posteriors.detach(),
            recurrent_states.detach(),
            horizon=cfg.algo.horizon,
            gamma=cfg.algo.gamma,
            lmbda=cfg.algo.lmbda,
            use_continues=use_continues,
        )
        policy_loss_task = actor_loss(discount * lambda_values_task)
    actor_task_grad = update(
        fabric, policy_loss_task, actor_task_optimizer, cfg.algo.actor.clip_gradients, error_if_nonfinite=False
    )
    with autocast_cache_scope(fabric):
        value_loss_task = compiled(value_loss_fn, fabric, cfg)(
            critic_task, imagined_trajectories, lambda_values_task, discount
        )
    critic_task_grad = update(
        fabric, value_loss_task, critic_task_optimizer, cfg.algo.critic.clip_gradients, error_if_nonfinite=False
    )

    metrics = {
        "Loss/world_model_loss": rec_loss.detach(),
        "Loss/observation_loss": losses["observation_loss"].detach(),
        "Loss/reward_loss": losses["reward_loss"].detach(),
        "Loss/state_loss": losses["state_loss"].detach(),
        "Loss/continue_loss": losses["continue_loss"].detach(),
        "State/kl": losses["kl"].mean().detach(),
        "Loss/ensemble_loss": loss.detach(),
        "Values_exploration/predicted_values": predicted_values_exploration.detach().mean(),
        "Values_exploration/lambda_values": lambda_values_exploration.detach().mean(),
        "Rewards/intrinsic": intrinsic_reward.detach().mean(),
        "Loss/policy_loss_exploration": policy_loss_exploration.detach(),
        "Loss/value_loss_exploration": value_loss_exploration.detach(),
        "Loss/policy_loss_task": policy_loss_task.detach(),
        "Loss/value_loss_task": value_loss_task.detach(),
    }
    if not MetricAggregator.disabled:
        posteriors_dist = Independent(Normal(losses["posteriors_mean"], losses["posteriors_std"]), 1)
        priors_dist = Independent(Normal(losses["priors_mean"], losses["priors_std"]), 1)
        metrics["State/post_entropy"] = posteriors_dist.entropy().mean().detach()
        metrics["State/prior_entropy"] = priors_dist.entropy().mean().detach()
    for name, grad in (
        ("Grads/world_model", world_grad),
        ("Grads/ensemble", ensemble_grad),
        ("Grads/actor_exploration", actor_exploration_grad),
        ("Grads/critic_exploration", critic_exploration_grad),
        ("Grads/actor_task", actor_task_grad),
        ("Grads/critic_task", critic_task_grad),
    ):
        if grad is not None:
            metrics[name] = grad.detach()

    # Reset everything
    actor_exploration_optimizer.zero_grad(set_to_none=True)
    critic_exploration_optimizer.zero_grad(set_to_none=True)
    actor_task_optimizer.zero_grad(set_to_none=True)
    critic_task_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)
    ensemble_optimizer.zero_grad(set_to_none=True)
    return metrics


class P2EDV1Exploration(Algorithm):
    """Every iteration plays one step in every environment with the exploration actor, then does `algo.replay_ratio`
    gradient steps per policy step, each on its own batch of sequences: the world model, the ensembles, the
    exploration actor and critic, the task actor and critic."""

    off_policy = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        # These arguments cannot be changed
        cfg.env.screen_size = 64
        cfg.env.frame_stack = 1
        cfg.algo.policy.actor_type = "exploration"
        # The policy steps at the end of the iteration, set for its last gradient step (see `batches`)
        self.exploration_step: Optional[int] = None

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[P2EDV1ExplorationState, ReplayStore]:
        cfg = self.cfg
        fabric = self.fabric
        self.actions_dim, self.is_continuous = actions_dim_of(action_space)
        check_keys(fabric, cfg, obs_space)

        world_model, ensembles, actor_task, critic_task, actor_exploration, critic_exploration, self._policy = (
            build_agent(fabric, self.actions_dim, self.is_continuous, cfg, obs_space, schedule=schedule)
        )

        def optimizer(optimizer_cfg: Dict[str, Any], module: nn.Module) -> Optimizer:
            return fabric.setup_optimizers(
                hydra.utils.instantiate(optimizer_cfg, params=module.parameters(), _convert_="all")
            )

        state = P2EDV1ExplorationState(
            world_model=world_model,
            ensembles=ensembles,
            actor_task=actor_task,
            critic_task=critic_task,
            actor_exploration=actor_exploration,
            critic_exploration=critic_exploration,
            world_optimizer=optimizer(cfg.algo.world_model.optimizer, world_model),
            actor_task_optimizer=optimizer(cfg.algo.actor.optimizer, actor_task),
            critic_task_optimizer=optimizer(cfg.algo.critic.optimizer, critic_task),
            ensemble_optimizer=optimizer(cfg.algo.ensembles.optimizer, ensembles),
            actor_exploration_optimizer=optimizer(cfg.algo.actor.optimizer, actor_exploration),
            critic_exploration_optimizer=optimizer(cfg.algo.critic.optimizer, critic_exploration),
        )
        self.schedule = schedule
        return state, sequence_store(
            fabric, cfg, log_dir, env_buffer_size(fabric, cfg, dry_run_size=2), cfg.algo.per_rank_sequence_length
        )

    def policy(self, state: P2EDV1ExplorationState) -> DreamerV1Policy:
        """The policy to play with: the exploration actor, which it shares its weights with (`build_agent`)."""
        return self._policy

    def task_policy(self, state: P2EDV1ExplorationState) -> DreamerV1Policy:
        """The policy of the task actor (zero-shot), to test it after the exploration."""
        policy = self.policy(state)
        policy.actor_type = "task"
        policy.actor = get_single_device_fabric(self.fabric).setup_module(unwrap_fabric(state.actor_task))
        return policy

    def test(self, state: TrainState, log_dir: str, policy_step: int = 0, test_name: str = "") -> None:
        # The task actor plays
        run_test(
            self.task_policy(state),
            self.fabric,
            self.cfg,
            log_dir,
            policy_step=policy_step,
            greedy=self.greedy_test,
            test_name=test_name,
        )

    def writer(self, state: P2EDV1ExplorationState, policy: DreamerV1Policy) -> DreamerV1Writer:
        return DreamerV1Writer(self.cfg, self.actions_dim)

    def batches(
        self, state: P2EDV1ExplorationState, buffer: ReplayStore, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        yield from sample_batches_of_iteration(self, buffer, n_steps, iteration)

    def train_step(self, state: P2EDV1ExplorationState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        metrics = train(
            self.fabric,
            state.world_model,
            state.actor_task,
            state.critic_task,
            state.world_optimizer,
            state.actor_task_optimizer,
            state.critic_task_optimizer,
            batch,
            self.cfg,
            ensembles=state.ensembles,
            ensemble_optimizer=state.ensemble_optimizer,
            actor_exploration=state.actor_exploration,
            critic_exploration=state.critic_exploration,
            actor_exploration_optimizer=state.actor_exploration_optimizer,
            critic_exploration_optimizer=state.critic_exploration_optimizer,
        )
        if self.exploration_step is not None:
            metrics.update(exploration_amounts(state, self.exploration_step))
        return metrics


def exploration_amounts(state: TrainState, policy_step: int) -> Dict[str, float]:
    """The amounts of exploration noise of the task and of the exploration actors after `policy_step` policy steps."""
    return {
        "Params/exploration_amount_task": state.actor_task._get_expl_amount(policy_step),
        "Params/exploration_amount_exploration": state.actor_exploration._get_expl_amount(policy_step),
    }


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = P2EDV1Exploration(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        algo.test(state, log_dir, policy_step=policy_step, test_name="zero-shot")

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": state.world_model,
            "ensembles": state.ensembles,
            "actor_exploration": state.actor_exploration,
            "critic_exploration": state.critic_exploration,
            "actor_task": state.actor_task,
            "critic_task": state.critic_task,
        }
        register_model(fabric, log_models, cfg, models_to_log)
