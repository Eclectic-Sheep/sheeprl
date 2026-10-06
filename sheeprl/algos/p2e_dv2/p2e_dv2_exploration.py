"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V2: the exploration phase.

Written on the shared training loop of `sheeprl.core`. The agent plays with an exploration actor, rewarded by the
disagreement of an ensemble of models of the dynamics (the novelty of the states); a task actor learns the task from
the same experience (zero-shot). The writer is the one of Dreamer-V2 (`sheeprl.algos.dreamer_v2.dreamer_v2`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import gymnasium as gym
import torch
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch import Tensor, nn
from torch.distributions import Bernoulli, Distribution, Independent, Normal, OneHotCategorical
from torch.distributions.utils import logits_to_probs
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import DreamerV2Policy, WorldModel
from sheeprl.algos.dreamer_v2.dreamer_v2 import SequenceWriter, check_keys
from sheeprl.algos.dreamer_v2.loss import reconstruction_loss
from sheeprl.algos.dreamer_v2.utils import (
    MAX_SAMPLED_BATCHES,
    actor_objective,
    build_optimizer,
    compute_lambda_values,
    test,
)
from sheeprl.algos.p2e_dv2.agent import build_agent
from sheeprl.core import Algorithm, TrainSchedule, TrainState, env_buffer_size, run, sequence_store
from sheeprl.data.store import ReplayStore
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.distribution import entropy as policy_entropy
from sheeprl.utils.env import actions_dim_of
from sheeprl.utils.fabric import autocast_cache_scope, get_single_device_fabric, update
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.model import ema_
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import unwrap_fabric

# Decomment the following line if you are using MineDojo on an headless machine
# os.environ["MINEDOJO_HEADLESS"] = "1"


@dataclass
class P2EDV2ExplorationState(TrainState):
    world_model: WorldModel
    # Predict the next stochastic state: their disagreement is the intrinsic reward
    ensembles: nn.ModuleList
    # Learn the task from the experience of the exploration (zero-shot)
    actor_task: nn.Module
    critic_task: nn.Module
    target_critic_task: nn.Module
    # Play in the environments, rewarded by the intrinsic reward
    actor_exploration: nn.Module
    critic_exploration: nn.Module
    target_critic_exploration: nn.Module
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
    discrete_size: int,
    recurrent_state_size: int,
    use_continues: bool,
    gamma: float,
    kl_balancing_alpha: float,
    kl_free_nats: float,
    kl_free_avg: bool,
    kl_regularizer: float,
    discount_scale_factor: float,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """The loss of the world model of DreamerV2 on a batch of sequences, whose reward and continue models learn from
    the latent states without changing them: the latent states learn only to reconstruct the observations. Can be
    compiled (`algo.compile`).

    Returns:
        The loss, the posteriors and the recurrent states of the batch, and the terms of the loss with the logits of the
        posteriors and of the priors.
    """
    sequence_length, batch_size = data["actions"].shape[:2]
    device = data["actions"].device
    batch_obs = {k: data[k] / 255 - 0.5 for k in cnn_keys}
    batch_obs.update({k: data[k] for k in mlp_keys})
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
    # The outputs of every step are concatenated at the end of the unroll: writing them in place into preallocated
    # tensors makes the backward pass copy the gradient of the whole tensor at every step
    recurrent_states = []
    priors_logits = []
    posteriors = []
    posteriors_logits = []
    embedded_obs = world_model.encoder(batch_obs)
    for i in range(0, sequence_length):
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
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

    decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)
    po = {k: Independent(Normal(rec_obs, 1), len(rec_obs.shape[2:])) for k, rec_obs in decoded_information.items()}
    pr = Independent(Normal(world_model.reward_model(latent_states.detach()), 1), 1)
    if use_continues:
        pc = Independent(Bernoulli(logits=world_model.continue_model(latent_states.detach())), 1)
        continues_targets = (1 - data["terminated"]) * gamma
    else:
        pc = continues_targets = None

    priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        po,
        batch_obs,
        pr,
        data["rewards"],
        priors_logits,
        posteriors_logits,
        kl_balancing_alpha,
        kl_free_nats,
        kl_free_avg,
        kl_regularizer,
        pc,
        continues_targets,
        discount_scale_factor,
    )
    losses = {
        "kl": kl,
        "state_loss": state_loss,
        "reward_loss": reward_loss,
        "observation_loss": observation_loss,
        "continue_loss": continue_loss,
        "posteriors_logits": posteriors_logits,
        "priors_logits": priors_logits,
    }
    return rec_loss, posteriors, recurrent_states, losses


def ensemble_loss(ensembles: nn.ModuleList, posteriors: Tensor, recurrent_states: Tensor, actions: Tensor) -> Tensor:
    """The loss of the ensembles, each predicting the next posterior from the latent state and the action. Can be
    compiled (`algo.compile`)."""
    sequence_length, batch_size = actions.shape[:2]
    loss = 0.0
    for ens in ensembles:
        out = ens(
            torch.cat(
                (
                    posteriors.view(*posteriors.shape[:-2], -1).detach(),
                    recurrent_states.detach(),
                    actions.detach(),
                ),
                -1,
            )
        )[:-1]
        next_obs_embedding_dist = Independent(Normal(out, 1), 1)
        loss -= next_obs_embedding_dist.log_prob(posteriors.view(sequence_length, batch_size, -1).detach()[1:]).mean()
    return loss


def behaviour_imagine(
    world_model: WorldModel,
    actor: _FabricModule,
    target_critic: nn.Module,
    ensembles: Optional[nn.ModuleList],
    posteriors: Tensor,
    recurrent_states: Tensor,
    terminated: Tensor,
    *,
    horizon: int,
    use_continues: bool,
    gamma: float,
    lmbda: float,
    intrinsic_reward_multiplier: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The trajectories imagined from the latent states of the batch with `actor`, rewarded by the disagreement of the
    `ensembles` (the exploration) or by the reward model (the task, without ensembles), and their lambda-values from the
    target critic. Can be compiled (`algo.compile`).

    Returns:
        The imagined trajectories and actions, the values of the target critic, the lambda-values, the discounts of the
        actor and critic losses, and the rewards.
    """
    stoch_state_size = posteriors.shape[-2] * posteriors.shape[-1]
    recurrent_state_size = recurrent_states.shape[-1]
    imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    imagined_trajectories = [imagined_latent_state]
    imagined_actions = [torch.zeros(1, imagined_prior.shape[1], sum(actor.actions_dim), device=posteriors.device)]
    for i in range(1, horizon + 1):
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions.append(actions)
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories.append(imagined_latent_state)
    imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
    imagined_actions = torch.cat(imagined_actions, dim=0)
    predicted_target_values = target_critic(imagined_trajectories)
    if ensembles is not None:
        # The intrinsic reward: the disagreement of the ensembles on the next posterior
        next_obs_embedding = torch.stack(
            [ens(torch.cat((imagined_trajectories, imagined_actions), -1)) for ens in ensembles], dim=0
        )
        rewards = next_obs_embedding.var(0).mean(-1, keepdim=True) * intrinsic_reward_multiplier
    else:
        rewards = world_model.reward_model(imagined_trajectories)
    if use_continues:
        continues = logits_to_probs(logits=world_model.continue_model(imagined_trajectories), is_binary=True)
        true_continue = (1 - terminated).flatten().reshape(1, -1, 1) * gamma
        continues = torch.cat((true_continue, continues[1:]))
    else:
        continues = torch.ones_like(rewards.detach()) * gamma
    lambda_values = compute_lambda_values(
        rewards[:-1],
        predicted_target_values[:-1],
        continues[:-1],
        bootstrap=predicted_target_values[-1:],
        horizon=horizon,
        lmbda=lmbda,
    )
    with torch.no_grad():
        discount = torch.cumprod(torch.cat((torch.ones_like(continues[:1]), continues[:-1]), 0), 0)
    return imagined_trajectories, imagined_actions, predicted_target_values, lambda_values, discount, rewards


def behaviour_actor_loss(
    actor: _FabricModule,
    imagined_trajectories: Tensor,
    imagined_actions: Tensor,
    predicted_target_values: Tensor,
    lambda_values: Tensor,
    discount: Tensor,
    *,
    objective_mix: Optional[float],
    is_continuous: bool,
    actions_dim: Tuple[int, ...],
    ent_coef: float,
) -> Tensor:
    """The loss of an actor (Eq. 6 of DreamerV2): the dynamics backpropagation of the lambda-values and REINFORCE, mixed
    by `objective_mix`, with the entropy of the policies. Can be compiled (`algo.compile`)."""
    policies: Sequence[Distribution] = actor(imagined_trajectories[:-2].detach())[1]

    def reinforce() -> Tensor:
        advantage = (lambda_values[1:] - predicted_target_values[:-2]).detach()
        logprobs = [
            p.log_prob(imgnd_act[1:-1].detach()).unsqueeze(-1)
            for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, -1))
        ]
        return torch.stack(logprobs, -1).sum(-1) * advantage

    objective = actor_objective(objective_mix, is_continuous, lambda_values[1:], reinforce)
    # The tanh-normal policies have no analytic entropy: it is estimated from samples
    entropy = ent_coef * torch.stack([policy_entropy(p) for p in policies], -1).sum(-1)
    return -torch.mean(discount[:-2] * (objective + entropy.unsqueeze(-1)))


def behaviour_critic_loss(
    critic: _FabricModule, imagined_trajectories: Tensor, lambda_values: Tensor, discount: Tensor
) -> Tensor:
    """The loss of a critic (Eq. 5 of DreamerV2), on the imagined trajectories. Can be compiled (`algo.compile`)."""
    qv = Independent(Normal(critic(imagined_trajectories.detach())[:-1], 1), 1)
    return -torch.mean(discount[:-1, ..., 0] * qv.log_prob(lambda_values.detach()))


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
    cfg: Dict[str, Any],
    ensembles: _FabricModule,
    ensemble_optimizer: _FabricOptimizer,
    actor_exploration: _FabricModule,
    critic_exploration: _FabricModule,
    target_critic_exploration: nn.Module,
    actor_exploration_optimizer: _FabricOptimizer,
    critic_exploration_optimizer: _FabricOptimizer,
    is_continuous: bool,
    actions_dim: Sequence[int],
) -> Dict[str, Tensor]:
    """One gradient step of Plan2Explore on DreamerV2 (Algorithm 1 of the paper): the world model, the ensembles, the
    exploration actor and critic (with the intrinsic rewards), the task actor and critic (zero-shot). The losses are
    compiled when `algo.compile.enabled` is set.

    Returns:
        The metrics of the step.
    """
    data = {k: data[k] for k in data.keys()}
    # Every sequence starts from the zero state, as an episode does
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])
    mark_gradient_step(fabric, cfg)
    world_model_cfg = cfg.algo.world_model
    use_continues = bool(world_model_cfg.use_continues and world_model.continue_model)

    # Dynamic Learning
    # Cast the weights to low precision once for the whole forward pass, not at every step of the unroll
    with autocast_cache_scope(fabric):
        rec_loss, posteriors, recurrent_states, losses = compiled(world_model_loss, fabric, cfg)(
            world_model,
            data,
            cnn_keys=tuple(cfg.algo.cnn_keys.encoder),
            mlp_keys=tuple(cfg.algo.mlp_keys.encoder),
            stochastic_size=world_model_cfg.stochastic_size,
            discrete_size=world_model_cfg.discrete_size,
            recurrent_state_size=world_model_cfg.recurrent_model.recurrent_state_size,
            use_continues=use_continues,
            gamma=cfg.algo.gamma,
            kl_balancing_alpha=world_model_cfg.kl_balancing_alpha,
            kl_free_nats=world_model_cfg.kl_free_nats,
            kl_free_avg=world_model_cfg.kl_free_avg,
            kl_regularizer=world_model_cfg.kl_regularizer,
            discount_scale_factor=world_model_cfg.discount_scale_factor,
        )
    world_grad = update(fabric, rec_loss, world_optimizer, world_model_cfg.clip_gradients, error_if_nonfinite=False)

    # Ensemble Learning
    with autocast_cache_scope(fabric):
        loss = compiled(ensemble_loss, fabric, cfg)(ensembles, posteriors, recurrent_states, data["actions"])
    ensemble_grad = update(
        fabric, loss, ensemble_optimizer, cfg.algo.ensembles.clip_gradients, error_if_nonfinite=False
    )

    behaviour = dict(
        horizon=cfg.algo.horizon,
        use_continues=use_continues,
        gamma=cfg.algo.gamma,
        lmbda=cfg.algo.lmbda,
        intrinsic_reward_multiplier=cfg.algo.intrinsic_reward_multiplier,
    )
    actor_kwargs = dict(
        objective_mix=cfg.algo.actor.objective_mix,
        is_continuous=is_continuous,
        actions_dim=tuple(int(dim) for dim in actions_dim),
        ent_coef=cfg.algo.actor.ent_coef,
    )
    # Behaviour Learning Exploration (the same compiled functions as the task: other modules, other graphs)
    with autocast_cache_scope(fabric):
        (
            imagined_trajectories,
            imagined_actions,
            predicted_target_values_exploration,
            lambda_values_exploration,
            discount,
            intrinsic_reward,
        ) = compiled(behaviour_imagine, fabric, cfg)(
            world_model,
            actor_exploration,
            target_critic_exploration,
            ensembles,
            posteriors,
            recurrent_states,
            data["terminated"],
            **behaviour,
        )
        policy_loss_exploration = compiled(behaviour_actor_loss, fabric, cfg)(
            actor_exploration,
            imagined_trajectories,
            imagined_actions,
            predicted_target_values_exploration,
            lambda_values_exploration,
            discount,
            **actor_kwargs,
        )
    actor_exploration_grad = update(
        fabric,
        policy_loss_exploration,
        actor_exploration_optimizer,
        cfg.algo.actor.clip_gradients,
        error_if_nonfinite=False,
    )
    with autocast_cache_scope(fabric):
        value_loss_exploration = compiled(behaviour_critic_loss, fabric, cfg)(
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

    # Behaviour Learning Task
    with autocast_cache_scope(fabric):
        (
            imagined_trajectories,
            imagined_actions,
            predicted_target_values_task,
            lambda_values_task,
            discount,
            _,
        ) = compiled(behaviour_imagine, fabric, cfg)(
            world_model,
            actor_task,
            target_critic_task,
            None,
            posteriors,
            recurrent_states,
            data["terminated"],
            **behaviour,
        )
        policy_loss_task = compiled(behaviour_actor_loss, fabric, cfg)(
            actor_task,
            imagined_trajectories,
            imagined_actions,
            predicted_target_values_task,
            lambda_values_task,
            discount,
            **actor_kwargs,
        )
    actor_task_grad = update(
        fabric, policy_loss_task, actor_task_optimizer, cfg.algo.actor.clip_gradients, error_if_nonfinite=False
    )
    with autocast_cache_scope(fabric):
        value_loss_task = compiled(behaviour_critic_loss, fabric, cfg)(
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
        "Rewards/intrinsic": intrinsic_reward.detach().mean(),
        "Values_exploration/predicted_values": predicted_target_values_exploration.detach().mean(),
        "Values_exploration/lambda_values": lambda_values_exploration.detach().mean(),
        "Loss/policy_loss_exploration": policy_loss_exploration.detach(),
        "Loss/value_loss_exploration": value_loss_exploration.detach(),
        "Loss/policy_loss_task": policy_loss_task.detach(),
        "Loss/value_loss_task": value_loss_task.detach(),
    }
    if not MetricAggregator.disabled:
        for name, logits in (("post", losses["posteriors_logits"]), ("prior", losses["priors_logits"])):
            entropy = Independent(OneHotCategorical(logits=logits.detach()), 1).entropy()
            metrics[f"State/{name}_entropy"] = entropy.mean().detach()
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


class P2EDV2Exploration(Algorithm):
    """Every iteration plays one step in every environment with the exploration actor, then does `algo.replay_ratio`
    gradient steps per policy step, each on its own batch of sequences: the world model, the ensembles, the
    exploration actor and critic, the task actor and critic."""

    off_policy = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        # These arguments cannot be changed
        cfg.env.screen_size = 64
        cfg.env.frame_stack = 1
        cfg.algo.player.actor_type = "exploration"

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[P2EDV2ExplorationState, ReplayStore]:
        cfg = self.cfg
        fabric = self.fabric
        self.actions_dim, self.is_continuous = actions_dim_of(action_space)
        check_keys(fabric, cfg, obs_space)

        (
            world_model,
            ensembles,
            actor_task,
            critic_task,
            target_critic_task,
            actor_exploration,
            critic_exploration,
            target_critic_exploration,
            self._policy,
        ) = build_agent(fabric, self.actions_dim, self.is_continuous, cfg, obs_space)

        def optimizer(optimizer_cfg: Dict[str, Any], module: nn.Module) -> Optimizer:
            return fabric.setup_optimizers(build_optimizer(optimizer_cfg, module.parameters()))

        state = P2EDV2ExplorationState(
            world_model=world_model,
            ensembles=ensembles,
            actor_task=actor_task,
            critic_task=critic_task,
            target_critic_task=target_critic_task,
            actor_exploration=actor_exploration,
            critic_exploration=critic_exploration,
            target_critic_exploration=target_critic_exploration,
            world_optimizer=optimizer(cfg.algo.world_model.optimizer, world_model),
            actor_task_optimizer=optimizer(cfg.algo.actor.optimizer, actor_task),
            critic_task_optimizer=optimizer(cfg.algo.critic.optimizer, critic_task),
            ensemble_optimizer=optimizer(cfg.algo.ensembles.optimizer, ensembles),
            actor_exploration_optimizer=optimizer(cfg.algo.actor.optimizer, actor_exploration),
            critic_exploration_optimizer=optimizer(cfg.algo.critic.optimizer, critic_exploration),
        )
        return state, sequence_store(
            fabric,
            cfg,
            log_dir,
            env_buffer_size(fabric, cfg, dry_run_size=4),
            cfg.algo.per_rank_sequence_length,
            buffer_type=cfg.buffer.type,
        )

    def policy(self, state: P2EDV2ExplorationState) -> DreamerV2Policy:
        """The policy to play with: the exploration actor, which it shares its weights with (`build_agent`)."""
        return self._policy

    def task_policy(self, state: P2EDV2ExplorationState) -> DreamerV2Policy:
        """The policy of the task actor (zero-shot), to test it after the exploration."""
        policy = self.policy(state)
        policy.actor_type = "task"
        policy.actor = get_single_device_fabric(self.fabric).setup_module(unwrap_fabric(state.actor_task))
        return policy

    def test(self, state: TrainState, log_dir: str, policy_step: int = 0, test_name: str = "") -> None:
        # The task actor plays
        test(self.task_policy(state), self.fabric, self.cfg, log_dir, test_name, policy_step=policy_step)

    def writer(self, state: P2EDV2ExplorationState, policy: DreamerV2Policy) -> SequenceWriter:
        return SequenceWriter(self.cfg, self.actions_dim)

    def batches(
        self,
        state: P2EDV2ExplorationState,
        buffer: ReplayStore,
        n_steps: int,
        iteration: int,
    ) -> Iterator[Dict[str, Tensor]]:
        yield from buffer.batches(n_steps, self.cfg.algo.per_rank_batch_size, MAX_SAMPLED_BATCHES)

    def train_step(self, state: P2EDV2ExplorationState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        # The target critics are copies of the critics, every `critic.per_rank_target_network_update_freq` gradient
        # steps
        if step % self.cfg.algo.critic.per_rank_target_network_update_freq == 0:
            ema_(state.target_critic_task, state.critic_task, 1)
            ema_(state.target_critic_exploration, state.critic_exploration, 1)
        metrics = train(
            self.fabric,
            state.world_model,
            state.actor_task,
            state.critic_task,
            state.target_critic_task,
            state.world_optimizer,
            state.actor_task_optimizer,
            state.critic_task_optimizer,
            batch,
            self.cfg,
            ensembles=state.ensembles,
            ensemble_optimizer=state.ensemble_optimizer,
            actor_exploration=state.actor_exploration,
            critic_exploration=state.critic_exploration,
            target_critic_exploration=state.target_critic_exploration,
            actor_exploration_optimizer=state.actor_exploration_optimizer,
            critic_exploration_optimizer=state.critic_exploration_optimizer,
            is_continuous=self.is_continuous,
            actions_dim=self.actions_dim,
        )
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = P2EDV2Exploration(fabric, cfg)
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
            "target_critic_exploration": state.target_critic_exploration,
            "actor_task": state.actor_task,
            "critic_task": state.critic_task,
            "target_critic_task": state.target_critic_task,
        }
        register_model(fabric, log_models, cfg, models_to_log)
