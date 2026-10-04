"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V1: the exploration phase.

Written on the shared training loop of `sheeprl.core`. The agent plays with an exploration actor, rewarded by the
disagreement of an ensemble of models of the dynamics (the novelty of the states); a task actor learns the task from
the same experience (zero-shot). The player is the one of Dreamer-V1 (`sheeprl.algos.dreamer_v1.dreamer_v1`).
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

from sheeprl.algos.dreamer_v1.agent import PlayerDV1, WorldModel
from sheeprl.algos.dreamer_v1.dreamer_v1 import SequencePlayer, build_buffer, sample_batches_of_iteration
from sheeprl.algos.dreamer_v1.loss import actor_loss, critic_loss, reconstruction_loss
from sheeprl.algos.dreamer_v1.utils import add_is_first, compute_lambda_values
from sheeprl.algos.dreamer_v2.dreamer_v2 import actions_dim_of, check_keys
from sheeprl.algos.dreamer_v2.utils import test
from sheeprl.algos.p2e_dv1.agent import build_agent
from sheeprl.core import Algorithm, Metrics, TrainSchedule, TrainState, run
from sheeprl.data.buffers import EnvIndependentReplayBuffer
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

    # Every sequence starts from the zero state, as an episode does: its first step is treated as the first one of an
    # episode (its action, which comes from before the sequence, is not seen)
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # Dynamic Learning
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, device=device)
    # Cast the weights to low precision once for the whole forward pass, not at every step of the unroll
    with autocast_cache_scope(fabric):
        # The outputs of every step are concatenated at the end of the unroll: writing them in place into
        # preallocated tensors makes the backward pass copy the gradient of the whole tensor at every step
        recurrent_states = []
        posteriors = []
        priors = []
        posteriors_mean = []
        posteriors_std = []
        priors_mean = []
        priors_std = []
        embedded_obs = world_model.encoder(batch_obs)

        for i in range(0, sequence_length):
            recurrent_state, posterior, prior, posterior_mean_std, prior_mean_std = world_model.rssm.dynamic(
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
            priors.append(prior)
        recurrent_states = torch.cat(recurrent_states, dim=0)
        posteriors = torch.cat(posteriors, dim=0)
        posteriors_mean = torch.cat(posteriors_mean, dim=0)
        posteriors_std = torch.cat(posteriors_std, dim=0)
        priors_mean = torch.cat(priors_mean, dim=0)
        priors_std = torch.cat(priors_std, dim=0)
        priors = torch.cat(priors, dim=0)
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
    world_grad = update(
        fabric, rec_loss, world_optimizer, cfg.algo.world_model.clip_gradients, error_if_nonfinite=False
    )

    # Ensemble Learning
    with autocast_cache_scope(fabric):
        loss = 0.0
        for ens in ensembles:
            out = ens(torch.cat((posteriors.detach(), recurrent_states.detach(), data["actions"].detach()), -1))[:-1]
            next_obs_embedding_dist = Independent(Normal(out, 1), 1)
            loss -= next_obs_embedding_dist.log_prob(embedded_obs.detach()[1:]).mean()
    ensemble_grad = update(
        fabric, loss, ensemble_optimizer, cfg.algo.ensembles.clip_gradients, error_if_nonfinite=False
    )

    # Behaviour Learning Exploration
    with autocast_cache_scope(fabric):
        imagined_prior = posteriors.detach().reshape(1, -1, stochastic_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        # the imagined states and actions are concatenated at the end of the imagination
        imagined_trajectories = []
        # initialize the list of imagined actions, they are used to compute the intrinsic reward
        imagined_actions = []

        # imagine trajectories in the latent space
        for i in range(cfg.algo.horizon):
            actions = torch.cat(actor_exploration(imagined_latent_state.detach())[0], dim=-1)
            imagined_actions.append(actions)
            imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories.append(imagined_latent_state)
        imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
        imagined_actions = torch.cat(imagined_actions, dim=0)
        predicted_values_exploration = critic_exploration(imagined_trajectories)

        # Predict intrinsic reward
        # The intrinsic reward is not detached from the imagined trajectories, as in the reference
        # implementation: with continuous actions the exploration actor is trained by backpropagating
        # the lambda-values, intrinsic rewards included, through the dynamics
        next_obs_embedding = torch.stack(
            [ens(torch.cat((imagined_trajectories, imagined_actions), -1)) for ens in ensembles], dim=0
        )

        # next_obs_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
        intrinsic_reward = next_obs_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier

        if cfg.algo.world_model.use_continues and world_model.continue_model:
            predicted_continues = logits_to_probs(
                logits=world_model.continue_model(imagined_trajectories), is_binary=True
            )
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
            discount = torch.cumprod(
                torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0
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
        qv = Independent(Normal(critic_exploration(imagined_trajectories.detach())[:-1], 1), 1)
        value_loss_exploration = critic_loss(qv, lambda_values_exploration.detach(), discount[..., 0])
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
        imagined_prior = posteriors.detach().reshape(1, -1, stochastic_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories = []
        for i in range(cfg.algo.horizon):
            actions = torch.cat(actor_task(imagined_latent_state.detach())[0], dim=-1)
            imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories.append(imagined_latent_state)
        imagined_trajectories = torch.cat(imagined_trajectories, dim=0)

        predicted_values_task = critic_task(imagined_trajectories)
        predicted_rewards = world_model.reward_model(imagined_trajectories)
        if cfg.algo.world_model.use_continues and world_model.continue_model:
            predicted_continues = logits_to_probs(
                logits=world_model.continue_model(imagined_trajectories), is_binary=True
            )
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
            discount = torch.cumprod(
                torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0
            )

        policy_loss_task = actor_loss(discount * lambda_values_task)
    actor_task_grad = update(
        fabric, policy_loss_task, actor_task_optimizer, cfg.algo.actor.clip_gradients, error_if_nonfinite=False
    )

    with autocast_cache_scope(fabric):
        qv = Independent(Normal(critic_task(imagined_trajectories.detach())[:-1], 1), 1)
        value_loss_task = critic_loss(qv, lambda_values_task.detach(), discount[..., 0])
    critic_task_grad = update(
        fabric, value_loss_task, critic_task_optimizer, cfg.algo.critic.clip_gradients, error_if_nonfinite=False
    )
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
        cfg.algo.player.actor_type = "exploration"
        # The policy steps at the end of the iteration, set for its last gradient step (see `batches`)
        self.exploration_step: Optional[int] = None

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[P2EDV1ExplorationState, EnvIndependentReplayBuffer]:
        cfg = self.cfg
        fabric = self.fabric
        self.actions_dim, self.is_continuous = actions_dim_of(action_space)
        check_keys(fabric, cfg, obs_space)

        world_model, ensembles, actor_task, critic_task, actor_exploration, critic_exploration, self._policy = (
            build_agent(fabric, self.actions_dim, self.is_continuous, cfg, obs_space)
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
        return state, build_buffer(fabric, cfg, log_dir, dry_run_size=2)

    def policy(self, state: P2EDV1ExplorationState) -> PlayerDV1:
        """The policy to play with: the exploration actor, which it shares its weights with (`build_agent`)."""
        return self._policy

    def task_policy(self, state: P2EDV1ExplorationState) -> PlayerDV1:
        """The policy of the task actor (zero-shot), to test it after the exploration."""
        policy = self.policy(state)
        policy.actor_type = "task"
        policy.actor = get_single_device_fabric(self.fabric).setup_module(unwrap_fabric(state.actor_task))
        return policy

    def load_store(self, saved: Any, store: EnvIndependentReplayBuffer) -> EnvIndependentReplayBuffer:
        # A buffer saved before `is_first` was stored
        return add_is_first(super().load_store(saved, store))

    def player(self, state: P2EDV1ExplorationState) -> SequencePlayer:
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
        self, state: P2EDV1ExplorationState, buffer: EnvIndependentReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        yield from sample_batches_of_iteration(self, buffer, n_steps, iteration)

    def train_step(self, state: P2EDV1ExplorationState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        metrics = Metrics()
        train(
            self.fabric,
            state.world_model,
            state.actor_task,
            state.critic_task,
            state.world_optimizer,
            state.actor_task_optimizer,
            state.critic_task_optimizer,
            batch,
            metrics,
            self.cfg,
            ensembles=state.ensembles,
            ensemble_optimizer=state.ensemble_optimizer,
            actor_exploration=state.actor_exploration,
            critic_exploration=state.critic_exploration,
            actor_exploration_optimizer=state.actor_exploration_optimizer,
            critic_exploration_optimizer=state.critic_exploration_optimizer,
        )
        if self.exploration_step is not None:
            metrics.values.update(exploration_amounts(state, self.exploration_step))
        return metrics.values


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

    # task test zero-shot
    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.task_policy(state), fabric, cfg, log_dir, "zero-shot", policy_step=policy_step)

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
