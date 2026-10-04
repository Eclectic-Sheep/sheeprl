"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V3: the exploration phase.

Written on the shared training loop of `sheeprl.core`. The agent plays with an exploration actor, rewarded by the
disagreement of an ensemble of models of the dynamics (the novelty of the states) and optionally by the task rewards;
a task actor learns the task from the same experience (zero-shot). The player and the world-model and task phases of a
gradient step are the ones of Dreamer-V3 (`sheeprl.algos.dreamer_v3.dreamer_v3`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence, Tuple

import gymnasium as gym
import hydra
import torch
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions import Distribution, Independent
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.utils import actor_objective, env_buffer_size, sample_batches
from sheeprl.algos.dreamer_v3.agent import PlayerDV3, WorldModel, clip_actions
from sheeprl.algos.dreamer_v3.dreamer_v3 import SequencePlayer, behaviour_learning, world_model_learning
from sheeprl.algos.dreamer_v3.utils import Moments, compute_lambda_values, test
from sheeprl.algos.p2e_dv3.agent import build_agent
from sheeprl.core import Algorithm, TrainSchedule, TrainState, run
from sheeprl.core.algorithm import load_module_state_dict
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.distribution import BernoulliSafeMode, MSEDistribution, TwoHotEncodingDistribution
from sheeprl.utils.distribution import entropy as policy_entropy
from sheeprl.utils.fabric import autocast_cache_scope, get_single_device_fabric, update
from sheeprl.utils.model import ema_
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import unwrap_fabric

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
        cfg (DictConfig): the configs.
        ensembles (_FabricModule): the ensemble models.
        ensemble_optimizer (_FabricOptimizer): the optimizer of the ensemble models.
        actor_exploration (_FabricModule): the actor for exploration.
        critics_exploration (Dict[str, Dict[str, Any]]): the critic for exploration.
        actor_exploration_optimizer (_FabricOptimizer): the optimizer of the actor for exploration.
        is_continuous (bool): whether or not are continuous actions.
        actions_dim (Sequence[int]): the actions dimension.
    """
    metrics: Dict[str, Tensor] = {}
    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stoch_state_size = cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
    data = {k: data[k] for k in data.keys()}

    # Dynamic Learning: the one of DreamerV3, whose reward and continue models learn from the latent states without
    # changing them
    posteriors, recurrent_states, world_model_metrics = world_model_learning(
        fabric, cfg, world_model, world_optimizer, data, detach_heads=True
    )
    world_optimizer.zero_grad(set_to_none=True)

    # Ensemble Learning
    with autocast_cache_scope(fabric):
        loss = 0.0
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
            loss -= next_state_embedding_dist.log_prob(
                posteriors.view(sequence_length, batch_size, -1).detach()[1:]
            ).mean()
    ensemble_grad = update(
        fabric, loss, ensemble_optimizer, cfg.algo.ensembles.clip_gradients, error_if_nonfinite=False
    )

    # Behaviour Learning Exploration
    with autocast_cache_scope(fabric):
        imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        # the imagined states and actions are concatenated at the end of the imagination: the samples of the actor,
        # whose log-probabilities REINFORCE takes, and the continuous ones clipped as the recurrent model and the
        # ensembles take them (DreamerV3 clips them in its RSSM)
        action_clip = float(cfg.algo.actor.action_clip) if is_continuous else 0.0
        imagined_trajectories = [imagined_latent_state]
        actions = torch.cat(actor_exploration(imagined_latent_state.detach(), clip=False)[0], dim=-1)
        imagined_actions = [actions]
        clipped_actions = [clip_actions(actions, action_clip)]

        # imagine trajectories in the latent space
        for i in range(1, cfg.algo.horizon + 1):
            imagined_prior, recurrent_state = world_model.rssm.imagination(
                imagined_prior, recurrent_state, clipped_actions[-1]
            )
            imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories.append(imagined_latent_state)
            actions = torch.cat(actor_exploration(imagined_latent_state.detach(), clip=False)[0], dim=-1)
            imagined_actions.append(actions)
            clipped_actions.append(clip_actions(actions, action_clip))
        imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
        imagined_actions = torch.cat(imagined_actions, dim=0)
        clipped_actions = torch.cat(clipped_actions, dim=0)

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
                # The intrinsic reward is not detached from the imagined trajectories, as in the reference
                # implementation: with continuous actions the exploration actor is trained by backpropagating
                # the lambda-values, intrinsic rewards included, through the dynamics
                next_state_embedding = torch.stack(
                    [ens(torch.cat((imagined_trajectories, clipped_actions), -1)) for ens in ensembles], dim=0
                )

                # next_state_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
                reward = next_state_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier
                metrics[f"Rewards/intrinsic_{k}"] = reward.detach().mean()
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

            metrics[f"Values_exploration/predicted_values_{k}"] = predicted_values.detach().mean()
            metrics[f"Values_exploration/lambda_values_{k}"] = lambda_values.detach().mean()

        advantage = torch.stack(advantages, dim=0).sum(dim=0)
        with torch.no_grad():
            discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

        policies: Sequence[Distribution] = actor_exploration(imagined_trajectories.detach())[1]

        def reinforce() -> Tensor:
            return (
                torch.stack(
                    [
                        p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                        for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                    ],
                    dim=-1,
                ).sum(dim=-1)
                * advantage.detach()
            )

        # The dynamics backpropagation of the advantages and REINFORCE, mixed by `algo.actor.objective_mix`
        objective = actor_objective(cfg.algo.actor.objective_mix, is_continuous, advantage, reinforce)
        # The tanh-normal policies have no analytic entropy: it is estimated from samples
        entropy = cfg.algo.actor.ent_coef * torch.stack([policy_entropy(p) for p in policies], -1).sum(dim=-1)

        policy_loss_exploration = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
    actor_grads_exploration = update(
        fabric,
        policy_loss_exploration,
        actor_exploration_optimizer,
        cfg.algo.actor.clip_gradients,
        error_if_nonfinite=False,
    )

    for k, critic in critics_exploration.items():
        with autocast_cache_scope(fabric):
            qv = TwoHotEncodingDistribution(critic["module"](imagined_trajectories.detach()[:-1]), dims=1)
            with torch.no_grad():
                predicted_target_values_expl = TwoHotEncodingDistribution(
                    critic["target_module"](imagined_trajectories.detach()[:-1]), dims=1
                ).mean
            # Critic optimization. Eq. 10 in the paper
            value_loss = -qv.log_prob(critic["lambda_values"].detach())
            value_loss = value_loss - qv.log_prob(predicted_target_values_expl.detach())
            value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

        critic_grads_exploration = update(
            fabric, value_loss, critic["optimizer"], cfg.algo.critic.clip_gradients, error_if_nonfinite=False
        )
        if critic_grads_exploration is not None:
            metrics[f"Grads/critic_exploration_{k}"] = critic_grads_exploration.mean().detach()
        metrics[f"Loss/value_loss_exploration_{k}"] = value_loss.detach()

    # reset the world_model gradients, to avoid interferences with task learning
    world_optimizer.zero_grad(set_to_none=True)

    # Behaviour Learning Task: the one of DreamerV3
    task_metrics = behaviour_learning(
        fabric,
        cfg,
        world_model,
        actor_task,
        critic_task,
        target_critic_task,
        actor_task_optimizer,
        critic_task_optimizer,
        moments_task,
        posteriors,
        recurrent_states,
        data["terminated"],
        is_continuous,
        actions_dim,
    )
    for name, value in world_model_metrics.items():
        metrics[name] = value
    metrics["Loss/ensemble_loss"] = loss.detach()
    metrics["Loss/policy_loss_exploration"] = policy_loss_exploration.detach()
    metrics["Loss/policy_loss_task"] = task_metrics["policy_loss"]
    metrics["Loss/value_loss_task"] = task_metrics["value_loss"]
    if ensemble_grad is not None:
        metrics["Grads/ensemble"] = ensemble_grad.detach()
    if actor_grads_exploration is not None:
        metrics["Grads/actor_exploration"] = actor_grads_exploration.mean().detach()
    if "actor_grads" in task_metrics:
        metrics["Grads/actor_task"] = task_metrics["actor_grads"]
    if "critic_grads" in task_metrics:
        metrics["Grads/critic_task"] = task_metrics["critic_grads"]

    # Reset everything
    actor_exploration_optimizer.zero_grad(set_to_none=True)
    actor_task_optimizer.zero_grad(set_to_none=True)
    critic_task_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)
    ensemble_optimizer.zero_grad(set_to_none=True)
    for c in critics_exploration.values():
        c["optimizer"].zero_grad(set_to_none=True)
    return metrics


class ExplorationCritics(Dict[str, Dict[str, Any]]):
    """The critics of the exploration actor by name, each with its own rewards: intrinsic (the disagreement of the
    ensembles) or of the task. Each one is a dictionary with the critic (`module`), its target (`target_module`), its
    `optimizer`, the normalization of its returns (`moments`), the `weight` of its advantages in their sum and its
    `reward_type`. Their weights are saved as `{name: {"module": ..., "target_module": ...}}`."""

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            k: {"module": c["module"].state_dict(), "target_module": c["target_module"].state_dict()}
            for k, c in self.items()
        }

    def load_state_dict(self, state: Dict[str, Dict[str, Any]]) -> None:
        for k, c in self.items():
            load_module_state_dict(c["module"], state[k]["module"])
            load_module_state_dict(c["target_module"], state[k]["target_module"])


@dataclass
class P2EDV3ExplorationState(TrainState):
    world_model: WorldModel
    # Predict the next stochastic state: their disagreement is the intrinsic reward
    ensembles: nn.ModuleList
    # Learn the task from the experience of the exploration (zero-shot)
    actor_task: nn.Module
    critic_task: nn.Module
    target_critic_task: nn.Module
    # Plays in the environments
    actor_exploration: nn.Module
    world_optimizer: Optimizer
    actor_task_optimizer: Optimizer
    critic_task_optimizer: Optimizer
    ensemble_optimizer: Optimizer
    actor_exploration_optimizer: Optimizer
    moments_task: Moments
    critics_exploration: ExplorationCritics

    def state_dict(self) -> Dict[str, Any]:
        # The optimizer and the moments of each exploration critic have their own entry, named after the critic, as
        # in the checkpoints of the old training loop
        state = super().state_dict()
        for k, c in self.critics_exploration.items():
            state[f"critic_exploration_optimizer_{k}"] = c["optimizer"].state_dict()
            state[f"moments_exploration_{k}"] = c["moments"].state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        for k, c in self.critics_exploration.items():
            c["optimizer"].load_state_dict(state[f"critic_exploration_optimizer_{k}"])
            load_module_state_dict(c["moments"], state[f"moments_exploration_{k}"])


class P2EDV3Exploration(Algorithm):
    """Every iteration plays one step in every environment with the exploration actor, then does `algo.replay_ratio`
    gradient steps per policy step, each on its own batch of sequences: the world model, the ensembles, the
    exploration actor and critics, the task actor and critic."""

    off_policy = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        # These arguments cannot be changed
        cfg.env.frame_stack = 1
        cfg.algo.player.actor_type = "exploration"

        # The metrics of the exploration critics are logged for every critic, under the key `<metric>_<critic>`
        metrics = cfg.metric.aggregator.metrics
        for k, c in cfg.algo.critics_exploration.items():
            if c.weight <= 0:
                continue
            for name in (
                "Loss/value_loss_exploration",
                "Values_exploration/predicted_values",
                "Values_exploration/lambda_values",
                "Grads/critic_exploration",
            ):
                if name in metrics:
                    metrics[f"{name}_{k}"] = metrics[name]
            if c.reward_type == "intrinsic" and "Rewards/intrinsic" in metrics:
                metrics[f"Rewards/intrinsic_{k}"] = metrics["Rewards/intrinsic"]
        for name in (
            "Loss/value_loss_exploration",
            "Values_exploration/predicted_values",
            "Values_exploration/lambda_values",
            "Grads/critic_exploration",
            "Rewards/intrinsic",
        ):
            metrics.pop(name, None)

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[P2EDV3ExplorationState, EnvIndependentReplayBuffer]:
        cfg = self.cfg
        fabric = self.fabric
        self.is_continuous = isinstance(action_space, gym.spaces.Box)
        is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        self.actions_dim = tuple(
            action_space.shape
            if self.is_continuous
            else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
        )
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

        (
            world_model,
            ensembles,
            actor_task,
            critic_task,
            target_critic_task,
            actor_exploration,
            critics,
            self._policy,
        ) = build_agent(fabric, self.actions_dim, self.is_continuous, cfg, obs_space)

        def optimizer(optimizer_cfg: Dict[str, Any], module: nn.Module) -> Optimizer:
            return fabric.setup_optimizers(
                hydra.utils.instantiate(optimizer_cfg, params=module.parameters(), _convert_="all")
            )

        def moments() -> Moments:
            return Moments(
                cfg.algo.actor.moments.decay,
                cfg.algo.actor.moments.max,
                cfg.algo.actor.moments.percentile.low,
                cfg.algo.actor.moments.percentile.high,
            )

        critics_exploration = ExplorationCritics()
        for k, c in critics.items():
            critics_exploration[k] = {
                **c,
                "optimizer": optimizer(cfg.algo.critic.optimizer, c["module"]),
                "moments": moments(),
            }

        state = P2EDV3ExplorationState(
            world_model=world_model,
            ensembles=ensembles,
            actor_task=actor_task,
            critic_task=critic_task,
            target_critic_task=target_critic_task,
            actor_exploration=actor_exploration,
            world_optimizer=optimizer(cfg.algo.world_model.optimizer, world_model),
            actor_task_optimizer=optimizer(cfg.algo.actor.optimizer, actor_task),
            critic_task_optimizer=optimizer(cfg.algo.critic.optimizer, critic_task),
            ensemble_optimizer=optimizer(cfg.algo.ensembles.optimizer, ensembles),
            actor_exploration_optimizer=optimizer(cfg.algo.actor.optimizer, actor_exploration),
            moments_task=moments(),
            critics_exploration=critics_exploration,
        )
        # One buffer of sequences per environment, sampled independently
        buffer = EnvIndependentReplayBuffer(
            env_buffer_size(fabric, cfg, dry_run_size=4),
            n_envs=cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            buffer_cls=SequentialReplayBuffer,
            seed=cfg.seed + fabric.global_rank,
        )
        self.schedule = schedule
        return state, buffer

    def policy(self, state: P2EDV3ExplorationState) -> PlayerDV3:
        """The policy to play with: the exploration actor, which it shares its weights with (`build_agent`)."""
        return self._policy

    def task_policy(self, state: P2EDV3ExplorationState) -> PlayerDV3:
        """The policy of the task actor (zero-shot), to test it after the exploration."""
        policy = self.policy(state)
        policy.actor_type = "task"
        policy.actor = get_single_device_fabric(self.fabric).setup_module(unwrap_fabric(state.actor_task))
        return policy

    def test(self, state: TrainState, log_dir: str, policy_step: int = 0, test_name: str = "") -> None:
        # The task actor plays
        test(self.task_policy(state), self.fabric, self.cfg, log_dir, test_name, greedy=False, policy_step=policy_step)

    def player(self, state: P2EDV3ExplorationState) -> SequencePlayer:
        # Random actions until `algo.learning_starts`, except with the MineDojo actor
        random_warmup = "minedojo" not in self.cfg.algo.actor.cls.lower()
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
        self, state: P2EDV3ExplorationState, buffer: EnvIndependentReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        yield from sample_batches(self.fabric, self.cfg, buffer, n_steps)

    def train_step(self, state: P2EDV3ExplorationState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg
        critics_exploration = state.critics_exploration
        # The target critics follow the critics: every `critic.per_rank_target_network_update_freq` gradient steps,
        # an exponential moving average with `critic.tau`; at the first gradient step, a copy
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            tau = 1 if step == 0 else cfg.algo.critic.tau
            ema_(state.target_critic_task, state.critic_task, tau)
            for c in critics_exploration.values():
                ema_(c["target_module"], c["module"], tau)
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
            cfg,
            ensembles=state.ensembles,
            ensemble_optimizer=state.ensemble_optimizer,
            actor_exploration=state.actor_exploration,
            critics_exploration=critics_exploration,
            actor_exploration_optimizer=state.actor_exploration_optimizer,
            moments_exploration={k: c["moments"] for k, c in critics_exploration.items()},
            moments_task=state.moments_task,
            is_continuous=self.is_continuous,
            actions_dim=self.actions_dim,
        )
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = P2EDV3Exploration(fabric, cfg)
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
            "actor_task": state.actor_task,
            "critic_task": state.critic_task,
            "target_critic_task": state.target_critic_task,
            "moments_task": state.moments_task,
        }
        for k, c in state.critics_exploration.items():
            models_to_log["critic_exploration_" + k] = c["module"]
            models_to_log["target_critic_exploration_" + k] = c["target_module"]
        for k, c in state.critics_exploration.items():
            models_to_log["moments_exploration_" + k] = c["moments"]
        register_model(fabric, log_models, cfg, models_to_log)
