"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V3: the exploration phase.

Written on the shared training loop of `sheeprl.core`. The agent plays with an exploration actor, rewarded by the
disagreement of an ensemble of models of the dynamics (the novelty of the states) and optionally by the task rewards;
a task actor learns the task from the same experience (zero-shot). The player and the world-model and task phases of a
gradient step are the ones of Dreamer-V3 (`sheeprl.algos.dreamer_v3.dreamer_v3`).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence, Tuple

import gymnasium as gym
import hydra
import torch
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.distributions import Distribution, Independent
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v3.agent import PlayerDV3
from sheeprl.algos.dreamer_v3.dreamer_v3 import SequencePlayer, behaviour_learning, world_model_learning
from sheeprl.algos.dreamer_v3.utils import Moments, compute_lambda_values, test
from sheeprl.algos.p2e_dv3.agent import build_models
from sheeprl.core import Algorithm, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.core.algorithm import load_module_state_dict
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.distribution import BernoulliSafeMode, MSEDistribution, TwoHotEncodingDistribution
from sheeprl.utils.registry import register_algorithm

# Decomment the following line if you are using MineDojo on an headless machine
# os.environ["MINEDOJO_HEADLESS"] = "1"


@dataclass
class ExplorationCritic:
    """A critic of the exploration actor, with its own rewards: intrinsic (the disagreement of the ensembles) or of
    the task. The advantages of the critics are summed with their weights."""

    module: nn.Module
    target_module: nn.Module
    optimizer: Optimizer
    moments: Moments
    weight: float
    reward_type: str


class ExplorationCritics(Dict[str, ExplorationCritic]):
    """The exploration critics by name. Their weights are saved as in the checkpoints of the old training loop:
    `{name: {"module": ..., "target_module": ...}}`."""

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            k: {"module": c.module.state_dict(), "target_module": c.target_module.state_dict()} for k, c in self.items()
        }

    def load_state_dict(self, state: Dict[str, Dict[str, Any]]) -> None:
        for k, c in self.items():
            load_module_state_dict(c.module, state[k]["module"])
            load_module_state_dict(c.target_module, state[k]["target_module"])


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
            state[f"critic_exploration_optimizer_{k}"] = c.optimizer.state_dict()
            state[f"moments_exploration_{k}"] = c.moments.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        for k, c in self.critics_exploration.items():
            c.optimizer.load_state_dict(state[f"critic_exploration_optimizer_{k}"])
            load_module_state_dict(c.moments, state[f"moments_exploration_{k}"])


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

        world_model, actor_task, critic_task, actor_exploration, critics, ensembles = build_models(
            fabric, self.actions_dim, self.is_continuous, cfg, obs_space
        )
        world_model.encoder = setup_module(fabric, world_model.encoder)
        world_model.observation_model = setup_module(fabric, world_model.observation_model)
        world_model.reward_model = setup_module(fabric, world_model.reward_model)
        world_model.rssm.recurrent_model = setup_module(fabric, world_model.rssm.recurrent_model)
        world_model.rssm.representation_model = setup_module(fabric, world_model.rssm.representation_model)
        world_model.rssm.transition_model = setup_module(fabric, world_model.rssm.transition_model)
        if world_model.continue_model:
            world_model.continue_model = setup_module(fabric, world_model.continue_model)
        actor_task = setup_module(fabric, actor_task)
        critic_task = setup_module(fabric, critic_task)
        target_critic_task = setup_module(fabric, copy.deepcopy(critic_task.module)).requires_grad_(False)
        actor_exploration = setup_module(fabric, actor_exploration)
        for i in range(len(ensembles)):
            ensembles[i] = setup_module(fabric, ensembles[i])

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
            module = setup_module(fabric, c["module"])
            critics_exploration[k] = ExplorationCritic(
                module=module,
                target_module=setup_module(fabric, copy.deepcopy(module.module)).requires_grad_(False),
                optimizer=optimizer(cfg.algo.critic.optimizer, module),
                moments=moments(),
                weight=c["weight"],
                reward_type=c["reward_type"],
            )

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
            # Known issue #13: the optimizer of the ensembles is configured by `algo.critic.optimizer`
            ensemble_optimizer=optimizer(cfg.algo.critic.optimizer, ensembles),
            actor_exploration_optimizer=optimizer(cfg.algo.actor.optimizer, actor_exploration),
            moments_task=moments(),
            critics_exploration=critics_exploration,
        )
        # One buffer of sequences per environment, sampled independently
        buffer = EnvIndependentReplayBuffer(
            cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 4,
            n_envs=cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            buffer_cls=SequentialReplayBuffer,
            seed=cfg.seed + fabric.global_rank,
        )
        self.schedule = schedule
        return state, buffer

    def policy(self, state: P2EDV3ExplorationState, actor: nn.Module, actor_type: str) -> PlayerDV3:
        """The policy to play with `actor`: it shares its modules (and so its weights) with the trained agent."""
        cfg = self.cfg
        return PlayerDV3(
            state.world_model.encoder,
            state.world_model.rssm,
            actor,
            self.actions_dim,
            cfg.env.num_envs,
            cfg.algo.world_model.stochastic_size,
            cfg.algo.world_model.recurrent_model.recurrent_state_size,
            self.fabric.device,
            discrete_size=cfg.algo.world_model.discrete_size,
            actor_type=actor_type,
        )

    def player(self, state: P2EDV3ExplorationState) -> SequencePlayer:
        # Random actions until `algo.learning_starts`, except when resuming and with the MineDojo actor
        random_warmup = self.cfg.checkpoint.resume_from is None and "minedojo" not in self.cfg.algo.actor.cls.lower()
        return SequencePlayer(
            self.fabric,
            self.cfg,
            self.policy(state, state.actor_exploration, "exploration"),
            self.schedule,
            self.actions_dim,
            self.is_continuous,
            random_warmup,
        )

    def batches(
        self, state: P2EDV3ExplorationState, buffer: EnvIndependentReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        sample = buffer.sample_tensors(
            cfg.algo.per_rank_batch_size,
            sequence_length=cfg.algo.per_rank_sequence_length,
            n_samples=n_steps,
            dtype=None,
            device=self.fabric.device,
            from_numpy=cfg.buffer.from_numpy,
        )  # [N_Steps, Sequence_Length, Batch_Size, ...]
        for i in range(n_steps):
            yield {k: v[i].float() for k, v in sample.items()}

    def train_step(self, state: P2EDV3ExplorationState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        """One gradient step, as in Algorithm 1 of the paper: the world model, the ensembles, the exploration actor and
        critics (behaviour learning with the intrinsic rewards), the task actor and critic (zero-shot)."""
        cfg = self.cfg
        fabric = self.fabric
        world_model = state.world_model
        critics = state.critics_exploration
        sequence_length = cfg.algo.per_rank_sequence_length
        batch_size = cfg.algo.per_rank_batch_size
        stoch_state_size = cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
        recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size

        # The target critics follow the critics: every `critic.per_rank_target_network_update_freq` gradient steps,
        # an exponential moving average with `critic.tau`; at the first gradient step, a copy
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            tau = 1 if step == 0 else cfg.algo.critic.tau
            for cp, tcp in zip(state.critic_task.module.parameters(), state.target_critic_task.parameters()):
                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
            for c in critics.values():
                for cp, tcp in zip(c.module.module.parameters(), c.target_module.parameters()):
                    tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)

        # Dynamic learning: the reward and continue models learn without changing the latent states
        posteriors, recurrent_states, metrics = world_model_learning(
            fabric, cfg, world_model, state.world_optimizer, batch, detach_heads=True
        )

        # Ensemble learning: each ensemble predicts the next posterior from the latent state and the action
        with autocast(fabric):
            loss = 0.0
            for ens in state.ensembles:
                out = ens(
                    torch.cat(
                        (
                            posteriors.view(*posteriors.shape[:-2], -1).detach(),
                            recurrent_states.detach(),
                            batch["actions"].detach(),
                        ),
                        -1,
                    )
                )[:-1]
                next_state_embedding_dist = MSEDistribution(out, 1)
                loss -= next_state_embedding_dist.log_prob(
                    posteriors.view(sequence_length, batch_size, -1).detach()[1:]
                ).mean()
        ensemble_grads = update(
            fabric,
            loss,
            state.ensemble_optimizer,
            max_grad_norm=cfg.algo.ensembles.clip_gradients or 0.0,
            error_if_nonfinite=False,
        )
        metrics["Loss/ensemble_loss"] = loss.detach()
        if ensemble_grads is not None:
            metrics["Grads/ensemble"] = ensemble_grads.detach()

        # Behaviour learning of the exploration
        with autocast(fabric):
            imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
            recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            # the imagined states and actions are concatenated at the end of the imagination
            imagined_trajectories = [imagined_latent_state]
            actions = torch.cat(state.actor_exploration(imagined_latent_state.detach())[0], dim=-1)
            imagined_actions = [actions]

            # imagine trajectories in the latent space
            for i in range(1, cfg.algo.horizon + 1):
                imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
                imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
                imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
                imagined_trajectories.append(imagined_latent_state)
                actions = torch.cat(state.actor_exploration(imagined_latent_state.detach())[0], dim=-1)
                imagined_actions.append(actions)
            imagined_trajectories = torch.cat(imagined_trajectories, dim=0)
            imagined_actions = torch.cat(imagined_actions, dim=0)

            # The advantage of the actor is the weighted sum of the advantages of the critics
            advantages = []
            lambda_values = {}
            weights_sum = sum([c.weight for c in critics.values()])
            for k, critic in critics.items():
                # Predict values and continues
                predicted_values = TwoHotEncodingDistribution(critic.module(imagined_trajectories), dims=1).mean
                continues = Independent(
                    BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1
                ).mode
                true_continue = (1 - batch["terminated"]).flatten().reshape(1, -1, 1)
                continues = torch.cat((true_continue, continues[1:]))

                if critic.reward_type == "intrinsic":
                    # Predict intrinsic reward
                    # The intrinsic reward is not detached from the imagined trajectories, as in the reference
                    # implementation: with continuous actions the exploration actor is trained by backpropagating
                    # the lambda-values, intrinsic rewards included, through the dynamics
                    next_state_embedding = torch.stack(
                        [ens(torch.cat((imagined_trajectories, imagined_actions), -1)) for ens in state.ensembles],
                        dim=0,
                    )
                    # next_state_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
                    reward = next_state_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier
                    metrics[f"Rewards/intrinsic_{k}"] = reward.detach().mean()
                else:
                    reward = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean

                lambda_values[k] = compute_lambda_values(
                    reward[1:],
                    predicted_values[1:],
                    continues[1:] * cfg.algo.gamma,
                    lmbda=cfg.algo.lmbda,
                )
                baseline = predicted_values[:-1]
                offset, invscale = critic.moments(lambda_values[k], fabric)
                normed_lambda_values = (lambda_values[k] - offset) / invscale
                normed_baseline = (baseline - offset) / invscale
                advantages.append((normed_lambda_values - normed_baseline) * critic.weight / weights_sum)
                metrics[f"Values_exploration/predicted_values_{k}"] = predicted_values.detach().mean()
                metrics[f"Values_exploration/lambda_values_{k}"] = lambda_values[k].detach().mean()

            advantage = torch.stack(advantages, dim=0).sum(dim=0)
            with torch.no_grad():
                discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

            policies: Sequence[Distribution] = state.actor_exploration(imagined_trajectories.detach())[1]
            if self.is_continuous:
                objective = advantage
            else:
                objective = (
                    torch.stack(
                        [
                            p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                            for p, imgnd_act in zip(policies, torch.split(imagined_actions, self.actions_dim, dim=-1))
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
        actor_grads = update(
            fabric,
            policy_loss_exploration,
            state.actor_exploration_optimizer,
            max_grad_norm=cfg.algo.actor.clip_gradients or 0.0,
            error_if_nonfinite=False,
        )
        metrics["Loss/policy_loss_exploration"] = policy_loss_exploration.detach()
        if actor_grads is not None:
            metrics["Grads/actor_exploration"] = actor_grads.mean().detach()

        for k, critic in critics.items():
            with autocast(fabric):
                qv = TwoHotEncodingDistribution(critic.module(imagined_trajectories.detach()[:-1]), dims=1)
                with torch.no_grad():
                    predicted_target_values = TwoHotEncodingDistribution(
                        critic.target_module(imagined_trajectories.detach()[:-1]), dims=1
                    ).mean
                # Critic optimization. Eq. 10 in the paper
                value_loss = -qv.log_prob(lambda_values[k].detach())
                value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
                value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))
            critic_grads = update(
                fabric,
                value_loss,
                critic.optimizer,
                max_grad_norm=cfg.algo.critic.clip_gradients or 0.0,
                error_if_nonfinite=False,
            )
            if critic_grads is not None:
                metrics[f"Grads/critic_exploration_{k}"] = critic_grads.mean().detach()
            metrics[f"Loss/value_loss_exploration_{k}"] = value_loss.detach()

        # Behaviour learning of the task (zero-shot), as in Dreamer-V3
        task = behaviour_learning(
            fabric,
            cfg,
            world_model,
            state.actor_task,
            state.critic_task,
            state.target_critic_task,
            state.actor_task_optimizer,
            state.critic_task_optimizer,
            state.moments_task,
            posteriors,
            recurrent_states,
            batch["terminated"],
            self.is_continuous,
            self.actions_dim,
        )
        metrics["Loss/policy_loss_task"] = task["policy_loss"]
        metrics["Loss/value_loss_task"] = task["value_loss"]
        if task["actor_grads"] is not None:
            metrics["Grads/actor_task"] = task["actor_grads"]
        if task["critic_grads"] is not None:
            metrics["Grads/critic_task"] = task["critic_grads"]
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = P2EDV3Exploration(fabric, cfg)
    state, log_dir = run(fabric, cfg, algo)

    # task test zero-shot
    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state, state.actor_task, "task"), fabric, cfg, log_dir, "zero-shot", greedy=False)

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
            models_to_log["critic_exploration_" + k] = c.module
            models_to_log["target_critic_exploration_" + k] = c.target_module
        for k, c in state.critics_exploration.items():
            models_to_log["moments_exploration_" + k] = c.moments
        register_model(fabric, log_models, cfg, models_to_log)
