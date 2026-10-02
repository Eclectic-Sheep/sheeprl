"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V2: the exploration phase.

Written on the shared training loop of `sheeprl.core`. The agent plays with an exploration actor, rewarded by the
disagreement of an ensemble of models of the dynamics (the novelty of the states); a task actor learns the task from
the same experience (zero-shot). The player and the world-model and behaviour phases of a gradient step are the ones
of Dreamer-V2 (`sheeprl.algos.dreamer_v2.dreamer_v2`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

import gymnasium as gym
import hydra
import torch
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.distributions import Independent, Normal
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import PlayerDV2, WorldModel
from sheeprl.algos.dreamer_v2.dreamer_v2 import (
    SequencePlayer,
    actions_dim_of,
    behaviour_learning,
    build_buffer,
    check_keys,
    sample_batches,
    setup_world_model,
    world_model_learning,
)
from sheeprl.algos.dreamer_v2.utils import test
from sheeprl.algos.p2e_dv2.agent import build_models
from sheeprl.core import Algorithm, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer
from sheeprl.utils.registry import register_algorithm

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
    ) -> Tuple[P2EDV2ExplorationState, EnvIndependentReplayBuffer | EpisodeBuffer]:
        cfg = self.cfg
        fabric = self.fabric
        self.actions_dim, self.is_continuous = actions_dim_of(action_space)
        check_keys(fabric, cfg, obs_space)

        world_model, actor_exploration, critic_exploration, actor_task, critic_task, ensembles = build_models(
            fabric, self.actions_dim, self.is_continuous, cfg, obs_space
        )
        world_model = setup_world_model(fabric, world_model)
        actor_exploration = setup_module(fabric, actor_exploration)
        critic_exploration = setup_module(fabric, critic_exploration)
        target_critic_exploration = setup_module(fabric, copy.deepcopy(critic_exploration.module))
        actor_task = setup_module(fabric, actor_task)
        critic_task = setup_module(fabric, critic_task)
        target_critic_task = setup_module(fabric, copy.deepcopy(critic_task.module))
        for i in range(len(ensembles)):
            ensembles[i] = setup_module(fabric, ensembles[i])

        def optimizer(optimizer_cfg: Dict[str, Any], module: nn.Module) -> Optimizer:
            return fabric.setup_optimizers(
                hydra.utils.instantiate(optimizer_cfg, params=module.parameters(), _convert_="all")
            )

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
        self.schedule = schedule
        return state, build_buffer(fabric, cfg, log_dir, dry_run_size=4)

    def policy(self, state: P2EDV2ExplorationState, actor: nn.Module, actor_type: str) -> PlayerDV2:
        """The policy to play with `actor`: it shares its modules (and so its weights) with the trained agent."""
        cfg = self.cfg
        return PlayerDV2(
            state.world_model.encoder,
            state.world_model.rssm.recurrent_model,
            state.world_model.rssm.representation_model,
            actor,
            self.actions_dim,
            cfg.env.num_envs,
            cfg.algo.world_model.stochastic_size,
            cfg.algo.world_model.recurrent_model.recurrent_state_size,
            self.fabric.device,
            discrete_size=cfg.algo.world_model.discrete_size,
            actor_type=actor_type,
        )

    def player(self, state: P2EDV2ExplorationState) -> SequencePlayer:
        # Random actions until `algo.learning_starts`, except with MineDojo (its action masks)
        random_warmup = "minedojo" not in self.cfg.env.wrapper._target_.lower()
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
        self,
        state: P2EDV2ExplorationState,
        buffer: EnvIndependentReplayBuffer | EpisodeBuffer,
        n_steps: int,
        iteration: int,
    ) -> Iterator[Dict[str, Tensor]]:
        yield from sample_batches(self.fabric, self.cfg, buffer, n_steps)

    def train_step(self, state: P2EDV2ExplorationState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        """One gradient step, as in Algorithm 1 of the paper: the world model, the ensembles, the exploration actor and
        critic (behaviour learning with the intrinsic rewards), the task actor and critic (zero-shot)."""
        cfg = self.cfg
        fabric = self.fabric
        sequence_length = cfg.algo.per_rank_sequence_length
        batch_size = cfg.algo.per_rank_batch_size

        # The target critics are copies of the critics, every `critic.per_rank_target_network_update_freq` gradient
        # steps
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            for cp, tcp in zip(state.critic_task.module.parameters(), state.target_critic_task.parameters()):
                tcp.data.copy_(cp.data)
            for cp, tcp in zip(
                state.critic_exploration.module.parameters(), state.target_critic_exploration.parameters()
            ):
                tcp.data.copy_(cp.data)

        # Dynamic learning: the reward and continue models learn without changing the latent states
        posteriors, recurrent_states, metrics = world_model_learning(
            fabric, cfg, state.world_model, state.world_optimizer, batch, detach_heads=True
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
                next_obs_embedding_dist = Independent(Normal(out, 1), 1)
                loss -= next_obs_embedding_dist.log_prob(
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

        def intrinsic_reward(imagined_trajectories: Tensor, imagined_actions: Tensor) -> Tensor:
            # The intrinsic reward is not detached from the imagined trajectories, as in the reference
            # implementation: with continuous actions the exploration actor is trained by backpropagating
            # the lambda-values, intrinsic rewards included, through the dynamics
            next_obs_embedding = torch.stack(
                [ens(torch.cat((imagined_trajectories, imagined_actions), -1)) for ens in state.ensembles], dim=0
            )
            # next_obs_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
            return next_obs_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier

        # Behaviour learning of the exploration
        exploration = behaviour_learning(
            fabric,
            cfg,
            state.world_model,
            state.actor_exploration,
            state.critic_exploration,
            state.target_critic_exploration,
            state.actor_exploration_optimizer,
            state.critic_exploration_optimizer,
            posteriors,
            recurrent_states,
            batch["terminated"],
            self.is_continuous,
            self.actions_dim,
            objective_mix=cfg.algo.actor.objective_mix,
            reward_fn=intrinsic_reward,
        )
        metrics["Rewards/intrinsic"] = exploration["rewards"].mean()
        metrics["Values_exploration/predicted_values"] = exploration["values"].mean()
        metrics["Values_exploration/lambda_values"] = exploration["lambda_values"].mean()
        metrics["Loss/policy_loss_exploration"] = exploration["policy_loss"]
        metrics["Loss/value_loss_exploration"] = exploration["value_loss"]
        if exploration["actor_grads"] is not None:
            metrics["Grads/actor_exploration"] = exploration["actor_grads"]
        if exploration["critic_grads"] is not None:
            metrics["Grads/critic_exploration"] = exploration["critic_grads"]

        # Behaviour learning of the task (zero-shot)
        task = behaviour_learning(
            fabric,
            cfg,
            state.world_model,
            state.actor_task,
            state.critic_task,
            state.target_critic_task,
            state.actor_task_optimizer,
            state.critic_task_optimizer,
            posteriors,
            recurrent_states,
            batch["terminated"],
            self.is_continuous,
            self.actions_dim,
            objective_mix=cfg.algo.actor.objective_mix,
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
    algo = P2EDV2Exploration(fabric, cfg)
    state, log_dir = run(fabric, cfg, algo)

    # task test zero-shot
    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state, state.actor_task, "task"), fabric, cfg, log_dir, "zero-shot")

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
