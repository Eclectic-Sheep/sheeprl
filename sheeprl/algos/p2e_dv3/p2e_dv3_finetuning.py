"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V3: the finetuning phase.

Written on the shared training loop of `sheeprl.core`. It starts from the checkpoint of the exploration
(`checkpoint.exploration_ckpt_path`) and trains the task actor and critic as Dreamer-V3 does. The agent plays with
the exploration actor (`algo.player.actor_type`) until the training starts, then with the task actor.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.dreamer_v2 import env_buffer_size, sample_batches
from sheeprl.algos.dreamer_v3.agent import PlayerDV3
from sheeprl.algos.dreamer_v3.dreamer_v3 import SequencePlayer, behaviour_learning, world_model_learning
from sheeprl.algos.dreamer_v3.utils import Moments, test
from sheeprl.algos.p2e_dv3.agent import build_models
from sheeprl.core import Algorithm, TrainSchedule, TrainState, load_replay_buffer, run, setup_module
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.registry import register_algorithm


@dataclass
class P2EDV3FinetuningState(TrainState):
    world_model: WorldModel
    actor_task: nn.Module
    critic_task: nn.Module
    target_critic_task: nn.Module
    # Plays until the training starts; not trained
    actor_exploration: nn.Module
    world_optimizer: Optimizer
    actor_task_optimizer: Optimizer
    critic_task_optimizer: Optimizer
    moments_task: Moments


class P2EDV3Finetuning(Algorithm):
    """Dreamer-V3 on the task, starting from the models (and optionally the replay buffer) of the exploration."""

    off_policy = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any], exploration_cfg: Optional[Dict[str, Any]] = None) -> None:
        """`exploration_cfg` is the configuration of the exploration to start from. Without it the algorithm only
        rebuilds the models of a finetuning from its own configuration, which already holds the values of the
        exploration (e.g. to evaluate a finetuning checkpoint)."""
        super().__init__(fabric, cfg)
        self.exploration_cfg = exploration_cfg
        if exploration_cfg is not None:
            # All the models must be equal to the ones of the exploration phase
            cfg.algo.gamma = exploration_cfg.algo.gamma
            cfg.algo.lmbda = exploration_cfg.algo.lmbda
            cfg.algo.horizon = exploration_cfg.algo.horizon
            cfg.algo.layer_norm = exploration_cfg.algo.layer_norm
            cfg.algo.dense_units = exploration_cfg.algo.dense_units
            cfg.algo.mlp_layers = exploration_cfg.algo.mlp_layers
            cfg.algo.dense_act = exploration_cfg.algo.dense_act
            cfg.algo.cnn_act = exploration_cfg.algo.cnn_act
            cfg.algo.unimix = exploration_cfg.algo.unimix
            cfg.algo.hafner_initialization = exploration_cfg.algo.hafner_initialization
            cfg.algo.world_model = exploration_cfg.algo.world_model
            cfg.algo.actor = exploration_cfg.algo.actor
            cfg.algo.critic = exploration_cfg.algo.critic
            # Rewards must be clipped in the same way as during exploration
            cfg.env.clip_rewards = exploration_cfg.env.clip_rewards
            # And the actions normalized in the same way (an exploration saved before the option didn't normalize them)
            cfg.algo.normalize_actions = exploration_cfg.algo.get("normalize_actions", False)
            # If the buffer is the same of the exploration, then we have to mantain the same number
            # of environments:
            #   - With less environments, you will replay too old experiences after a certain number of steps.
            #   - With more environments, you will raise an exception when you add new experienves.
            if cfg.buffer.load_from_exploration and exploration_cfg.buffer.checkpoint:
                cfg.env.num_envs = exploration_cfg.env.num_envs
            # There must be the same cnn and mlp keys during exploration and finetuning
            cfg.algo.cnn_keys = exploration_cfg.algo.cnn_keys
            cfg.algo.mlp_keys = exploration_cfg.algo.mlp_keys

        # These arguments cannot be changed
        cfg.env.frame_stack = 1

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[P2EDV3FinetuningState, EnvIndependentReplayBuffer]:
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

        # The exploration critics and the ensembles are not used: they are built to initialize the models in the same
        # order as the exploration did
        world_model, actor_task, critic_task, actor_exploration, _, _ = build_models(
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

        def optimizer(optimizer_cfg: Dict[str, Any], module: nn.Module) -> Optimizer:
            return fabric.setup_optimizers(
                hydra.utils.instantiate(optimizer_cfg, params=module.parameters(), _convert_="all")
            )

        state = P2EDV3FinetuningState(
            world_model=world_model,
            actor_task=actor_task,
            critic_task=critic_task,
            target_critic_task=target_critic_task,
            actor_exploration=actor_exploration,
            world_optimizer=optimizer(cfg.algo.world_model.optimizer, world_model),
            actor_task_optimizer=optimizer(cfg.algo.actor.optimizer, actor_task),
            critic_task_optimizer=optimizer(cfg.algo.critic.optimizer, critic_task),
            moments_task=Moments(
                cfg.algo.actor.moments.decay,
                cfg.algo.actor.moments.max,
                cfg.algo.actor.moments.percentile.low,
                cfg.algo.actor.moments.percentile.high,
            ),
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

        # A new finetuning starts from the exploration (a resumed one from its own checkpoint, restored by the loop)
        if self.exploration_cfg is not None and not cfg.checkpoint.resume_from:
            exploration = fabric.load(cfg.checkpoint.exploration_ckpt_path, weights_only=False)
            state.load_state_dict(exploration)
            if cfg.buffer.load_from_exploration and self.exploration_cfg.buffer.checkpoint:
                buffer = load_replay_buffer(fabric, exploration["rb"], buffer)
        self.schedule = schedule
        return state, buffer

    def policy(self, state: P2EDV3FinetuningState, actor: nn.Module, actor_type: str | None) -> PlayerDV3:
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

    def player(self, state: P2EDV3FinetuningState) -> SequencePlayer:
        # The actor of `algo.player.actor_type` plays until the training starts (see `batches`); no random actions
        actor = state.actor_exploration if self.cfg.algo.player.actor_type == "exploration" else state.actor_task
        self.acting_policy = self.policy(state, actor, None)
        return SequencePlayer(
            self.fabric,
            self.cfg,
            self.acting_policy,
            self.schedule,
            self.actions_dim,
            self.is_continuous,
            random_warmup=False,
        )

    def batches(
        self, state: P2EDV3FinetuningState, buffer: EnvIndependentReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        # From the first training on, the task actor plays
        if self.acting_policy.actor_type != "task":
            self.acting_policy.actor_type = "task"
            self.acting_policy.actor = state.actor_task
        yield from sample_batches(self.fabric, cfg, buffer, n_steps)

    def train_step(self, state: P2EDV3FinetuningState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg
        # The target critic follows the critic: every `critic.per_rank_target_network_update_freq` gradient steps, an
        # exponential moving average with `critic.tau`. Unlike Dreamer-V3, the first gradient step doesn't copy the
        # critic: the target critic is the one of the exploration, which has followed the critic since then
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            tau = cfg.algo.critic.tau
            for cp, tcp in zip(state.critic_task.module.parameters(), state.target_critic_task.parameters()):
                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)

        posteriors, recurrent_states, metrics = world_model_learning(
            self.fabric, cfg, state.world_model, state.world_optimizer, batch
        )
        task = behaviour_learning(
            self.fabric,
            cfg,
            state.world_model,
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
        metrics["Loss/policy_loss"] = task["policy_loss"]
        metrics["Loss/value_loss"] = task["value_loss"]
        if task["actor_grads"] is not None:
            metrics["Grads/actor"] = task["actor_grads"]
        if task["critic_grads"] is not None:
            metrics["Grads/critic"] = task["critic_grads"]
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any], exploration_cfg: Dict[str, Any]):
    algo = P2EDV3Finetuning(fabric, cfg, exploration_cfg)
    state, log_dir = run(fabric, cfg, algo)

    # task test few-shot
    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state, state.actor_task, "task"), fabric, cfg, log_dir, "few-shot", greedy=False)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": state.world_model,
            "actor_task": state.actor_task,
            "critic_task": state.critic_task,
            "target_critic_task": state.target_critic_task,
            "moments_task": state.moments_task,
        }
        register_model(fabric, log_models, cfg, models_to_log)
