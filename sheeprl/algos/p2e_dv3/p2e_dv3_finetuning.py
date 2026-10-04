"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V3: the finetuning phase.

Written on the shared training loop of `sheeprl.core`. It starts from the checkpoint of the exploration
(`checkpoint.exploration_ckpt_path`) and trains the task actor and critic as Dreamer-V3 does. The agent plays with
the exploration actor (`algo.player.actor_type`) until the training starts, then with the task actor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.utils import env_buffer_size, sample_batches
from sheeprl.algos.dreamer_v3.agent import PlayerDV3
from sheeprl.algos.dreamer_v3.dreamer_v3 import SequencePlayer, train
from sheeprl.algos.dreamer_v3.utils import Moments, test
from sheeprl.algos.p2e_dv3.agent import build_agent
from sheeprl.core import Algorithm, TrainSchedule, TrainState, load_replay_buffer, run
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.model import ema_
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import unwrap_fabric


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
        # The exploration critics and the ensembles are not used: they are built to initialize the models in the same
        # order as the exploration did. The policy plays with the actor of `algo.player.actor_type` until the training
        # starts (see `batches`)
        world_model, _, actor_task, critic_task, target_critic_task, actor_exploration, _, self._policy = build_agent(
            fabric, self.actions_dim, self.is_continuous, cfg, obs_space
        )

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

    def policy(self, state: P2EDV3FinetuningState) -> PlayerDV3:
        """The policy to play with: it shares its weights with the trained agent (`build_agent`)."""
        return self._policy

    def task_policy(self, state: P2EDV3FinetuningState) -> PlayerDV3:
        """The policy of the task actor, which plays from the first training on."""
        policy = self.policy(state)
        if policy.actor_type != "task":
            policy.actor_type = "task"
            policy.actor = get_single_device_fabric(self.fabric).setup_module(unwrap_fabric(state.actor_task))
            for agent_p, p in zip(state.actor_task.parameters(), policy.actor.parameters()):
                p.data = agent_p.data
        return policy

    def test(self, state: TrainState, log_dir: str, policy_step: int = 0, test_name: str = "") -> None:
        # The task actor plays
        test(self.task_policy(state), self.fabric, self.cfg, log_dir, test_name, greedy=False, policy_step=policy_step)

    def player(self, state: P2EDV3FinetuningState) -> SequencePlayer:
        # No random actions
        return SequencePlayer(
            self.fabric,
            self.cfg,
            self.policy(state),
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
        self.task_policy(state)
        yield from sample_batches(self.fabric, cfg, buffer, n_steps)

    def train_step(self, state: P2EDV3FinetuningState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg
        # The target critic follows the critic: every `critic.per_rank_target_network_update_freq` gradient steps, an
        # exponential moving average with `critic.tau`. Unlike Dreamer-V3, the first gradient step doesn't copy the
        # critic: the target critic is the one of the exploration, which has followed the critic since then
        if step % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            tau = cfg.algo.critic.tau
            ema_(state.target_critic_task, state.critic_task, tau)

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
            is_continuous=self.is_continuous,
            actions_dim=self.actions_dim,
            moments=state.moments_task,
        )
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any], exploration_cfg: Dict[str, Any]):
    algo = P2EDV3Finetuning(fabric, cfg, exploration_cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        algo.test(state, log_dir, policy_step=policy_step, test_name="few-shot")

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
