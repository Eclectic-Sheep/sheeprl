"""Plan2Explore (https://arxiv.org/abs/2005.05960) on Dreamer-V1: the finetuning phase.

Written on the shared training loop of `sheeprl.core`. It starts from the checkpoint of the exploration
(`checkpoint.exploration_ckpt_path`) and trains the task actor and critic as Dreamer-V1 does. The agent plays with
the exploration actor (`algo.player.actor_type`) until the training starts, then with the task actor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import Optimizer

from sheeprl.algos.dreamer_v1.agent import DreamerV1Policy, WorldModel
from sheeprl.algos.dreamer_v1.dreamer_v1 import SequenceWriter, sample_batches_of_iteration, train
from sheeprl.algos.dreamer_v2.dreamer_v2 import check_keys
from sheeprl.algos.dreamer_v2.utils import test
from sheeprl.algos.p2e_dv1.agent import build_agent
from sheeprl.algos.p2e_dv1.p2e_dv1_exploration import exploration_amounts
from sheeprl.core import Algorithm, TrainSchedule, TrainState, env_buffer_size, load_replay_buffer, run, sequence_store
from sheeprl.data.store import ReplayStore
from sheeprl.utils import fs
from sheeprl.utils.env import actions_dim_of
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import unwrap_fabric


@dataclass
class P2EDV1FinetuningState(TrainState):
    world_model: WorldModel
    actor_task: nn.Module
    critic_task: nn.Module
    # Plays until the training starts; not trained
    actor_exploration: nn.Module
    world_optimizer: Optimizer
    actor_task_optimizer: Optimizer
    critic_task_optimizer: Optimizer


class P2EDV1Finetuning(Algorithm):
    """Dreamer-V1 on the task, starting from the models (and optionally the replay buffer) of the exploration."""

    off_policy = True
    # No random actions: the policy trained by the exploration plays from the first step
    random_warmup = False

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
        cfg.env.screen_size = 64
        cfg.env.frame_stack = 1
        # The policy steps at the end of the iteration, set for its last gradient step (see `batches`)
        self.exploration_step: Optional[int] = None

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[P2EDV1FinetuningState, ReplayStore]:
        cfg = self.cfg
        fabric = self.fabric
        self.actions_dim, self.is_continuous = actions_dim_of(action_space)
        check_keys(fabric, cfg, obs_space)

        # The exploration critic and the ensembles are not used: they are built to initialize the models in the same
        # order as the exploration did. The policy plays with the actor of `algo.player.actor_type` until the training
        # starts (see `batches`)
        world_model, _, actor_task, critic_task, actor_exploration, _, self._policy = build_agent(
            fabric, self.actions_dim, self.is_continuous, cfg, obs_space
        )

        def optimizer(optimizer_cfg: Dict[str, Any], module: nn.Module) -> Optimizer:
            return fabric.setup_optimizers(
                hydra.utils.instantiate(optimizer_cfg, params=module.parameters(), _convert_="all")
            )

        state = P2EDV1FinetuningState(
            world_model=world_model,
            actor_task=actor_task,
            critic_task=critic_task,
            actor_exploration=actor_exploration,
            world_optimizer=optimizer(cfg.algo.world_model.optimizer, world_model),
            actor_task_optimizer=optimizer(cfg.algo.actor.optimizer, actor_task),
            critic_task_optimizer=optimizer(cfg.algo.critic.optimizer, critic_task),
        )
        buffer = sequence_store(
            fabric, cfg, log_dir, env_buffer_size(fabric, cfg, dry_run_size=4), cfg.algo.per_rank_sequence_length
        )

        # A new finetuning starts from the exploration (a resumed one from its own checkpoint, restored by the loop):
        # its models and optimizers
        if self.exploration_cfg is not None and not cfg.checkpoint.resume_from:
            exploration = fs.load_checkpoint(fabric, cfg.checkpoint.exploration_ckpt_path, weights_only=False)
            state.load_state_dict(exploration)
            if cfg.buffer.load_from_exploration and self.exploration_cfg.buffer.checkpoint:
                buffer = load_replay_buffer(fabric, exploration["rb"], buffer)
        # The exploration noise of the policy decays with the policy steps
        self._policy.schedule = schedule
        self.schedule = schedule
        return state, buffer

    def policy(self, state: P2EDV1FinetuningState) -> DreamerV1Policy:
        """The policy to play with: it shares its weights with the trained agent (`build_agent`)."""
        return self._policy

    def task_policy(self, state: P2EDV1FinetuningState) -> DreamerV1Policy:
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
        test(self.task_policy(state), self.fabric, self.cfg, log_dir, test_name, policy_step=policy_step)

    def writer(self, state: P2EDV1FinetuningState, policy: DreamerV1Policy) -> SequenceWriter:
        return SequenceWriter(self.cfg, self.actions_dim)

    def batches(
        self, state: P2EDV1FinetuningState, buffer: ReplayStore, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        # From the first training on, the task actor plays
        self.task_policy(state)
        yield from sample_batches_of_iteration(self, buffer, n_steps, iteration)

    def train_step(self, state: P2EDV1FinetuningState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
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
        )
        if self.exploration_step is not None:
            metrics.update(exploration_amounts(state, self.exploration_step))
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any], exploration_cfg: Dict[str, Any]):
    algo = P2EDV1Finetuning(fabric, cfg, exploration_cfg)
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
        }
        register_model(fabric, log_models, cfg, models_to_log)
