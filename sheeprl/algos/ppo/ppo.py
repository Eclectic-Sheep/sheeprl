"""Proximal Policy Optimization (https://arxiv.org/abs/1707.06347), written on the shared training loop of
`sheeprl.core`: `PPO` says how to build, play and train; `sheeprl.core.loop.run` does the rest."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler

from sheeprl.algos.ppo.agent import PPOAgent, PPOPlayer
from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.utils import normalize_obs, prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, Rollout, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay


@dataclass
class PPOState(TrainState):
    # Feature extractor, actor and critic
    agent: PPOAgent
    optimizer: Optimizer
    # Anneals the learning rate (`algo.anneal_lr`)
    scheduler: Optional[PolynomialLR]
    # Annealed values are tensors, so that changing them never recompiles a compiled training step
    clip_coef: Tensor
    ent_coef: Tensor


def policy(state: PPOState) -> PPOPlayer:
    """The policy to play with: it shares its modules (and so its weights) with the trained agent."""
    return PPOPlayer(state.agent.feature_extractor, state.agent.actor, state.agent.critic)


class RolloutPlayer:
    """Plays the policy in the environments and writes every step in the rollout."""

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any], policy: PPOPlayer) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.policy = policy
        self.cnn_keys = cfg.algo.cnn_keys.encoder
        self.obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    def step(self, env: EnvRunner, rollout: Rollout) -> None:
        cfg = self.cfg
        num_envs = env.num_envs

        # Sample the actions: one-hot for discrete actions, while the environments take their indices
        obs = {k: env.obs[k] for k in self.obs_keys}
        obs = prepare_obs(self.fabric, obs, cnn_keys=self.cnn_keys, num_envs=num_envs)
        actions, logprobs, values = self.policy(obs)
        if self.policy.actor.is_continuous:
            env_actions = torch.stack(actions, dim=-1).cpu().numpy()
        else:
            env_actions = torch.stack([act.argmax(dim=-1) for act in actions], dim=-1).cpu().numpy()
        actions = torch.cat(actions, dim=-1).cpu().numpy()

        step = env.step(env_actions)

        # The episodes truncated by the time limit don't end in the MDP: bootstrap the value of their final observation
        rewards = step.rewards
        truncated_envs = np.nonzero(step.truncated)[0]
        if len(truncated_envs) > 0:
            final_obs = prepare_obs(
                self.fabric,
                step.final_obs(truncated_envs, self.obs_keys),
                cnn_keys=self.cnn_keys,
                num_envs=len(truncated_envs),
            )
            final_values = self.policy.get_values(final_obs).cpu().numpy()
            rewards[truncated_envs] += cfg.algo.gamma * final_values.reshape(rewards[truncated_envs].shape)
        dones = np.logical_or(step.terminated, step.truncated).reshape(num_envs, -1).astype(np.uint8)
        if cfg.env.clip_rewards:
            rewards = np.tanh(rewards)
        rewards = rewards.reshape(num_envs, -1).astype(np.float32)

        # The stacked frames of an image are stored as its channels
        data = {}
        for k in self.obs_keys:
            data[k] = step.obs[k]
            if k in self.cnn_keys:
                data[k] = data[k].reshape(num_envs, -1, *data[k].shape[-2:])
            data[k] = data[k][np.newaxis]
        data["dones"] = dones[np.newaxis]
        data["values"] = values.cpu().numpy()[np.newaxis]
        data["actions"] = actions[np.newaxis]
        data["logprobs"] = logprobs.cpu().numpy()[np.newaxis]
        data["rewards"] = rewards[np.newaxis]
        if cfg.buffer.memmap:
            data["returns"] = np.zeros_like(rewards, shape=(1, *rewards.shape))
            data["advantages"] = np.zeros_like(rewards, shape=(1, *rewards.shape))
        rollout.add(data, step.next_obs, validate_args=cfg.buffer.validate_args)


class PPO(Algorithm):
    """Every iteration plays `algo.rollout_steps` steps, then trains for `algo.update_epochs` epochs of minibatches
    of the rollout with the clipped surrogate objective."""

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        if "minedojo" in cfg.env.wrapper._target_.lower():
            raise ValueError(
                "MineDojo is not currently supported by PPO agent, since it does not take "
                "into consideration the action masks provided by the environment, but needed "
                "in order to play correctly the game. "
                "As an alternative you can use one of the Dreamers' agents."
            )
        if cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder == []:
            raise RuntimeError(
                "You should specify at least one CNN keys or MLP keys from the cli: "
                "`cnn_keys.encoder=[rgb]` or `mlp_keys.encoder=[state]`"
            )
        if cfg.buffer.size < cfg.algo.rollout_steps:
            raise ValueError(
                f"The size of the buffer ({cfg.buffer.size}) cannot be lower "
                f"than the rollout steps ({cfg.algo.rollout_steps})"
            )
        self.steps_per_iteration = cfg.algo.rollout_steps

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[PPOState, Rollout]:
        cfg = self.cfg
        if not isinstance(obs_space, gym.spaces.Dict):
            raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {obs_space}")
        if cfg.metric.log_level > 0:
            self.fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
            self.fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
        is_continuous = isinstance(action_space, gym.spaces.Box)
        is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        actions_dim = tuple(
            action_space.shape
            if is_continuous
            else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
        )

        agent = PPOAgent(
            actions_dim=actions_dim,
            obs_space=obs_space,
            encoder_cfg=cfg.algo.encoder,
            actor_cfg=cfg.algo.actor,
            critic_cfg=cfg.algo.critic,
            cnn_keys=cfg.algo.cnn_keys.encoder,
            mlp_keys=cfg.algo.mlp_keys.encoder,
            screen_size=cfg.env.screen_size,
            distribution_cfg=cfg.distribution,
            is_continuous=is_continuous,
        )
        agent.feature_extractor = setup_module(self.fabric, agent.feature_extractor)
        agent.actor = setup_module(self.fabric, agent.actor)
        agent.critic = setup_module(self.fabric, agent.critic)

        optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")
        optimizer = self.fabric.setup_optimizers(optimizer)
        # Linear decay of the learning rate to 0 at the end of the training
        scheduler = PolynomialLR(optimizer, total_iters=schedule.total_iters, power=1.0) if cfg.algo.anneal_lr else None
        self.total_iters = schedule.total_iters

        state = PPOState(
            agent=agent,
            optimizer=optimizer,
            scheduler=scheduler,
            clip_coef=torch.tensor(cfg.algo.clip_coef, device=self.fabric.device),
            ent_coef=torch.tensor(cfg.algo.ent_coef, device=self.fabric.device),
        )
        buffer = ReplayBuffer(
            cfg.buffer.size,
            cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{self.fabric.global_rank}"),
            obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
        )
        return state, Rollout(buffer)

    def player(self, state: PPOState) -> RolloutPlayer:
        return RolloutPlayer(self.fabric, self.cfg, policy(state))

    def batches(self, state: PPOState, rollout: Rollout, n_steps: Optional[int]) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        data = rollout.buffer.to_tensor(dtype=None, device=self.fabric.device, from_numpy=cfg.buffer.from_numpy)

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.inference_mode():
            next_obs = {k: rollout.next_obs[k] for k in cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder}
            next_obs = prepare_obs(self.fabric, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
            next_values = state.agent.critic(state.agent.feature_extractor(next_obs))
            returns, advantages = gae(
                data["rewards"],
                data["values"],
                data["dones"],
                next_values,
                cfg.algo.rollout_steps,
                cfg.algo.gamma,
                cfg.algo.gae_lambda,
            )
            data["returns"] = returns.float()
            data["advantages"] = advantages.float()

        if cfg.buffer.share_data and self.fabric.world_size > 1:
            # Train on the rollouts of all the processes: flatten [World_Size, Rollout_Steps, Num_Envs]
            data = self.fabric.all_gather(data)
            data = {k: v.flatten(start_dim=0, end_dim=2).float() for k, v in data.items()}
        else:
            # Flatten [Rollout_Steps, Num_Envs]
            data = {k: v.flatten(start_dim=0, end_dim=1).float() for k, v in data.items()}

        indexes = list(range(next(iter(data.values())).shape[0]))
        if cfg.buffer.share_data:
            sampler = DistributedSampler(
                indexes,
                num_replicas=self.fabric.world_size,
                rank=self.fabric.global_rank,
                shuffle=True,
                seed=cfg.seed,
            )
        else:
            sampler = RandomSampler(indexes)
        sampler = BatchSampler(sampler, batch_size=cfg.algo.per_rank_batch_size, drop_last=False)
        for epoch in range(cfg.algo.update_epochs):
            if cfg.buffer.share_data:
                sampler.sampler.set_epoch(epoch)
            for batch_idxes in sampler:
                yield {k: v[batch_idxes] for k, v in data.items()}

    def train_step(self, state: PPOState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        obs = normalize_obs(batch, cfg.cnn_keys.encoder, cfg.mlp_keys.encoder + cfg.cnn_keys.encoder)
        with autocast(self.fabric):
            _, logprobs, entropy, values = state.agent(
                obs, torch.split(batch["actions"], state.agent.actions_dim, dim=-1)
            )
            advantages = batch["advantages"]
            if cfg.normalize_advantages:
                advantages = normalize_tensor(advantages)
            pg_loss = policy_loss(logprobs, batch["logprobs"], advantages, state.clip_coef, cfg.loss_reduction)
            v_loss = value_loss(
                values, batch["values"], batch["returns"], state.clip_coef, cfg.clip_vloss, cfg.loss_reduction
            )
            ent_loss = entropy_loss(entropy, cfg.loss_reduction)
            # Equation (9) in the paper
            loss = pg_loss + cfg.vf_coef * v_loss + state.ent_coef * ent_loss
        update(self.fabric, loss, state.optimizer, max_grad_norm=cfg.max_grad_norm)
        return {
            "Loss/policy_loss": pg_loss.detach(),
            "Loss/value_loss": v_loss.detach(),
            "Loss/entropy_loss": ent_loss.detach(),
        }

    def end_iteration(self, state: PPOState, iteration: int) -> Dict[str, float]:
        cfg = self.cfg.algo
        # The values used in this iteration
        info = {
            "Info/learning_rate": state.optimizer.param_groups[0]["lr"],
            "Info/clip_coef": state.clip_coef.item(),
            "Info/ent_coef": state.ent_coef.item(),
        }
        # The values for the next iteration: linear decay to 0 at the end of the training
        if state.scheduler is not None:
            state.scheduler.step()
        if cfg.anneal_clip_coef:
            state.clip_coef.fill_(
                polynomial_decay(iteration, initial=cfg.clip_coef, final=0.0, max_decay_steps=self.total_iters)
            )
        if cfg.anneal_ent_coef:
            state.ent_coef.fill_(
                polynomial_decay(iteration, initial=cfg.ent_coef, final=0.0, max_decay_steps=self.total_iters)
            )
        return info


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    state, log_dir = run(fabric, cfg, PPO(fabric, cfg))

    if fabric.is_global_zero and cfg.algo.run_test:
        test(policy(state), fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.ppo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
