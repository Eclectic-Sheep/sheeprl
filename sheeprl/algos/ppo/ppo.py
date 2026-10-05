"""Proximal Policy Optimization (https://arxiv.org/abs/1707.06347), written on the shared training loop of
`sheeprl.core`: `PPO` says how to build, play and train; `sheeprl.core.loop.run` does the rest."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer

from sheeprl.algos.ppo.agent import PPOAgent, PPOPlayer, build_agent
from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.utils import anneal, bootstrap_truncated, normalize_obs, prepare_obs, test
from sheeprl.core import (
    Algorithm,
    EnvRunner,
    ReplayStore,
    TrainSchedule,
    TrainState,
    autocast,
    rollout_store,
    run,
    update,
)
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae_function, normalize_tensor


@dataclass
class PPOState(TrainState):
    # Feature extractor, actor and critic
    agent: PPOAgent
    optimizer: Optimizer


class RolloutPlayer:
    """Plays the policy in the environments and writes every step in the rollout."""

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any], policy: PPOPlayer) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.policy = policy
        self.cnn_keys = cfg.algo.cnn_keys.encoder
        self.obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    def step(self, env: EnvRunner, rollout: ReplayStore) -> None:
        cfg = self.cfg
        num_envs = env.num_envs

        # Sample the actions: one-hot for discrete actions, while the environments take their indices
        obs = {k: env.obs[k] for k in self.obs_keys}
        obs = prepare_obs(self.fabric, obs, cnn_keys=self.cnn_keys, num_envs=num_envs)
        actions, logprobs, values = self.policy(obs)
        env_actions = self.policy.env_actions(actions).cpu().numpy()
        actions = torch.cat(actions, dim=-1).cpu().numpy()

        step = env.step(env_actions)

        def final_values(env_idxes: np.ndarray) -> np.ndarray:
            final_obs = step.final_obs(env_idxes, self.obs_keys)
            final_obs = prepare_obs(self.fabric, final_obs, cnn_keys=self.cnn_keys, num_envs=len(env_idxes))
            return self.policy.get_values(final_obs).cpu().numpy()

        # The episodes truncated by the time limit (and not terminated in the same step) don't end in the MDP: the
        # value of their final observation is added to the reward, after the rewards are clipped
        rewards = np.tanh(step.rewards) if cfg.env.clip_rewards else step.rewards
        rewards = bootstrap_truncated(rewards, step.terminated, step.truncated, final_values, cfg.algo.gamma)
        dones = np.logical_or(step.terminated, step.truncated).reshape(num_envs, -1).astype(np.uint8)
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
        rollout.add(data, validate_args=cfg.buffer.validate_args)
        # The observations after the last step of the rollout bootstrap its returns
        rollout.context["next_obs"] = step.next_obs


def ppo_loss(
    agent: PPOAgent,
    obs: Dict[str, Tensor],
    actions: Tensor,
    logprobs: Tensor,
    values: Tensor,
    returns: Tensor,
    advantages: Tensor,
    clip_coef: Tensor,
    ent_coef: Tensor,
    *,
    actions_dim: Tuple[int, ...],
    normalize_advantages: bool,
    vf_coef: float,
    clip_vloss: bool,
    reduction: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The loss of PPO on a minibatch (Equation 9 in the paper) and its terms: the clipped surrogate objective, the
    value loss and the entropy of the policy. Can be compiled (`algo.compile`)."""
    _, new_logprobs, entropy, new_values = agent(obs, torch.split(actions, actions_dim, dim=-1))
    if normalize_advantages:
        advantages = normalize_tensor(advantages)
    pg_loss = policy_loss(new_logprobs, logprobs, advantages, clip_coef, reduction)
    v_loss = value_loss(new_values, values, returns, clip_coef, clip_vloss, reduction)
    ent_loss = entropy_loss(entropy, reduction)
    return pg_loss + vf_coef * v_loss + ent_coef * ent_loss, pg_loss, v_loss, ent_loss


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
        # The buffer holds one rollout: every row is trained on, and the returns are computed over all of them
        if cfg.buffer.size != cfg.algo.rollout_steps:
            raise ValueError(
                f"The size of the buffer ({cfg.buffer.size}) must be equal "
                f"to the rollout steps ({cfg.algo.rollout_steps})"
            )
        self.steps_per_iteration = cfg.algo.rollout_steps
        self.gae = gae_function(cfg.algo.gae_method)
        # The values the annealed coefficients start from
        self.initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)
        self.initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[PPOState, ReplayStore]:
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
        agent, self._policy = build_agent(self.fabric, actions_dim, is_continuous, cfg, obs_space)

        optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")
        optimizer = self.fabric.setup_optimizers(optimizer)
        self.total_iters = schedule.total_iters
        # The annealed coefficients of the iteration as tensors: a compiled step doesn't recompile when they change
        self.clip_coef = torch.tensor(float(cfg.algo.clip_coef), device=self.fabric.device)
        self.ent_coef = torch.tensor(float(cfg.algo.ent_coef), device=self.fabric.device)

        state = PPOState(agent=agent, optimizer=optimizer)
        return state, rollout_store(self.fabric, cfg, log_dir, cfg.buffer.size)

    def policy(self, state: PPOState) -> PPOPlayer:
        """The policy to play with: it shares its weights with the trained agent (`build_agent`)."""
        return self._policy

    def test(self, state: TrainState, log_dir: str, policy_step: int = 0) -> None:
        test(self.policy(state), self.fabric, self.cfg, log_dir, policy_step=policy_step)

    def player(self, state: PPOState) -> RolloutPlayer:
        return RolloutPlayer(self.fabric, self.cfg, self.policy(state))

    def batches(
        self, state: PPOState, rollout: ReplayStore, n_steps: Optional[int], iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        # The learning rate and the coefficients of the iteration
        anneal(cfg, state.optimizer, iteration, self.total_iters, self.initial_clip_coef, self.initial_ent_coef)
        self.clip_coef.fill_(cfg.algo.clip_coef)
        self.ent_coef.fill_(cfg.algo.ent_coef)
        data = rollout.read()

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.inference_mode():
            next_obs = {
                k: rollout.context["next_obs"][k] for k in cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
            }
            next_obs = prepare_obs(self.fabric, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
            next_values = self.policy(state).get_values(next_obs)
            returns, advantages = self.gae(
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

        yield from rollout.minibatches(data, cfg.algo.update_epochs)

    def train_step(self, state: PPOState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        # The loss is compiled when `algo.compile.enabled` is set
        mark_gradient_step(self.fabric, self.cfg)
        obs = normalize_obs(batch, cfg.cnn_keys.encoder, cfg.mlp_keys.encoder + cfg.cnn_keys.encoder)
        with autocast(self.fabric):
            loss, pg_loss, v_loss, ent_loss = compiled(ppo_loss, self.fabric, self.cfg)(
                state.agent,
                obs,
                batch["actions"],
                batch["logprobs"],
                batch["values"],
                batch["returns"],
                batch["advantages"],
                self.clip_coef,
                self.ent_coef,
                actions_dim=tuple(int(dim) for dim in state.agent.actions_dim),
                normalize_advantages=cfg.normalize_advantages,
                vf_coef=cfg.vf_coef,
                clip_vloss=cfg.clip_vloss,
                reduction=cfg.loss_reduction,
            )
        update(self.fabric, loss, state.optimizer, max_grad_norm=cfg.max_grad_norm)
        return {
            "Loss/policy_loss": pg_loss.detach(),
            "Loss/value_loss": v_loss.detach(),
            "Loss/entropy_loss": ent_loss.detach(),
        }


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = PPO(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        algo.test(state, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.ppo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
