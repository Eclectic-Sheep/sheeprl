"""Advantage Actor-Critic (https://arxiv.org/abs/1602.01783), written on the shared training loop of `sheeprl.core`:
`A2C` says how to build, play and train; `sheeprl.core.loop.run` does the rest. The agent is the one of PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import PolynomialLR

from sheeprl.algos.a2c.loss import policy_loss
from sheeprl.algos.ppo.agent import PPOAgent, PPOPlayer, build_agent
from sheeprl.algos.ppo.loss import entropy_loss, value_loss
from sheeprl.algos.ppo.utils import bootstrap_truncated, normalize_obs, prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, Rollout, TrainSchedule, TrainState, all_reduce_gradients, autocast, run
from sheeprl.utils.compile import compiled
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae, normalize_tensor


@dataclass
class A2CState(TrainState):
    # Feature extractor, actor and critic
    agent: PPOAgent
    optimizer: Optimizer
    # Created with `algo.anneal_lr`, but never stepped: the learning rate is not annealed
    scheduler: Optional[PolynomialLR]


class RolloutPlayer:
    """Plays the policy in the environments and writes every step in the rollout. Unlike PPO's player, the rewards
    are stored in float64 and are not clipped (`env.clip_rewards` is ignored)."""

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
        env_actions = self.policy.env_actions(actions).cpu().numpy()
        actions = torch.cat(actions, dim=-1).cpu().numpy()

        step = env.step(env_actions)

        def final_values(env_idxes: np.ndarray) -> np.ndarray:
            final_obs = step.final_obs(env_idxes, self.obs_keys)
            final_obs = prepare_obs(self.fabric, final_obs, cnn_keys=self.cnn_keys, num_envs=len(env_idxes))
            return self.policy.get_values(final_obs).cpu().numpy()

        # The episodes truncated by the time limit (and not terminated in the same step) don't end in the MDP:
        # bootstrap the value of their final observation
        rewards = bootstrap_truncated(step.rewards, step.terminated, step.truncated, final_values, cfg.algo.gamma)
        dones = np.logical_or(step.terminated, step.truncated).reshape(num_envs, -1).astype(np.uint8)
        rewards = rewards.reshape(num_envs, -1)

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


def a2c_loss(
    agent: PPOAgent,
    obs: Dict[str, Tensor],
    actions: Tensor,
    values: Tensor,
    returns: Tensor,
    advantages: Tensor,
    *,
    actions_dim: Tuple[int, ...],
    normalize_advantages: bool,
    vf_coef: float,
    ent_coef: float,
    reduction: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """The loss of A2C on a minibatch and its policy and value terms. Can be compiled (`algo.compile`)."""
    _, logprobs, entropy, new_values = agent(obs, torch.split(actions, actions_dim, dim=-1))
    if normalize_advantages:
        advantages = normalize_tensor(advantages)
    pg_loss = policy_loss(logprobs, advantages, reduction)
    v_loss = value_loss(new_values, values, returns, 0.0, False, reduction)
    ent_loss = entropy_loss(entropy, reduction)
    return pg_loss + vf_coef * v_loss + ent_coef * ent_loss, pg_loss, v_loss


class A2C(Algorithm):
    """Every iteration plays `algo.rollout_steps` steps, then does one optimizer step on the whole rollout: the
    gradients of its minibatches are accumulated."""

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

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[A2CState, Rollout]:
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

        # The agent of PPO
        agent, self._policy = build_agent(self.fabric, actions_dim, is_continuous, cfg, obs_space)

        optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")
        optimizer = self.fabric.setup_optimizers(optimizer)
        scheduler = PolynomialLR(optimizer, total_iters=schedule.total_iters, power=1.0) if cfg.algo.anneal_lr else None

        state = A2CState(agent=agent, optimizer=optimizer, scheduler=scheduler)
        return state, Rollout.build(self.fabric, cfg, log_dir, cfg.buffer.size)

    def policy(self, state: A2CState) -> PPOPlayer:
        """The policy to play with: it shares its weights with the trained agent (`build_agent`)."""
        return self._policy

    def test(self, state: TrainState, log_dir: str, policy_step: int = 0) -> None:
        test(self.policy(state), self.fabric, self.cfg, log_dir, policy_step=policy_step)

    def player(self, state: A2CState) -> RolloutPlayer:
        return RolloutPlayer(self.fabric, self.cfg, self.policy(state))

    def batches(
        self, state: A2CState, rollout: Rollout, n_steps: Optional[int], iteration: int
    ) -> Iterator[List[Dict[str, Tensor]]]:
        """One batch per iteration: the minibatches of the whole rollout, whose gradients `train_step` accumulates."""
        cfg = self.cfg
        data = rollout.read()

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.inference_mode():
            next_obs = prepare_obs(
                self.fabric, rollout.next_obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs
            )
            next_values = self.policy(state).get_values(next_obs)
            returns, advantages = gae(
                data["rewards"].to(torch.float64),
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

        # One batch: the minibatches of the rollout, whose gradients `train_step` accumulates
        yield list(rollout.minibatches(data, 1))

    def train_step(self, state: A2CState, minibatches: List[Dict[str, Tensor]], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        params = [p for group in state.optimizer.param_groups for p in group["params"]]
        # Even though in the Spinning-Up A2C algorithm implementation
        # (https://spinningup.openai.com/en/latest/algorithms/vpg.html) the policy gradient is estimated
        # by taking the mean over all the sequences collected
        # of the sum of the actions log-probabilities gradients' multiplied by the advantages,
        # we do not do that, instead we take the overall sum (or mean, depending on the loss reduction).
        # This is achieved by accumulating the gradients and calling the backward method only at the end.
        state.optimizer.zero_grad(set_to_none=True)
        policy_losses, value_losses = [], []
        for batch in minibatches:
            obs = normalize_obs(batch, cfg.cnn_keys.encoder, cfg.mlp_keys.encoder + cfg.cnn_keys.encoder)
            # Compiled with `algo.compile.enabled`, without CUDA graphs: the gradients are accumulated
            with autocast(self.fabric):
                loss, pg_loss, v_loss = compiled(a2c_loss, self.fabric, self.cfg, cuda_graphs=False)(
                    state.agent,
                    obs,
                    batch["actions"],
                    batch["values"],
                    batch["returns"],
                    batch["advantages"],
                    actions_dim=tuple(int(dim) for dim in state.agent.actions_dim),
                    normalize_advantages=cfg.normalize_advantages,
                    vf_coef=cfg.vf_coef,
                    ent_coef=cfg.ent_coef,
                    reduction=cfg.loss_reduction,
                )
            self.fabric.backward(loss, inputs=params)
            policy_losses.append(pg_loss.detach())
            value_losses.append(v_loss.detach())

        # One optimizer step with the gradients of all the minibatches, averaged over the processes
        all_reduce_gradients(self.fabric, params)
        if cfg.max_grad_norm > 0.0:
            self.fabric.clip_gradients(None, state.optimizer, max_norm=cfg.max_grad_norm)
        state.optimizer.step()
        # The metrics of every minibatch
        return {"Loss/policy_loss": torch.stack(policy_losses), "Loss/value_loss": torch.stack(value_losses)}


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = A2C(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        algo.test(state, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.ppo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
