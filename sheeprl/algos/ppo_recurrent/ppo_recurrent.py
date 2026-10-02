"""PPO with a recurrent (LSTM) agent, written on the shared training loop of `sheeprl.core`: `PPORecurrent` says how to
build, play and train; `sheeprl.core.loop.run` does the rest."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.sampler import BatchSampler, RandomSampler

from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.ppo import anneal, annealed_values
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent, RecurrentPPOPlayer
from sheeprl.algos.ppo_recurrent.utils import prepare_obs, test
from sheeprl.core import Algorithm, EnvRunner, Rollout, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae, normalize_tensor


@dataclass
class PPORecurrentState(TrainState):
    # Feature extractor, LSTM, actor and critic
    agent: RecurrentPPOAgent
    optimizer: Optimizer
    # Annealed values are tensors, so that changing them never recompiles a compiled training step (the annealed
    # learning rate is the one of the optimizer)
    clip_coef: Tensor
    ent_coef: Tensor


@dataclass
class RecurrentRollout(Rollout):
    """The rollout of PPO-recurrent. With the observations that follow its last step, the actions and the recurrent
    states of that step give the value that bootstraps the returns."""

    actions: Optional[Tensor] = None
    states: Optional[Tuple[Tensor, Tensor]] = None


class RecurrentRolloutPlayer:
    """Plays the policy in the environments and writes every step in the rollout, with the recurrent state and the
    actions that preceded it (the input of the LSTM). The recurrent state is reset at the end of every episode when
    `algo.reset_recurrent_state_on_done` is set, the previous actions always."""

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any], policy: RecurrentPPOPlayer) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.policy = policy
        self.cnn_keys = cfg.algo.cnn_keys.encoder
        self.obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
        # The recurrent states and the actions preceding the next step; created at the first step
        self.prev_states: Optional[Tuple[Tensor, Tensor]] = None
        self.prev_actions: Optional[np.ndarray] = None

    def step(self, env: EnvRunner, rollout: RecurrentRollout) -> None:
        cfg = self.cfg
        num_envs = env.num_envs
        device = self.fabric.device
        if self.prev_states is None:
            hidden_size = self.policy.rnn_hidden_size
            self.prev_states = (
                torch.zeros(1, num_envs, hidden_size, device=device),
                torch.zeros(1, num_envs, hidden_size, device=device),
            )
            self.prev_actions = np.zeros((1, num_envs, sum(self.policy.actions_dim)))

        # The stacked frames of an image are stored as its channels
        obs = {}
        for k in self.obs_keys:
            obs[k] = env.obs[k]
            if k in self.cnn_keys:
                obs[k] = obs[k].reshape(num_envs, -1, *obs[k].shape[-2:])
            obs[k] = obs[k][np.newaxis]
        torch_obs = prepare_obs(self.fabric, obs, cnn_keys=self.cnn_keys, num_envs=num_envs)
        torch_prev_actions = torch.from_numpy(self.prev_actions).to(device).float()
        actions, logprobs, values, states = self.policy(
            torch_obs, prev_actions=torch_prev_actions, prev_states=self.prev_states
        )
        if self.policy.actor.is_continuous:
            env_actions = torch.stack(actions, -1).cpu().numpy()
        else:
            env_actions = torch.stack([act.argmax(dim=-1) for act in actions], dim=-1).cpu().numpy()
        torch_actions = torch.cat(actions, dim=-1)
        actions = torch_actions.cpu().numpy()

        step = env.step(env_actions)

        rewards = np.tanh(step.rewards) if cfg.env.clip_rewards else step.rewards
        # The episodes truncated by the time limit (and not terminated in the same step) don't end in the MDP:
        # bootstrap the value of their final observation, in the scale of the clipped rewards the critic learns
        truncated_envs = np.nonzero(np.logical_and(step.truncated, np.logical_not(step.terminated)))[0]
        if len(truncated_envs) > 0:
            final_obs = prepare_obs(
                self.fabric,
                step.final_obs(truncated_envs, self.obs_keys),
                cnn_keys=self.cnn_keys,
                num_envs=len(truncated_envs),
            )
            final_values, _ = self.policy.get_values(
                final_obs, torch_actions[:, truncated_envs, :], tuple(s[:, truncated_envs, ...] for s in states)
            )
            final_values = final_values.view(rewards[truncated_envs].shape).cpu().numpy()
            rewards[truncated_envs] += cfg.algo.gamma * final_values.reshape(rewards[truncated_envs].shape)
        dones = np.logical_or(step.terminated, step.truncated).reshape(1, num_envs, -1).astype(np.float32)
        rewards = rewards.reshape(1, num_envs, -1).astype(np.float32)

        data = dict(obs)
        data["dones"] = dones
        data["values"] = values.cpu().numpy().reshape(1, num_envs, -1)
        data["actions"] = actions.reshape(1, num_envs, -1)
        data["rewards"] = rewards
        data["logprobs"] = logprobs.cpu().numpy()
        data["prev_hx"] = self.prev_states[0].cpu().numpy().reshape(1, num_envs, -1)
        data["prev_cx"] = self.prev_states[1].cpu().numpy().reshape(1, num_envs, -1)
        data["prev_actions"] = self.prev_actions.reshape(1, num_envs, -1)
        if cfg.buffer.memmap:
            data["returns"] = np.zeros_like(rewards)
            data["advantages"] = np.zeros_like(rewards)
        rollout.add(data, step.next_obs, validate_args=cfg.buffer.validate_args)
        rollout.actions, rollout.states = torch_actions, states

        # The next step starts a new episode where this one ended one
        self.prev_actions = (1 - dones) * actions
        if cfg.algo.reset_recurrent_state_on_done:
            self.prev_states = tuple((1 - torch.as_tensor(dones, device=device)) * s for s in states)
        else:
            self.prev_states = states


def split_in_sequences(data: Dict[str, Tensor], sequence_length: int) -> Dict[str, Tensor]:
    """Split the rollout (`[Rollout_Steps, Num_Envs, ...]`) of every environment at the end of its episodes, and every
    episode in sequences of `sequence_length` steps (the last one can be shorter).

    Returns the sequences, `[Sequence_Length, Num_Sequences, ...]`, zero-padded to `sequence_length` steps, with
    their `mask` (`True` on the steps of the sequences).
    """
    num_steps, num_envs = data["dones"].shape[:2]
    sequences: Dict[str, List[Tensor]] = {k: [] for k in data}
    lengths: List[int] = []
    for env_idx in range(num_envs):
        env_data = {k: v[:, env_idx].float() for k, v in data.items()}
        # An episode ends with its done step
        episode_ends = env_data["dones"].nonzero(as_tuple=True)[0].tolist() + [num_steps]
        start = 0
        for end in episode_ends:
            if start <= end and start < num_steps:
                for k, v in env_data.items():
                    sequences[k].extend(torch.split(v[start : end + 1], sequence_length))
                lengths.extend(len(s) for s in torch.split(env_data["dones"][start : end + 1], sequence_length))
            start = end + 1
    padded = {}
    for k, v in sequences.items():
        padded[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=False, padding_value=0)
        if padded[k].shape[0] < sequence_length:
            padding = padded[k].new_zeros(sequence_length - padded[k].shape[0], *padded[k].shape[1:])
            padded[k] = torch.cat((padded[k], padding), dim=0)
    lengths = torch.as_tensor(lengths)
    padded["mask"] = (torch.arange(sequence_length).expand(len(lengths), sequence_length) < lengths.unsqueeze(1)).T
    padded["mask"] = padded["mask"].to(data["dones"].device)
    return padded


class PPORecurrent(Algorithm):
    """Every iteration plays `algo.rollout_steps` steps, splits the rollout of every environment in sequences of
    `algo.per_rank_sequence_length` steps that don't cross the end of an episode, then trains for `algo.update_epochs`
    epochs of `algo.per_rank_num_batches` minibatches of sequences, each starting from the recurrent state stored at
    its first step."""

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        if "minedojo" in cfg.env.wrapper._target_.lower():
            raise ValueError(
                "MineDojo is not currently supported by PPO Recurrent agent, since it does not take "
                "into consideration the action masks provided by the environment, but needed "
                "in order to play correctly the game. "
                "As an alternative you can use one of the Dreamers' agents."
            )
        if cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder == []:
            raise RuntimeError(
                "You should specify at least one CNN keys or MLP keys from the cli: "
                "`cnn_keys.encoder=[rgb]` or `mlp_keys.encoder=[state]`"
            )
        if not cfg.algo.per_rank_sequence_length or cfg.algo.per_rank_sequence_length <= 0:
            raise ValueError(f"The sequence length must be greater than zero, got: {cfg.algo.per_rank_sequence_length}")
        if cfg.buffer.share_data:
            warnings.warn(
                "The script has been called with `buffer.share_data=True`: "
                "with recurrent PPO only gradients are shared"
            )
        self.steps_per_iteration = cfg.algo.rollout_steps

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[PPORecurrentState, RecurrentRollout]:
        cfg = self.cfg
        if not isinstance(obs_space, gym.spaces.Dict):
            raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {obs_space}")
        if cfg.metric.log_level > 0:
            self.fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
            self.fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
        is_continuous = isinstance(action_space, gym.spaces.Box)
        is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        self.actions_dim = tuple(
            action_space.shape
            if is_continuous
            else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
        )

        agent = RecurrentPPOAgent(
            actions_dim=self.actions_dim,
            obs_space=obs_space,
            encoder_cfg=cfg.algo.encoder,
            rnn_cfg=cfg.algo.rnn,
            actor_cfg=cfg.algo.actor,
            critic_cfg=cfg.algo.critic,
            cnn_keys=cfg.algo.cnn_keys.encoder,
            mlp_keys=cfg.algo.mlp_keys.encoder,
            is_continuous=is_continuous,
            distribution_cfg=cfg.distribution,
            num_envs=cfg.env.num_envs,
            screen_size=cfg.env.screen_size,
            device=self.fabric.device,
        )
        agent.feature_extractor = setup_module(self.fabric, agent.feature_extractor)
        agent.rnn = setup_module(self.fabric, agent.rnn)
        agent.critic = setup_module(self.fabric, agent.critic)
        agent.actor = setup_module(self.fabric, agent.actor)

        optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")
        optimizer = self.fabric.setup_optimizers(optimizer)
        self.total_iters = schedule.total_iters

        state = PPORecurrentState(
            agent=agent,
            optimizer=optimizer,
            clip_coef=torch.tensor(cfg.algo.clip_coef, device=self.fabric.device),
            ent_coef=torch.tensor(cfg.algo.ent_coef, device=self.fabric.device),
        )
        # One rollout, whatever `buffer.size`
        buffer = ReplayBuffer(
            cfg.algo.rollout_steps,
            cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{self.fabric.global_rank}"),
            obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
        )
        return state, RecurrentRollout(buffer)

    def policy(self, state: PPORecurrentState) -> RecurrentPPOPlayer:
        """The policy to play with: it shares its modules (and so its weights) with the trained agent."""
        agent = state.agent
        return RecurrentPPOPlayer(
            agent.feature_extractor,
            agent.rnn,
            agent.actor,
            agent.critic,
            self.cfg.algo.rnn.lstm.hidden_size,
            self.actions_dim,
        )

    def player(self, state: PPORecurrentState) -> RecurrentRolloutPlayer:
        return RecurrentRolloutPlayer(self.fabric, self.cfg, self.policy(state))

    def batches(
        self, state: PPORecurrentState, rollout: RecurrentRollout, n_steps: Optional[int], iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        anneal(cfg.algo, state, iteration, self.total_iters)
        data = rollout.buffer.to_tensor(dtype=None, device=self.fabric.device, from_numpy=cfg.buffer.from_numpy)

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.inference_mode():
            next_obs = {}
            for k in cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder:
                next_obs[k] = rollout.next_obs[k]
                if k in cfg.algo.cnn_keys.encoder:
                    next_obs[k] = next_obs[k].reshape(cfg.env.num_envs, -1, *next_obs[k].shape[-2:])
            next_obs = prepare_obs(self.fabric, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
            next_values, _ = self.policy(state).get_values(next_obs, rollout.actions, rollout.states)
            returns, advantages = gae(
                data["rewards"].to(torch.float64),
                data["values"],
                data["dones"],
                next_values,
                cfg.algo.rollout_steps,
                cfg.algo.gamma,
                cfg.algo.gae_lambda,
            )
            data["rewards"] = data["rewards"].float()
            data["returns"] = returns.float()
            data["advantages"] = advantages.float()

        # Sequences of `per_rank_sequence_length` steps that don't cross the end of an episode
        sequences = split_in_sequences(data, cfg.algo.per_rank_sequence_length)
        num_sequences = sequences["mask"].shape[1]
        if cfg.algo.per_rank_num_batches > 0:
            batch_size = num_sequences // cfg.algo.per_rank_num_batches
            batch_size = batch_size if batch_size > 0 else num_sequences
        else:
            batch_size = 1
        # Every process splits its own rollout, so they can have different numbers of minibatches, while every
        # gradient step averages the gradients of all of them: the processes with fewer minibatches do the
        # missing steps with a zero loss, after their last minibatch (as `torch.distributed.algorithms.Join` does).
        # They agree on the number of steps before the first one
        num_batches = cfg.algo.update_epochs * -(-num_sequences // batch_size)
        padding = 0
        if self.fabric.world_size > 1:
            all_batches = self.fabric.all_reduce(torch.tensor(num_batches, device=self.fabric.device), reduce_op="max")
            padding = int(all_batches.item()) - num_batches
        for _ in range(cfg.algo.update_epochs):
            sampler = BatchSampler(RandomSampler(range(num_sequences)), batch_size=batch_size, drop_last=False)
            for idxes in sampler:
                batch = {k: v[:, idxes] for k, v in sequences.items()}
                yield batch
        for _ in range(padding):
            yield {**batch, "loss_weight": torch.zeros((), device=self.fabric.device)}

    def train_step(self, state: PPORecurrentState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        batch = dict(batch)
        mask = batch["mask"].unsqueeze(-1)
        for k in cfg.cnn_keys.encoder:
            batch[k] = batch[k] / 255.0 - 0.5
        with autocast(self.fabric):
            _, logprobs, entropies, values, _ = state.agent(
                {k: batch[k] for k in set(cfg.cnn_keys.encoder + cfg.mlp_keys.encoder)},
                prev_actions=batch["prev_actions"],
                prev_states=(batch["prev_hx"][:1], batch["prev_cx"][:1]),
                actions=torch.split(batch["actions"], state.agent.actions_dim, dim=-1),
                mask=mask,
            )
            normalized_advantages = batch["advantages"][mask]
            if cfg.normalize_advantages and len(normalized_advantages) > 1:
                normalized_advantages = normalize_tensor(normalized_advantages)
            pg_loss = policy_loss(
                logprobs[mask], batch["logprobs"][mask], normalized_advantages, state.clip_coef, "mean"
            )
            v_loss = value_loss(
                values[mask], batch["values"][mask], batch["returns"][mask], state.clip_coef, cfg.clip_vloss, "mean"
            )
            ent_loss = entropy_loss(entropies[mask], cfg.loss_reduction)
            # Equation (9) in the paper
            loss = pg_loss + cfg.vf_coef * v_loss + state.ent_coef * ent_loss
            if "loss_weight" in batch:
                loss = loss * batch["loss_weight"]
        update(self.fabric, loss, state.optimizer, max_grad_norm=cfg.max_grad_norm)
        if "loss_weight" in batch:
            return {}
        return {
            "Loss/policy_loss": pg_loss.detach(),
            "Loss/value_loss": v_loss.detach(),
            "Loss/entropy_loss": ent_loss.detach(),
        }

    def end_iteration(self, state: PPORecurrentState, iteration: int) -> Dict[str, float]:
        return annealed_values(state)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = PPORecurrent(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.ppo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
