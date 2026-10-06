"""PPO with a recurrent (LSTM) agent, written on the shared training loop of `sheeprl.core`: `PPORecurrent` says how to
build, play and train; `sheeprl.core.loop.run` does the rest."""

from __future__ import annotations

import copy
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

from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.utils import anneal, bootstrap_truncated
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent, RecurrentPPOPlayer, build_agent
from sheeprl.algos.ppo_recurrent.utils import prepare_obs, test
from sheeprl.core import (
    Algorithm,
    Environment,
    Player,
    ReplayStore,
    TrainSchedule,
    TrainState,
    autocast,
    rollout_store,
    run,
    update,
)
from sheeprl.utils.compile import compile_enabled, compiled, mark_gradient_step
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae_function, normalize_tensor


@dataclass
class PPORecurrentState(TrainState):
    # Feature extractor, LSTM, actor and critic
    agent: RecurrentPPOAgent
    optimizer: Optimizer


class RecurrentRolloutPlayer(Player):
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

    def step(self, env: Environment, rollout: ReplayStore) -> None:
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

        def final_values(env_idxes: np.ndarray) -> np.ndarray:
            final_obs = step.stack_final_obs(env_idxes, self.obs_keys)
            final_obs = prepare_obs(self.fabric, final_obs, cnn_keys=self.cnn_keys, num_envs=len(env_idxes))
            values, _ = self.policy.get_values(
                final_obs, torch_actions[:, env_idxes, :], tuple(s[:, env_idxes, ...] for s in states)
            )
            return values.cpu().numpy()

        # The episodes truncated by the time limit (and not terminated in the same step) don't end in the MDP: the
        # value of their final observation is added to the reward, after the rewards are clipped
        rewards = np.tanh(step.rewards) if cfg.env.clip_rewards else step.rewards
        rewards = bootstrap_truncated(rewards, step.terminated, step.truncated, final_values, cfg.algo.gamma)
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
        rollout.add(data, validate_args=cfg.buffer.validate_args)
        # The observations after the last step of the rollout bootstrap its returns
        rollout.context["next_obs"] = step.next_obs
        rollout.context["actions"], rollout.context["states"] = torch_actions, states

        # The next step starts a new episode where this one ended one
        self.prev_actions = (1 - dones) * actions
        if cfg.algo.reset_recurrent_state_on_done:
            self.prev_states = tuple((1 - torch.as_tensor(dones, device=device)) * s for s in states)
        else:
            self.prev_states = states


# Compiled, the minibatches are padded to a multiple of this number of sequences (`PPORecurrent.batches`)
SEQUENCES_MULTIPLE = 16
# The entries of a minibatch given to the loss as they are, `[Sequence_Length, Num_Sequences, ...]`
SEQUENCES_KEYS = ("prev_actions", "actions", "logprobs", "values", "returns", "advantages")


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


@torch.no_grad()
def recurrent_states(
    agent: RecurrentPPOAgent,
    obs: Dict[str, Tensor],
    prev_actions: Tensor,
    dones: Tensor,
    initial_states: Tuple[Tensor, Tensor],
    reset_on_done: bool,
) -> Tuple[Tensor, Tensor]:
    """The recurrent states before every step of a rollout (`[Rollout_Steps, Num_Envs, Hidden_Size]`), unrolled with the
    current weights of `agent` from the states before its first step (`initial_states`), and reset after the end of
    every episode as the player does when `reset_on_done` (`algo.reset_recurrent_state_on_done`)."""
    rnn = agent.rnn
    x = rnn._pre_mlp(torch.cat((agent.feature_extractor(obs), prev_actions), dim=-1))
    rnn._lstm.flatten_parameters()
    hx, cx = initial_states
    all_hx, all_cx = [], []
    for t in range(x.shape[0]):
        all_hx.append(hx)
        all_cx.append(cx)
        _, (hx, cx) = rnn._lstm(x[t : t + 1], (hx, cx))
        if reset_on_done:
            hx, cx = (1 - dones[t : t + 1]) * hx, (1 - dones[t : t + 1]) * cx
    return torch.cat(all_hx), torch.cat(all_cx)


def masked_mean(tensor: Tensor, mask: Tensor) -> Tensor:
    """The mean of the elements of `tensor` selected by `mask`."""
    return torch.where(mask, tensor, 0).sum() / mask.sum()


def ppo_recurrent_loss(
    agent: RecurrentPPOAgent,
    obs: Dict[str, Tensor],
    prev_actions: Tensor,
    prev_states: Tuple[Tensor, Tensor],
    actions: Tensor,
    mask: Tensor,
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
    entropy_reduction: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The loss of PPO on a minibatch of sequences, on their steps selected by `mask`, and its terms. Can be compiled
    (`algo.compile`)."""
    _, new_logprobs, entropies, new_values, _ = agent(
        obs,
        prev_actions=prev_actions,
        prev_states=prev_states,
        actions=torch.split(actions, actions_dim, dim=-1),
        mask=mask,
    )
    # The steps of the sequences are selected by masked sums, not by indexing them: the shapes don't depend on the data
    if normalize_advantages:
        advantages = normalize_tensor(advantages, mask=mask)
    pg_loss = masked_mean(policy_loss(new_logprobs, logprobs, advantages, clip_coef, "none"), mask)
    v_loss = masked_mean(value_loss(new_values, values, returns, clip_coef, clip_vloss, "none"), mask)
    entropies = entropy_loss(entropies, "none")
    ent_loss = masked_mean(entropies, mask) if entropy_reduction == "mean" else torch.where(mask, entropies, 0).sum()
    # Equation (9) in the paper
    return pg_loss + vf_coef * v_loss + ent_coef * ent_loss, pg_loss, v_loss, ent_loss


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
        self.gae = gae_function(cfg.algo.gae_method)
        # The values the annealed coefficients start from
        self.initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)
        self.initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[PPORecurrentState, ReplayStore]:
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

        state = PPORecurrentState(agent=agent, optimizer=optimizer)
        # One rollout, whatever `buffer.size`
        return state, rollout_store(self.fabric, cfg, log_dir, cfg.algo.rollout_steps)

    def policy(self, state: PPORecurrentState) -> RecurrentPPOPlayer:
        """The policy to play with: it shares the modules (and so the weights) of the trained agent (`build_agent`)."""
        return self._policy

    def test(self, state: TrainState, log_dir: str, policy_step: int = 0) -> None:
        test(self.policy(state), self.fabric, self.cfg, log_dir, policy_step=policy_step)

    def player(self, state: PPORecurrentState) -> RecurrentRolloutPlayer:
        return RecurrentRolloutPlayer(self.fabric, self.cfg, self.policy(state))

    def batches(
        self, state: PPORecurrentState, rollout: ReplayStore, n_steps: Optional[int], iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        # The learning rate and the coefficients of the iteration
        anneal(cfg, state.optimizer, iteration, self.total_iters, self.initial_clip_coef, self.initial_ent_coef)
        self.clip_coef.fill_(cfg.algo.clip_coef)
        self.ent_coef.fill_(cfg.algo.ent_coef)
        data = rollout.read()

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.inference_mode():
            next_obs = {}
            for k in cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder:
                next_obs[k] = rollout.context["next_obs"][k]
                if k in cfg.algo.cnn_keys.encoder:
                    next_obs[k] = next_obs[k].reshape(cfg.env.num_envs, -1, *next_obs[k].shape[-2:])
            next_obs = prepare_obs(self.fabric, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
            next_values, _ = self.policy(state).get_values(
                next_obs, rollout.context["actions"], rollout.context["states"]
            )
            returns, advantages = self.gae(
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
        # The number of sequences changes with the episodes that end in the rollout, and so do the sizes of the
        # minibatches: compiled with CUDA graphs, every new size is a new recording. Every minibatch of the rollout,
        # the last one included, is padded to the size rounded up to a multiple of `SEQUENCES_MULTIPLE`, with copies
        # of its first sequence out of the mask: the losses don't change and a run sees a few sizes
        padded_size = (
            -(-batch_size // SEQUENCES_MULTIPLE) * SEQUENCES_MULTIPLE if compile_enabled(self.fabric, cfg) else 0
        )
        for epoch in range(cfg.algo.update_epochs):
            if cfg.algo.refresh_recurrent_states and epoch > 0:
                # The sequences start from the recurrent states of the rollout, computed by the weights that played it:
                # every epoch after the first one unrolls them again with the current weights, from the state before
                # the first step of the rollout
                # As the sequences, in single precision (`split_in_sequences`)
                obs = {k: data[k].float() / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
                obs.update({k: data[k].float() for k in cfg.algo.mlp_keys.encoder})
                with autocast(self.fabric):
                    prev_hx, prev_cx = recurrent_states(
                        state.agent,
                        obs,
                        data["prev_actions"].float(),
                        data["dones"].float(),
                        (data["prev_hx"][:1].float(), data["prev_cx"][:1].float()),
                        cfg.algo.reset_recurrent_state_on_done,
                    )
                refreshed = split_in_sequences(
                    {"dones": data["dones"], "prev_hx": prev_hx.float(), "prev_cx": prev_cx.float()},
                    cfg.algo.per_rank_sequence_length,
                )
                sequences["prev_hx"], sequences["prev_cx"] = refreshed["prev_hx"], refreshed["prev_cx"]
            for idxes, size in rollout.sampler.epochs(
                num_sequences, 1, pad_to=padded_size or None, batch_size=batch_size
            ):
                batch = {k: v[:, idxes] for k, v in sequences.items()}
                batch["mask"][:, size:] = False
                yield batch
        for _ in range(padding):
            yield {**batch, "loss_weight": torch.zeros((), device=self.fabric.device)}

    def train_step(self, state: PPORecurrentState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        batch = dict(batch)
        for k in cfg.cnn_keys.encoder:
            batch[k] = batch[k] / 255.0 - 0.5
        obs = {k: batch[k] for k in set(cfg.cnn_keys.encoder + cfg.mlp_keys.encoder)}
        states = (batch["prev_hx"][:1], batch["prev_cx"][:1])
        mask = batch["mask"].unsqueeze(-1)
        if compile_enabled(self.fabric, self.cfg):
            # The number of sequences changes between the rollouts (`batches`): compiled once for all of them
            for tensor in (*obs.values(), *states, mask, *(batch[k] for k in SEQUENCES_KEYS)):
                torch._dynamo.maybe_mark_dynamic(tensor, 1)
        # The loss is compiled when `algo.compile.enabled` is set
        mark_gradient_step(self.fabric, self.cfg)
        with autocast(self.fabric):
            loss, pg_loss, v_loss, ent_loss = compiled(ppo_recurrent_loss, self.fabric, self.cfg)(
                state.agent,
                obs,
                batch["prev_actions"],
                states,
                batch["actions"],
                mask,
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
                entropy_reduction=cfg.loss_reduction,
            )
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


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = PPORecurrent(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        algo.test(state, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.ppo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
