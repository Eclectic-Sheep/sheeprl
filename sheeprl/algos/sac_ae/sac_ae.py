"""SAC-AE (https://arxiv.org/abs/1910.01741): SAC from images, whose critics encode the observations with an encoder
trained also to reconstruct them through a decoder. Written on the shared training loop of `sheeprl.core`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import BatchSampler

from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.algos.sac.sac import sample_batches
from sheeprl.algos.sac_ae.agent import SACAEAgent, SACAEPlayer, build_models
from sheeprl.algos.sac_ae.utils import prepare_obs, preprocess_obs, test
from sheeprl.core import Algorithm, EnvRunner, TrainSchedule, TrainState, autocast, run, setup_module, update
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.registry import register_algorithm


@dataclass
class SACAEState(TrainState):
    # Actor, critics (with the encoder), target critics and the entropy coefficient (its logarithm, `log_alpha`)
    agent: SACAEAgent
    # The encoder of the critics, whose convolutional and MLP layers the actor shares, and the decoder: trained to
    # reconstruct the observations
    encoder: nn.Module
    decoder: nn.Module
    qf_optimizer: Optimizer
    actor_optimizer: Optimizer
    alpha_optimizer: Optimizer
    encoder_optimizer: Optimizer
    decoder_optimizer: Optimizer


class ReplayPlayer:
    """Plays in the environments and writes every step in the replay buffer: random actions until
    `algo.learning_starts`, then actions sampled from the policy. The stacked frames of an image are stored as its
    channels."""

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any], policy: SACAEPlayer, schedule: TrainSchedule) -> None:
        self.fabric = fabric
        self.cfg = cfg
        self.policy = policy
        self.schedule = schedule
        self.cnn_keys = cfg.algo.cnn_keys.encoder

    def images_as_channels(self, obs: Dict[str, np.ndarray], num_envs: int) -> Dict[str, np.ndarray]:
        return {k: v.reshape(num_envs, -1, *v.shape[-2:]) if k in self.cnn_keys else v for k, v in obs.items()}

    def step(self, env: EnvRunner, buffer: ReplayBuffer) -> None:
        num_envs = env.num_envs
        obs = self.images_as_channels(env.obs, num_envs)
        if self.schedule.warmup(env.policy_step):
            actions = env.random_actions()
        else:
            torch_obs = prepare_obs(self.fabric, obs, cnn_keys=self.cnn_keys, num_envs=num_envs)
            actions = self.policy(torch_obs).cpu().numpy()

        step = env.step(actions)

        # The observations that follow the actions: for the episodes that have just ended, their last observation,
        # not the first one of the next episode
        next_obs = {k: v.copy() for k, v in step.next_obs.items()}
        ended_envs = np.nonzero(np.logical_or(step.terminated, step.truncated))[0]
        if len(ended_envs) > 0:
            for k, final_obs in step.final_obs(ended_envs, list(next_obs)).items():
                next_obs[k][ended_envs] = final_obs
        next_obs = self.images_as_channels(next_obs, num_envs)

        data = {}
        for k in obs:
            data[k] = obs[k][np.newaxis]
            if not self.cfg.buffer.sample_next_obs:
                data[f"next_{k}"] = next_obs[k][np.newaxis]
        data["terminated"] = step.terminated.reshape(1, num_envs, -1).astype(np.float32)
        data["truncated"] = step.truncated.reshape(1, num_envs, -1).astype(np.float32)
        data["actions"] = actions.reshape(1, num_envs, -1).astype(np.float32)
        data["rewards"] = step.rewards.reshape(1, num_envs, -1).astype(np.float32)
        buffer.add(data, validate_args=self.cfg.buffer.validate_args)


class SACAE(Algorithm):
    """Every iteration plays one step in every environment and writes it in the replay buffer, then does
    `algo.replay_ratio` gradient steps per policy step, each on its own batch sampled from the buffer: the critics
    (and the encoder) at every step, the target networks, the actor and the entropy coefficient, and the encoder and
    the decoder on the reconstruction of the observations, each every `per_rank_*_freq` gradient steps."""

    off_policy = True

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        if "minedojo" in cfg.env.wrapper._target_.lower():
            raise ValueError(
                "MineDojo is not currently supported by SAC-AE agent, since it does not take "
                "into consideration the action masks provided by the environment, but needed "
                "in order to play correctly the game. "
                "As an alternative you can use one of the Dreamers' agents."
            )
        # These arguments cannot be changed
        cfg.env.screen_size = 64

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[SACAEState, ReplayBuffer]:
        cfg = self.cfg
        fabric = self.fabric
        if not isinstance(obs_space, gym.spaces.Dict):
            raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {obs_space}")
        if not isinstance(action_space, gym.spaces.Box):
            raise RuntimeError(
                f"Unexpected action space, should be of type continuous (of type Box), got: {action_space}"
            )
        keys = cfg.algo.cnn_keys, cfg.algo.mlp_keys
        if all(len(set(k.encoder).intersection(set(k.decoder))) == 0 for k in keys):
            raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjoint")
        for kind, k in zip(("CNN", "MLP"), keys):
            if len(set(k.decoder) - set(k.encoder)) > 0:
                raise RuntimeError(
                    f"The {kind} keys of the decoder must be contained in the encoder ones. "
                    f"Those keys are decoded without being encoded: {list(set(k.decoder))}"
                )
        if cfg.metric.log_level > 0:
            fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
            fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
            fabric.print("Decoder CNN keys:", cfg.algo.cnn_keys.decoder)
            fabric.print("Decoder MLP keys:", cfg.algo.mlp_keys.decoder)

        agent, encoder, decoder = build_models(cfg, obs_space, action_space, fabric.device)
        encoder = setup_module(fabric, encoder)
        decoder = setup_module(fabric, decoder)
        agent.actor = setup_module(fabric, agent.actor)
        # Setting the critic also creates the target critic, as a copy of it
        agent.critic = setup_module(fabric, agent.critic)
        agent.critic_target = setup_module(fabric, agent.critic_target)

        optimizers = [
            hydra.utils.instantiate(optimizer_cfg, params=params, _convert_="all")
            for optimizer_cfg, params in (
                (cfg.algo.critic.optimizer, agent.critic.parameters()),
                (cfg.algo.actor.optimizer, agent.actor.parameters()),
                (cfg.algo.alpha.optimizer, [agent.log_alpha]),
                (cfg.algo.encoder.optimizer, encoder.parameters()),
                (cfg.algo.decoder.optimizer, decoder.parameters()),
            )
        ]
        qf_optimizer, actor_optimizer, alpha_optimizer, encoder_optimizer, decoder_optimizer = fabric.setup_optimizers(
            *optimizers
        )

        state = SACAEState(
            agent=agent,
            encoder=encoder,
            decoder=decoder,
            qf_optimizer=qf_optimizer,
            actor_optimizer=actor_optimizer,
            alpha_optimizer=alpha_optimizer,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
        )
        buffer = ReplayBuffer(
            cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 1,
            cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
            seed=cfg.seed + fabric.global_rank,
        )
        self.schedule = schedule
        self.action_space = action_space
        return state, buffer

    def policy(self, state: SACAEState) -> SACAEPlayer:
        """The policy to play with: it shares its modules (and so its weights) with the trained actor."""
        actor = state.agent.actor.module
        # Its modules run in the precision of the run, as the actor does
        fabric = get_single_device_fabric(self.fabric)
        policy = SACAEPlayer(
            fabric.setup_module(actor.encoder),
            fabric.setup_module(actor.model),
            fabric.setup_module(actor.fc_mean),
            fabric.setup_module(actor.fc_logstd),
            action_low=self.action_space.low,
            action_high=self.action_space.high,
        )
        policy.action_scale = policy.action_scale.to(fabric.device)
        policy.action_bias = policy.action_bias.to(fabric.device)
        return policy

    def player(self, state: SACAEState) -> ReplayPlayer:
        return ReplayPlayer(self.fabric, self.cfg, self.policy(state), self.schedule)

    def batches(
        self, state: SACAEState, buffer: ReplayBuffer, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        # Sample the batches of all the gradient steps at once
        data, sampler = sample_batches(
            self.fabric, cfg, buffer, n_steps * cfg.algo.per_rank_batch_size, sample_next_obs=cfg.buffer.sample_next_obs
        )
        for batch_idxes in BatchSampler(sampler, batch_size=cfg.algo.per_rank_batch_size, drop_last=False):
            yield {k: v[batch_idxes] for k, v in data.items()}

    def train_step(self, state: SACAEState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        agent = state.agent
        cnn_keys = cfg.cnn_keys.encoder
        obs = {k: batch[k] / 255.0 if k in cnn_keys else batch[k] for k in cnn_keys + cfg.mlp_keys.encoder}
        next_obs = {
            k: batch[f"next_{k}"] / 255.0 if k in cnn_keys else batch[f"next_{k}"]
            for k in cnn_keys + cfg.mlp_keys.encoder
        }

        # Critics (and the encoder): regress the soft Q-values towards the one-step target of the target critics
        with autocast(self.fabric):
            target_qf_values = agent.get_next_target_q_values(
                next_obs, batch["rewards"], batch["terminated"], cfg.gamma
            )
            qf_values = agent.get_q_values(obs, batch["actions"])
            qf_loss = critic_loss(qf_values, target_qf_values, agent.num_critics)
        update(self.fabric, qf_loss, state.qf_optimizer)
        metrics = {"Loss/value_loss": qf_loss.detach()}
        if step % cfg.critic.per_rank_target_network_update_freq == 0:
            agent.critic_target_ema()
            agent.critic_encoder_target_ema()

        # Actor: maximize the smallest Q-value of its actions plus their entropy, on the features of the encoder
        # (not trained by this loss)
        if step % cfg.actor.per_rank_update_freq == 0:
            with autocast(self.fabric):
                actions, logprobs = agent.get_actions_and_log_probs(obs, detach_encoder_features=True)
                qf_values = agent.get_q_values(obs, actions, detach_encoder_features=True)
                min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
                actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
            update(self.fabric, actor_loss, state.actor_optimizer)

            # Entropy coefficient: towards the target entropy
            alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
            update(self.fabric, alpha_loss, state.alpha_optimizer)
            metrics["Loss/policy_loss"] = actor_loss.detach()
            metrics["Loss/alpha_loss"] = alpha_loss.detach()

        # Encoder and decoder: reconstruct the observations (the images with 5 bits per channel), with an L2 penalty
        # on the features, once for every decoded key
        if step % cfg.decoder.per_rank_update_freq == 0:
            with autocast(self.fabric):
                hidden = state.encoder(obs)
                reconstruction = state.decoder(hidden)
                reconstruction_loss = 0
                for k in cfg.cnn_keys.decoder + cfg.mlp_keys.decoder:
                    target = preprocess_obs(batch[k], bits=5) if k in cfg.cnn_keys.decoder else batch[k]
                    reconstruction_loss += (
                        F.mse_loss(target, reconstruction[k])
                        + cfg.decoder.l2_lambda * (0.5 * hidden.pow(2).sum(1)).mean()
                    )
            # One backward pass for both, then the step of the encoder and the one of the decoder
            state.decoder_optimizer.zero_grad(set_to_none=True)
            params = [*state.encoder.parameters(), *state.decoder.parameters()]
            update(self.fabric, reconstruction_loss, state.encoder_optimizer, params=params)
            state.decoder_optimizer.step()
            metrics["Loss/reconstruction_loss"] = reconstruction_loss.detach()
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = SACAE(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        test(algo.policy(state), fabric, cfg, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac_ae.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(
            fabric, log_models, cfg, {"agent": state.agent, "encoder": state.encoder, "decoder": state.decoder}
        )
