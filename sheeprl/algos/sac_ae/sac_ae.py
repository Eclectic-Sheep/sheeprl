"""SAC-AE (https://arxiv.org/abs/1910.01741): SAC from images, whose critics encode the observations with an encoder
trained also to reconstruct them through a decoder. Written on the shared training loop of `sheeprl.core`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.optim import Optimizer

from sheeprl.algos.sac.loss import critic_loss, policy_loss
from sheeprl.algos.sac_ae.agent import SACAEAgent, SACAEPolicy, build_agent, tie_actor_convolutions, tie_actor_optimizer
from sheeprl.algos.sac_ae.loss import entropy_loss
from sheeprl.algos.sac_ae.utils import preprocess_obs
from sheeprl.core import Act, Algorithm, EnvStep, TrainSchedule, TrainState, Writer, run, transition_store, update
from sheeprl.data.store import ReplayStore
from sheeprl.models.models import MultiDecoder, MultiEncoder
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.fabric import autocast_cache_scope
from sheeprl.utils.obs import images_as_channels
from sheeprl.utils.registry import register_algorithm

# The most gradient steps whose batches are sampled (and moved to the device) at once
MAX_SAMPLED_BATCHES = 16


def critic_loss_fn(
    agent: SACAEAgent,
    observations: Dict[str, Tensor],
    next_observations: Dict[str, Tensor],
    actions: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    gamma: float,
) -> Tensor:
    """The loss of the critics, with the targets of the target critics on the next observations."""
    next_target_qf_value = agent.get_next_target_q_values(next_observations, rewards, terminated, gamma)
    qf_values = agent.get_q_values(observations, actions)
    return critic_loss(qf_values, next_target_qf_value, agent.num_critics)


def actor_loss_fn(agent: SACAEAgent, observations: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    """The loss of the actor, which doesn't train the convolutions of the encoder, and the log-probabilities of its
    actions, for the loss of the temperature."""
    actions, logprobs = agent.get_actions_and_log_probs(observations, detach_encoder_features=True)
    qf_values = agent.get_q_values(observations, actions, detach_encoder_features=True)
    min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
    return policy_loss(agent.log_alpha.exp().detach(), logprobs, min_qf_values), logprobs.detach()


def reconstruction_loss_fn(
    encoder: Union[MultiEncoder, _FabricModule],
    decoder: Union[MultiDecoder, _FabricModule],
    observations: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    cnn_keys: Tuple[str, ...],
    mlp_keys: Tuple[str, ...],
    l2_lambda: float,
) -> Tensor:
    """The loss of the reconstruction of the observations (the images dequantized), with an L2 penalty on the hidden
    state."""
    hidden = encoder(observations)
    reconstruction = decoder(hidden)
    reconstruction_loss = 0
    for k in cnn_keys + mlp_keys:
        target = preprocess_obs(targets[k], bits=5) if k in cnn_keys else targets[k]
        reconstruction_loss += (
            F.mse_loss(target, reconstruction[k])  # Reconstruction
            + l2_lambda * (0.5 * hidden.pow(2).sum(1)).mean()  # L2 penalty on the hidden state
        )
    return reconstruction_loss


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

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # The checkpoints saved when the actor kept its own convolutions, never trained (see `SACAEAgent`): the actor
        # gets the ones of the critic, and its optimizer forgets the others
        state = dict(state)
        if "actor_optimizer" in state:
            state["actor_optimizer"] = tie_actor_optimizer(state["actor_optimizer"], state["agent"])
        state["agent"] = tie_actor_convolutions(state["agent"])
        super().load_state_dict(state)


class ReplayWriter(Writer):
    """Writes every step in the replay buffer, with the actions played in the environments. The stacked frames of an
    image are stored as its channels."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.cnn_keys = cfg.algo.cnn_keys.encoder

    def write(self, buffer: ReplayStore, step: EnvStep, act: Act) -> None:
        num_envs = len(step.rewards)
        obs = images_as_channels(step.obs, self.cnn_keys, num_envs)

        # The observations that follow the actions: for the episodes that have just ended, their last observation,
        # not the first one of the next episode
        next_obs = {k: v.copy() for k, v in step.next_obs.items()}
        ended_envs = np.nonzero(np.logical_or(step.terminated, step.truncated))[0]
        if len(ended_envs) > 0:
            for k, final_obs in step.stack_final_obs(ended_envs, list(next_obs)).items():
                next_obs[k][ended_envs] = final_obs
        next_obs = images_as_channels(next_obs, self.cnn_keys, num_envs)

        data = {}
        for k in obs:
            data[k] = obs[k][np.newaxis]
            if not self.cfg.buffer.sample_next_obs:
                data[f"next_{k}"] = next_obs[k][np.newaxis]
        data["terminated"] = step.terminated.reshape(1, num_envs, -1).astype(np.float32)
        data["truncated"] = step.truncated.reshape(1, num_envs, -1).astype(np.float32)
        data["actions"] = act.env_actions.reshape(1, num_envs, -1).astype(np.float32)
        data["rewards"] = step.rewards.reshape(1, num_envs, -1).astype(np.float32)
        buffer.add(data, validate_args=self.cfg.buffer.validate_args)


def train(
    fabric: Fabric,
    agent: SACAEAgent,
    encoder: Union[MultiEncoder, _FabricModule],
    decoder: Union[MultiDecoder, _FabricModule],
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    encoder_optimizer: Optimizer,
    decoder_optimizer: Optimizer,
    data: Dict[str, Tensor],
    cumulative_per_rank_gradient_steps: int,
    cfg: Dict[str, Any],
):
    metrics: Dict[str, Tensor] = {}
    normalized_next_obs = {}
    normalized_obs = {}
    for k in cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder:
        if k in cfg.algo.cnn_keys.encoder:
            normalized_obs[k] = data[k] / 255.0
            normalized_next_obs[k] = data[f"next_{k}"] / 255.0
        else:
            normalized_obs[k] = data[k]
            normalized_next_obs[k] = data[f"next_{k}"]

    # The losses are compiled when `algo.compile.enabled` is set
    mark_gradient_step(fabric, cfg)

    # Update the soft-critic
    with autocast_cache_scope(fabric):
        qf_loss = compiled(critic_loss_fn, fabric, cfg)(
            agent,
            normalized_obs,
            normalized_next_obs,
            data["actions"],
            data["rewards"],
            data["terminated"],
            cfg.algo.gamma,
        )
    update(fabric, qf_loss, qf_optimizer)
    metrics["Loss/value_loss"] = qf_loss

    # Update the target networks with EMA
    if cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq == 0:
        agent.critic_target_ema()
        agent.critic_encoder_target_ema()

    # Update the actor
    if cumulative_per_rank_gradient_steps % cfg.algo.actor.per_rank_update_freq == 0:
        with autocast_cache_scope(fabric):
            actor_loss, logprobs = compiled(actor_loss_fn, fabric, cfg)(agent, normalized_obs)
        update(fabric, actor_loss, actor_optimizer)

        # Update the entropy value
        alpha_loss = entropy_loss(agent.log_alpha, logprobs, agent.target_entropy)
        update(fabric, alpha_loss, alpha_optimizer)

        metrics["Loss/policy_loss"] = actor_loss
        metrics["Loss/alpha_loss"] = alpha_loss

    # Update the decoder
    if cumulative_per_rank_gradient_steps % cfg.algo.decoder.per_rank_update_freq == 0:
        with autocast_cache_scope(fabric):
            reconstruction_loss = compiled(reconstruction_loss_fn, fabric, cfg)(
                encoder,
                decoder,
                normalized_obs,
                data,
                tuple(cfg.algo.cnn_keys.decoder),
                tuple(cfg.algo.mlp_keys.decoder),
                cfg.algo.decoder.l2_lambda,
            )
        # One backward pass for both, then the step of the encoder and the one of the decoder
        decoder_optimizer.zero_grad(set_to_none=True)
        update(
            fabric,
            reconstruction_loss,
            encoder_optimizer,
            params=[*encoder.parameters(), *decoder.parameters()],
        )
        decoder_optimizer.step()
        metrics["Loss/reconstruction_loss"] = reconstruction_loss
    return metrics


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

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[SACAEState, ReplayStore]:
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

        agent, encoder, decoder, self._policy = build_agent(fabric, cfg, obs_space, action_space)

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
        self.schedule = schedule
        obs_keys = tuple(cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder)
        return state, transition_store(fabric, cfg, log_dir, obs_keys=obs_keys)

    def policy(self, state: SACAEState) -> SACAEPolicy:
        """The policy to play with: it shares its weights with the trained actor (`build_agent`)."""
        return self._policy

    def writer(self, state: SACAEState, policy: SACAEPolicy) -> ReplayWriter:
        return ReplayWriter(self.cfg)

    def batches(
        self, state: SACAEState, buffer: ReplayStore, n_steps: int, iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        # The batches of the gradient steps are sampled `MAX_SAMPLED_BATCHES` at a time: the images of all the ones of
        # the first training (with the pretraining) don't fit in the memory
        yield from buffer.batches(n_steps, self.cfg.algo.per_rank_batch_size, MAX_SAMPLED_BATCHES)

    def train_step(self, state: SACAEState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        metrics = train(
            self.fabric,
            state.agent,
            state.encoder,
            state.decoder,
            state.actor_optimizer,
            state.qf_optimizer,
            state.alpha_optimizer,
            state.encoder_optimizer,
            state.decoder_optimizer,
            batch,
            step,
            self.cfg,
        )
        return metrics


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = SACAE(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        algo.test(state, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac_ae.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(
            fabric, log_models, cfg, {"agent": state.agent, "encoder": state.encoder, "decoder": state.decoder}
        )
