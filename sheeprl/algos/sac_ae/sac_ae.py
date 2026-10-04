from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler

from sheeprl.algos.sac.loss import critic_loss, policy_loss
from sheeprl.algos.sac_ae.agent import SACAEAgent, build_agent, tie_actor_optimizer
from sheeprl.algos.sac_ae.loss import entropy_loss
from sheeprl.algos.sac_ae.utils import prepare_obs, preprocess_obs, test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.models.models import MultiDecoder, MultiEncoder
from sheeprl.utils.compile import compiled, mark_gradient_step
from sheeprl.utils.env import get_episode_stats, get_vector_env_cls, make_env
from sheeprl.utils.fabric import autocast_cache_scope, update
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import phase_timer, timer, training_timer
from sheeprl.utils.utils import off_policy_schedule, save_configs

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
    aggregator: MetricAggregator | None,
    cumulative_per_rank_gradient_steps: int,
    cfg: Dict[str, Any],
):
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
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/value_loss", qf_loss)

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

        if aggregator and not aggregator.disabled:
            aggregator.update("Loss/policy_loss", actor_loss)
            aggregator.update("Loss/alpha_loss", alpha_loss)

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
        if aggregator and not aggregator.disabled:
            aggregator.update("Loss/reconstruction_loss", reconstruction_loss)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    if "minedojo" in cfg.env.wrapper._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by SAC-AE agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from, weights_only=False)

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    vectorized_env = get_vector_env_cls(cfg.env.sync_env)
    envs = vectorized_env(
        [
            make_env(
                cfg,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank * cfg.env.num_envs,
                log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    # Seed the random actions played before the training starts
    envs.action_space.seed(cfg.seed + rank)
    observation_space = envs.single_observation_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise RuntimeError(
            f"Unexpected action space, should be of type continuous (of type Box), got: {observation_space}"
        )

    if (
        len(set(cfg.algo.cnn_keys.encoder).intersection(set(cfg.algo.cnn_keys.decoder))) == 0
        and len(set(cfg.algo.mlp_keys.encoder).intersection(set(cfg.algo.mlp_keys.decoder))) == 0
    ):
        raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjoint")
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
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    # Define the agent and the optimizer and setup them with Fabric
    agent, encoder, decoder, player = build_agent(
        fabric,
        cfg,
        observation_space,
        envs.single_action_space,
        state["agent"] if cfg.checkpoint.resume_from else None,
        state["encoder"] if cfg.checkpoint.resume_from else None,
        state["decoder"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers
    qf_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer,
        params=agent.critic.parameters(),
        _convert_="all",
    )
    actor_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer,
        params=agent.actor.parameters(),
        _convert_="all",
    )
    alpha_optimizer = hydra.utils.instantiate(
        cfg.algo.alpha.optimizer,
        params=[agent.log_alpha],
        _convert_="all",
    )
    encoder_optimizer = hydra.utils.instantiate(
        cfg.algo.encoder.optimizer,
        params=encoder.parameters(),
        _convert_="all",
    )
    decoder_optimizer = hydra.utils.instantiate(
        cfg.algo.decoder.optimizer,
        params=decoder.parameters(),
        _convert_="all",
    )

    if cfg.checkpoint.resume_from:
        qf_optimizer.load_state_dict(state["qf_optimizer"])
        actor_optimizer.load_state_dict(tie_actor_optimizer(state["actor_optimizer"], state["agent"]))
        alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        encoder_optimizer.load_state_dict(state["encoder_optimizer"])
        decoder_optimizer.load_state_dict(state["decoder_optimizer"])

    qf_optimizer, actor_optimizer, alpha_optimizer, encoder_optimizer, decoder_optimizer = fabric.setup_optimizers(
        qf_optimizer, actor_optimizer, alpha_optimizer, encoder_optimizer, decoder_optimizer
    )

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        device=fabric.device if cfg.buffer.memmap else "cpu",
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
        seed=cfg.seed + rank,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], ReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")

    # Global variables
    last_train = 0
    train_step = 0
    start_iter = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["iter_num"] // fabric.world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    # The policy step since which the interaction is timed: a resumed run times only its own steps, not the ones
    # played after the last log of the run it resumes
    last_timed_step = policy_step
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
    total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
    # Random actions in the iterations up to `learning_starts`, training from `train_starts`
    learning_starts, train_starts, pretrain_steps, total_iters, ratio = off_policy_schedule(
        cfg,
        state if cfg.checkpoint.resume_from else None,
        start_iter,
        total_iters,
        policy_steps_per_iter,
        fabric.world_size,
    )

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )
    if cfg.checkpoint.every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )

    # Get the first environment observation and start the optimization
    step_data = {}
    obs = envs.reset(seed=cfg.seed + rank * cfg.env.num_envs)[0]  # [N_envs, N_obs]
    for k in obs_keys:
        if k in cfg.algo.cnn_keys.encoder:
            obs[k] = obs[k].reshape(cfg.env.num_envs, -1, *obs[k].shape[-2:])

    per_rank_gradient_steps = 0
    # The gradient steps of every process from the start of the run, also in the run it resumes (the older
    # checkpoints don't have them)
    cumulative_per_rank_gradient_steps = state.get("per_rank_gradient_steps", 0) if cfg.checkpoint.resume_from else 0
    for iter_num in range(start_iter, total_iters + 1):
        policy_step += policy_steps_per_iter

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with phase_timer("Time/env_interaction_time"):
            if iter_num <= learning_starts:
                actions = envs.action_space.sample()
            else:
                with torch.inference_mode():
                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                    actions = player(torch_obs).cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions.reshape(envs.action_space.shape))

        if cfg.metric.log_level > 0:
            for i, ep_rew, ep_len in get_episode_stats(infos):
                if aggregator and not aggregator.disabled:
                    aggregator.update("Rewards/rew_avg", ep_rew)
                    aggregator.update("Game/ep_len_avg", ep_len)
                fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew}")

        # Save the real next observation
        real_next_obs = copy.deepcopy(next_obs)
        if "final_obs" in infos:
            for idx, final_obs in enumerate(infos["final_obs"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v

        for k in real_next_obs.keys():
            if k in cfg.algo.cnn_keys.encoder:
                next_obs[k] = next_obs[k].reshape(cfg.env.num_envs, -1, *next_obs[k].shape[-2:])
            step_data[k] = obs[k][np.newaxis]

            if not cfg.buffer.sample_next_obs:
                step_data[f"next_{k}"] = real_next_obs[k][np.newaxis]
                if k in cfg.algo.cnn_keys.encoder:
                    step_data[f"next_{k}"] = step_data[f"next_{k}"].reshape(
                        1, cfg.env.num_envs, -1, *step_data[f"next_{k}"].shape[-2:]
                    )

        step_data["terminated"] = terminated.reshape(1, cfg.env.num_envs, -1).astype(np.float32)
        step_data["truncated"] = truncated.reshape(1, cfg.env.num_envs, -1).astype(np.float32)
        step_data["actions"] = actions.reshape(1, cfg.env.num_envs, -1).astype(np.float32)
        step_data["rewards"] = rewards.reshape(1, cfg.env.num_envs, -1).astype(np.float32)
        rb.add(step_data, validate_args=cfg.buffer.validate_args)

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        if iter_num >= train_starts:
            ratio_steps = policy_step - (train_starts - 1) * policy_steps_per_iter
            per_rank_gradient_steps = ratio(ratio_steps / world_size)
            if iter_num == train_starts:
                # The pretraining on the filled buffer (the `pretrain` of DreamerV1 and DreamerV2)
                per_rank_gradient_steps += pretrain_steps
            if per_rank_gradient_steps > 0:
                with training_timer(fabric.device):
                    # The batches of the gradient steps are sampled `MAX_SAMPLED_BATCHES` at a time: the images of all
                    # the ones of the first training (with the pretraining) don't fit in the memory
                    for first in range(0, per_rank_gradient_steps, MAX_SAMPLED_BATCHES):
                        n_steps = min(MAX_SAMPLED_BATCHES, per_rank_gradient_steps - first)
                        sample = rb.sample_tensors(
                            n_steps * cfg.algo.per_rank_batch_size,
                            sample_next_obs=cfg.buffer.sample_next_obs,
                            from_numpy=cfg.buffer.from_numpy,
                            online=cfg.buffer.online,
                        )  # [1, G*B]
                        # [World, 1, G*B] with several processes, [1, G*B] with one (no dimension of the processes)
                        gathered_data: Dict[str, torch.Tensor] = fabric.all_gather(sample)
                        for k, v in gathered_data.items():
                            gathered_data[k] = v.reshape(-1, *sample[k].shape[2:]).float()  # [G*B*World]
                        len_data = len(gathered_data[next(iter(gathered_data.keys()))])
                        if fabric.world_size > 1:
                            dist_sampler: DistributedSampler = DistributedSampler(
                                range(len_data),
                                num_replicas=fabric.world_size,
                                rank=fabric.global_rank,
                                shuffle=True,
                                seed=cfg.seed,
                                drop_last=False,
                            )
                            sampler: BatchSampler = BatchSampler(
                                sampler=dist_sampler, batch_size=cfg.algo.per_rank_batch_size, drop_last=False
                            )
                        else:
                            sampler = BatchSampler(
                                sampler=range(len_data), batch_size=cfg.algo.per_rank_batch_size, drop_last=False
                            )
                        for batch_idxes in sampler:
                            train(
                                fabric,
                                agent,
                                encoder,
                                decoder,
                                actor_optimizer,
                                qf_optimizer,
                                alpha_optimizer,
                                encoder_optimizer,
                                decoder_optimizer,
                                {k: v[batch_idxes] for k, v in gathered_data.items()},
                                aggregator,
                                cumulative_per_rank_gradient_steps,
                                cfg,
                            )
                            cumulative_per_rank_gradient_steps += 1
                    # The gradient steps of all the processes
                    train_step += world_size * per_rank_gradient_steps

        # Log metrics
        if cfg.metric.log_level and (policy_step - last_log >= cfg.metric.log_every or iter_num == total_iters):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
            )

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics and timer_metrics["Time/train_time"] > 0:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_timed_step) * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step
            last_timed_step = policy_step
            last_train = train_step

        # Checkpoint model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            iter_num == total_iters and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "agent": agent.state_dict(),
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "ratio": ratio.state_dict(),
                "per_rank_gradient_steps": cumulative_per_rank_gradient_steps,
                "iter_num": iter_num * fabric.world_size,
                "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir, policy_step=policy_step)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac_ae.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"agent": agent, "encoder": encoder, "decoder": decoder}
        register_model(fabric, log_models, cfg, models_to_log)
