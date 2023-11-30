from __future__ import annotations

import copy
import os
import time
import warnings
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import hydra
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.wrappers import _FabricModule
from mlflow.models.model import ModelInfo
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torchmetrics import SumMetric

from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.algos.sac_ae.agent import SACAEAgent, build_agent
from sheeprl.algos.sac_ae.utils import preprocess_obs, test_sac_ae
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.models.models import MultiDecoder, MultiEncoder
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import register_model, unwrap_fabric


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
    data: TensorDictBase,
    aggregator: MetricAggregator | None,
    update: int,
    cfg: Dict[str, Any],
    policy_steps_per_update: int,
    group: Optional[CollectibleGroup] = None,
):
    critic_target_network_frequency = cfg.algo.critic.target_network_frequency // policy_steps_per_update + 1
    actor_network_frequency = cfg.algo.actor.network_frequency // policy_steps_per_update + 1
    decoder_update_freq = cfg.algo.decoder.update_freq // policy_steps_per_update + 1
    data = data.to(fabric.device)
    normalized_obs = {}
    normalized_next_obs = {}
    for k in cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder:
        if k in cfg.algo.cnn_keys.encoder:
            normalized_obs[k] = data[k] / 255.0
            normalized_next_obs[k] = data[f"next_{k}"] / 255.0
        else:
            normalized_obs[k] = data[k]
            normalized_next_obs[k] = data[f"next_{k}"]

    # Update the soft-critic
    next_target_qf_value = agent.get_next_target_q_values(
        normalized_next_obs, data["rewards"], data["dones"], cfg.algo.gamma
    )
    qf_values = agent.get_q_values(normalized_obs, data["actions"])
    qf_loss = critic_loss(qf_values, next_target_qf_value, agent.num_critics)
    qf_optimizer.zero_grad(set_to_none=True)
    fabric.backward(qf_loss)
    qf_optimizer.step()
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/value_loss", qf_loss)

    # Update the target networks with EMA
    if update % critic_target_network_frequency == 0:
        agent.critic_target_ema()
        agent.critic_encoder_target_ema()

    # Update the actor
    if update % actor_network_frequency == 0:
        actions, logprobs = agent.get_actions_and_log_probs(normalized_obs, detach_encoder_features=True)
        qf_values = agent.get_q_values(normalized_obs, actions, detach_encoder_features=True)
        min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
        actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
        actor_optimizer.zero_grad(set_to_none=True)
        fabric.backward(actor_loss)
        actor_optimizer.step()

        # Update the entropy value
        alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
        alpha_optimizer.zero_grad(set_to_none=True)
        fabric.backward(alpha_loss)
        agent.log_alpha.grad = fabric.all_reduce(agent.log_alpha.grad, group=group)
        alpha_optimizer.step()

        if aggregator and not aggregator.disabled:
            aggregator.update("Loss/policy_loss", actor_loss)
            aggregator.update("Loss/alpha_loss", alpha_loss)

    # Update the decoder
    if update % decoder_update_freq == 0:
        hidden = encoder(normalized_obs)
        reconstruction = decoder(hidden)
        reconstruction_loss = 0
        for k in cfg.algo.cnn_keys.decoder + cfg.algo.mlp_keys.decoder:
            target = preprocess_obs(data[k], bits=5) if k in cfg.algo.cnn_keys.decoder else data[k]
            reconstruction_loss += (
                F.mse_loss(target, reconstruction[k])  # Reconstruction
                + cfg.algo.decoder.l2_lambda * (0.5 * hidden.pow(2).sum(1)).mean()  # L2 penalty on the hidden state
            )
        encoder_optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)
        fabric.backward(reconstruction_loss)
        encoder_optimizer.step()
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
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size

    # These arguments cannot be changed
    cfg.env.screen_size = 64

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
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
    observation_space = envs.single_observation_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

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

    # Define the agent and the optimizer and setup them with Fabric
    agent, encoder, decoder = build_agent(
        fabric,
        cfg,
        observation_space,
        envs.single_action_space,
        state["agent"] if cfg.checkpoint.resume_from else None,
        state["encoder"] if cfg.checkpoint.resume_from else None,
        state["decoder"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers
    qf_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=agent.critic.parameters())
    actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=agent.actor.parameters())
    alpha_optimizer = hydra.utils.instantiate(cfg.algo.alpha.optimizer, params=[agent.log_alpha])
    encoder_optimizer = hydra.utils.instantiate(cfg.algo.encoder.optimizer, params=encoder.parameters())
    decoder_optimizer = hydra.utils.instantiate(cfg.algo.decoder.optimizer, params=decoder.parameters())

    if cfg.checkpoint.resume_from:
        qf_optimizer.load_state_dict(state["qf_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        encoder_optimizer.load_state_dict(state["encoder_optimizer"])
        decoder_optimizer.load_state_dict(state["decoder_optimizer"])

    qf_optimizer, actor_optimizer, alpha_optimizer, encoder_optimizer, decoder_optimizer = fabric.setup_optimizers(
        qf_optimizer, actor_optimizer, alpha_optimizer, encoder_optimizer, decoder_optimizer
    )

    local_vars = locals()

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator).to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        device=fabric.device if cfg.buffer.memmap else "cpu",
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], ReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=fabric.device if cfg.buffer.memmap else "cpu")

    # Global variables
    last_train = 0
    train_step = 0
    start_step = state["update"] // fabric.world_size if cfg.checkpoint.resume_from else 1
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    time.time()
    policy_steps_per_update = int(cfg.env.num_envs * fabric.world_size)
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from and not cfg.buffer.checkpoint:
        learning_starts += start_step

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    # Get the first environment observation and start the optimization
    o = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    obs = {}
    for k in o.keys():
        if k in cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device)
            if k in cfg.algo.cnn_keys.encoder:
                torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
            if k in cfg.algo.mlp_keys.encoder:
                torch_obs = torch_obs.float()
            obs[k] = torch_obs

    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * fabric.world_size

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
            if update < learning_starts:
                actions = envs.action_space.sample()
            else:
                with torch.no_grad():
                    normalized_obs = {k: v / 255 if k in cfg.algo.cnn_keys.encoder else v for k, v in obs.items()}
                    actions, _ = agent.actor.module(normalized_obs)
                    actions = actions.cpu().numpy()
            o, rewards, dones, truncated, infos = envs.step(actions)
            dones = np.logical_or(dones, truncated)

        if cfg.metric.log_level > 0 and "final_info" in infos:
            for i, agent_ep_info in enumerate(infos["final_info"]):
                if agent_ep_info is not None:
                    ep_rew = agent_ep_info["episode"]["r"]
                    ep_len = agent_ep_info["episode"]["l"]
                    if aggregator and not aggregator.disabled:
                        aggregator.update("Rewards/rew_avg", ep_rew)
                        aggregator.update("Game/ep_len_avg", ep_len)
                    fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Save the real next observation
        real_next_obs = copy.deepcopy(o)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v

        next_obs = {}
        for k in real_next_obs.keys():
            next_obs[k] = torch.from_numpy(o[k]).to(fabric.device)
            if k in cfg.algo.cnn_keys.encoder:
                next_obs[k] = next_obs[k].view(cfg.env.num_envs, -1, *next_obs[k].shape[-2:])
            if k in cfg.algo.mlp_keys.encoder:
                next_obs[k] = next_obs[k].float()

            step_data[k] = obs[k]
            if not cfg.buffer.sample_next_obs:
                step_data[f"next_{k}"] = torch.from_numpy(real_next_obs[k]).to(fabric.device)
                if k in cfg.algo.cnn_keys.encoder:
                    step_data[f"next_{k}"] = step_data[f"next_{k}"].view(
                        cfg.env.num_envs, -1, *step_data[f"next_{k}"].shape[-2:]
                    )
                if k in cfg.algo.mlp_keys.encoder:
                    step_data[f"next_{k}"] = step_data[f"next_{k}"].float()
        actions = torch.from_numpy(actions).view(cfg.env.num_envs, -1).float().to(fabric.device)
        rewards = torch.from_numpy(rewards).view(cfg.env.num_envs, -1).float().to(fabric.device)
        dones = torch.from_numpy(dones).view(cfg.env.num_envs, -1).float().to(fabric.device)

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        if update >= learning_starts - 1:
            training_steps = learning_starts if update == learning_starts - 1 else 1

            # We sample one time to reduce the communications between processes
            sample = rb.sample(
                training_steps * cfg.algo.per_rank_gradient_steps * cfg.algo.per_rank_batch_size,
                sample_next_obs=cfg.buffer.sample_next_obs,
            )  # [G*B, 1]
            gathered_data = fabric.all_gather(sample.to_dict())  # [G*B, World, 1]
            gathered_data = make_tensordict(gathered_data).view(-1)  # [G*B*World]
            if fabric.world_size > 1:
                dist_sampler: DistributedSampler = DistributedSampler(
                    range(len(gathered_data)),
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
                    sampler=range(len(gathered_data)), batch_size=cfg.algo.per_rank_batch_size, drop_last=False
                )

            # Start training
            with timer("Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute)):
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
                        gathered_data[batch_idxes],
                        aggregator,
                        update,
                        cfg,
                        policy_steps_per_update,
                    )
                train_step += world_size

        # Log metrics
        if cfg.metric.log_level and (policy_step - last_log >= cfg.metric.log_every or update == num_updates):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            update == num_updates and cfg.checkpoint.save_last
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
                "update": update * fabric.world_size,
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
    if fabric.is_global_zero:
        test_sac_ae(agent.actor.module, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:

        def log_models(
            run_id: str, experiment_id: str | None = None, run_name: str | None = None
        ) -> Dict[str, ModelInfo]:
            with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=True) as _:
                model_info = {}
                unwrapped_models = {}
                for k in cfg.model_manager.models.keys():
                    unwrapped_models[k] = unwrap_fabric(local_vars[k])
                    model_info[k] = mlflow.pytorch.log_model(unwrapped_models[k], artifact_path=k)
                mlflow.log_dict(cfg, "config.json")
            return model_info

        register_model(fabric, log_models, cfg)
