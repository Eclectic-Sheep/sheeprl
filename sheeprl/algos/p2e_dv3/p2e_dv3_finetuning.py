import copy
import os
import pathlib
import warnings
from typing import Any, Dict

import gymnasium as gym
import hydra
import mlflow
import numpy as np
import torch
from lightning.fabric import Fabric
from mlflow.models.model import ModelInfo
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import BatchSampler
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_v3.agent import PlayerDV3
from sheeprl.algos.dreamer_v3.dreamer_v3 import train
from sheeprl.algos.dreamer_v3.utils import Moments, test
from sheeprl.algos.p2e_dv3.agent import build_agent
from sheeprl.data.buffers import AsyncReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import polynomial_decay, register_model, unwrap_fabric


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any], exploration_cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    ckpt_path = pathlib.Path(cfg.checkpoint.exploration_ckpt_path)
    resume_from_checkpoint = cfg.checkpoint.resume_from is not None

    # Finetuning that was interrupted for some reason
    if resume_from_checkpoint:
        state = fabric.load(pathlib.Path(cfg.checkpoint.resume_from))
        cfg.algo.per_rank_batch_size = state["batch_size"] // world_size
    else:
        state = fabric.load(ckpt_path)

    # All the models must be equal to the ones of the exploration phase
    cfg.algo.gamma = exploration_cfg.algo.gamma
    cfg.algo.lmbda = exploration_cfg.algo.lmbda
    cfg.algo.horizon = exploration_cfg.algo.horizon
    cfg.algo.layer_norm = exploration_cfg.algo.layer_norm
    cfg.algo.dense_units = exploration_cfg.algo.dense_units
    cfg.algo.mlp_layers = exploration_cfg.algo.mlp_layers
    cfg.algo.dense_act = exploration_cfg.algo.dense_act
    cfg.algo.cnn_act = exploration_cfg.algo.cnn_act
    cfg.algo.unimix = exploration_cfg.algo.unimix
    cfg.algo.hafner_initialization = exploration_cfg.algo.hafner_initialization
    cfg.algo.world_model = exploration_cfg.algo.world_model
    cfg.algo.actor = exploration_cfg.algo.actor
    cfg.algo.critic = exploration_cfg.algo.critic
    # Rewards must be clipped in the same way as during exploration
    cfg.env.clip_rewards = exploration_cfg.env.clip_rewards
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
    cfg.env.frame_stack = 1

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
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    clip_rewards_fn = lambda r: torch.tanh(r) if cfg.env.clip_rewards else r
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    if (
        len(set(cfg.algo.cnn_keys.encoder).intersection(set(cfg.algo.cnn_keys.decoder))) == 0
        and len(set(cfg.algo.mlp_keys.encoder).intersection(set(cfg.algo.mlp_keys.decoder))) == 0
    ):
        raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjointed")
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

    (
        world_model,
        _,
        actor_task,
        critic_task,
        target_critic_task,
        actor_exploration,
        _,
    ) = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"],
        None,
        state["actor_task"],
        state["critic_task"],
        state["target_critic_task"],
        state["actor_exploration"],
    )

    # initialize the ensembles with different seeds to be sure they have different weights
    player = PlayerDV3(
        world_model.encoder.module,
        world_model.rssm,
        actor_exploration.module if cfg.algo.player.actor_type == "exploration" else actor_task.module,
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        fabric.device,
        discrete_size=cfg.algo.world_model.discrete_size,
        actor_type=cfg.algo.player.actor_type,
    )

    # Optimizers
    world_optimizer = hydra.utils.instantiate(cfg.algo.world_model.optimizer, params=world_model.parameters())
    actor_task_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor_task.parameters())
    critic_task_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic_task.parameters())
    world_optimizer.load_state_dict(state["world_optimizer"])
    actor_task_optimizer.load_state_dict(state["actor_task_optimizer"])
    critic_task_optimizer.load_state_dict(state["critic_task_optimizer"])
    world_optimizer, actor_task_optimizer, critic_task_optimizer = fabric.setup_optimizers(
        world_optimizer, actor_task_optimizer, critic_task_optimizer
    )

    moments_task = Moments(
        fabric,
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments_task.load_state_dict(state["moments_task"])

    local_vars = locals()

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator).to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * world_size) if not cfg.dry_run else 4
    rb = AsyncReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        device="cpu",
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        sequential=True,
    )
    if resume_from_checkpoint or (cfg.buffer.load_from_exploration and exploration_cfg.buffer.checkpoint):
        if isinstance(state["rb"], list) and world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], AsyncReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {world_size} processes are instantiated")
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device="cpu")
    expl_decay_steps = state["expl_decay_steps"] if resume_from_checkpoint else 0

    # Global variables
    train_step = 0
    last_train = 0
    start_step = state["update"] // fabric.world_size if resume_from_checkpoint else 1
    policy_step = state["update"] * cfg.env.num_envs if resume_from_checkpoint else 0
    last_log = state["last_log"] if resume_from_checkpoint else 0
    last_checkpoint = state["last_checkpoint"] if resume_from_checkpoint else 0
    policy_steps_per_update = int(cfg.env.num_envs * fabric.world_size)
    updates_before_training = cfg.algo.train_every // policy_steps_per_update
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if resume_from_checkpoint and not cfg.buffer.checkpoint:
        learning_starts += start_step
    max_step_expl_decay = cfg.algo.actor.max_step_expl_decay // (cfg.algo.per_rank_gradient_steps * fabric.world_size)
    if resume_from_checkpoint:
        actor_task.expl_amount = polynomial_decay(
            expl_decay_steps,
            initial=cfg.algo.actor.expl_amount,
            final=cfg.algo.actor.expl_min,
            max_decay_steps=max_step_expl_decay,
        )
        actor_exploration.expl_amount = polynomial_decay(
            expl_decay_steps,
            initial=cfg.algo.actor.expl_amount,
            final=cfg.algo.actor.expl_min,
            max_decay_steps=max_step_expl_decay,
        )

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
    o = envs.reset(seed=cfg.seed)[0]
    obs = {k: torch.from_numpy(v).view(cfg.env.num_envs, *v.shape[1:]) for k, v in o.items() if k.startswith("mask")}
    for k in obs_keys:
        torch_obs = torch.from_numpy(o[k]).view(cfg.env.num_envs, *o[k].shape[1:])
        if k in cfg.algo.mlp_keys.encoder:
            # Images stay uint8 to save space
            torch_obs = torch_obs.float()
        step_data[k] = torch_obs
        obs[k] = torch_obs
    step_data["dones"] = torch.zeros(cfg.env.num_envs, 1).float()
    step_data["rewards"] = torch.zeros(cfg.env.num_envs, 1).float()
    step_data["is_first"] = torch.ones_like(step_data["dones"]).float()
    player.init_states()

    per_rank_gradient_steps = 0
    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * world_size

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
            with torch.no_grad():
                preprocessed_obs = {}
                for k, v in obs.items():
                    if k in cfg.algo.cnn_keys.encoder:
                        preprocessed_obs[k] = v[None, ...].to(device) / 255.0
                    else:
                        preprocessed_obs[k] = v[None, ...].to(device)
                mask = {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
                if len(mask) == 0:
                    mask = None
                real_actions = actions = player.get_exploration_action(preprocessed_obs, mask)
                actions = torch.cat(actions, -1).cpu().numpy()
                if is_continuous:
                    real_actions = torch.cat(real_actions, dim=-1).cpu().numpy()
                else:
                    real_actions = np.array([real_act.cpu().argmax(dim=-1).numpy() for real_act in real_actions])

            step_data["actions"] = torch.from_numpy(actions).view(cfg.env.num_envs, -1).float()
            rb.add(step_data[None, ...])

            o, rewards, dones, truncated, infos = envs.step(real_actions.reshape(envs.action_space.shape))
            dones = np.logical_or(dones, truncated)

        step_data["is_first"] = torch.zeros_like(step_data["dones"])
        if "restart_on_exception" in infos:
            for i, agent_roe in enumerate(infos["restart_on_exception"]):
                if agent_roe and not dones[i]:
                    last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                    rb.buffer[i]["dones"][last_inserted_idx] = torch.ones_like(rb.buffer[i]["dones"][last_inserted_idx])
                    rb.buffer[i]["is_first"][last_inserted_idx] = torch.zeros_like(
                        rb.buffer[i]["is_first"][last_inserted_idx]
                    )
                    step_data["is_first"][i] = torch.ones_like(step_data["is_first"][i])

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

        next_obs: Dict[str, Tensor] = {
            k: torch.from_numpy(v).view(cfg.env.num_envs, *v.shape[1:]) for k, v in o.items() if k.startswith("mask")
        }
        for k in real_next_obs.keys():  # [N_envs, N_obs]
            if k in obs_keys:
                next_obs[k] = torch.from_numpy(o[k]).view(cfg.env.num_envs, *o[k].shape[1:])
                step_data[k] = next_obs[k]
                if k in cfg.algo.mlp_keys.encoder:
                    next_obs[k] = next_obs[k].float()
                    step_data[k] = step_data[k].float()

        # next_obs becomes the new obs
        obs = next_obs

        rewards = torch.from_numpy(rewards).view(cfg.env.num_envs, -1).float()
        dones = torch.from_numpy(dones).view(cfg.env.num_envs, -1).float()
        step_data["dones"] = dones
        step_data["rewards"] = clip_rewards_fn(rewards)

        dones_idxes = dones.nonzero(as_tuple=True)[0].tolist()
        reset_envs = len(dones_idxes)
        if reset_envs > 0:
            reset_data = TensorDict({}, batch_size=[reset_envs], device="cpu")
            for k in real_next_obs.keys():
                if k in obs_keys:
                    reset_data[k] = real_next_obs[k][dones_idxes]
                    if k in cfg.algo.mlp_keys.encoder:
                        reset_data[k] = reset_data[k].float()
            reset_data["dones"] = torch.ones(reset_envs, 1).float()
            reset_data["actions"] = torch.zeros(reset_envs, np.sum(actions_dim)).float()
            reset_data["rewards"] = step_data["rewards"][dones_idxes].float()
            reset_data["is_first"] = torch.zeros_like(reset_data["dones"]).float()
            rb.add(reset_data[None, ...], dones_idxes)

            # Reset already inserted step data
            step_data["rewards"][dones_idxes] = torch.zeros_like(reset_data["rewards"]).float()
            step_data["dones"][dones_idxes] = torch.zeros_like(step_data["dones"][dones_idxes]).float()
            step_data["is_first"][dones_idxes] = torch.ones_like(step_data["is_first"][dones_idxes]).float()
            player.init_states(dones_idxes)

        updates_before_training -= 1

        # Train the agent
        if update >= learning_starts and updates_before_training <= 0:
            if player.actor_type == "exploration":
                player.actor = actor_task.module
                player.actor_type = "task"
            local_data = rb.sample(
                cfg.algo.per_rank_batch_size,
                sequence_length=cfg.algo.per_rank_sequence_length,
                n_samples=cfg.algo.per_rank_pretrain_steps
                if update == learning_starts
                else cfg.algo.per_rank_gradient_steps,
            ).to(device)
            distributed_sampler = BatchSampler(range(local_data.shape[0]), batch_size=1, drop_last=False)
            # Start training
            with timer("Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute)):
                for i in distributed_sampler:
                    tau = 1 if per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                    if per_rank_gradient_steps % cfg.algo.critic.target_network_update_freq == 0:
                        for cp, tcp in zip(critic_task.module.parameters(), target_critic_task.parameters()):
                            tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                    train(
                        fabric,
                        world_model,
                        actor_task,
                        critic_task,
                        target_critic_task,
                        world_optimizer,
                        actor_task_optimizer,
                        critic_task_optimizer,
                        local_data[i].view(cfg.algo.per_rank_sequence_length, cfg.algo.per_rank_batch_size),
                        aggregator,
                        cfg,
                        is_continuous=is_continuous,
                        actions_dim=actions_dim,
                        moments=moments_task,
                    )
                train_step += world_size
            updates_before_training = cfg.algo.train_every // policy_steps_per_update
            if cfg.algo.actor.expl_decay:
                expl_decay_steps += 1
                actor_task.expl_amount = polynomial_decay(
                    expl_decay_steps,
                    initial=cfg.algo.actor.expl_amount,
                    final=cfg.algo.actor.expl_min,
                    max_decay_steps=max_step_expl_decay,
                )
                actor_exploration.expl_amount = polynomial_decay(
                    expl_decay_steps,
                    initial=cfg.algo.actor.expl_amount,
                    final=cfg.algo.actor.expl_min,
                    max_decay_steps=max_step_expl_decay,
                )
            if aggregator and not aggregator.disabled:
                aggregator.update("Params/exploration_amount_task", actor_task.expl_amount)
                aggregator.update("Params/exploration_amount_exploration", actor_exploration.expl_amount)

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or update == num_updates):
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

        # Checkpoint Model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            update == num_updates and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "world_model": world_model.state_dict(),
                "actor_task": actor_task.state_dict(),
                "critic_task": critic_task.state_dict(),
                "target_critic_task": target_critic_task.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_task_optimizer": actor_task_optimizer.state_dict(),
                "critic_task_optimizer": critic_task_optimizer.state_dict(),
                "expl_decay_steps": expl_decay_steps,
                "update": update * world_size,
                "batch_size": cfg.algo.per_rank_batch_size * world_size,
                "actor_exploration": actor_exploration.state_dict(),
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
                "moments_task": moments_task.state_dict(),
            }
            ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    envs.close()
    # task test few-shot
    if fabric.is_global_zero:
        player.actor = actor_task.module
        player.actor_type = "task"
        test(player, fabric, cfg, log_dir, "few-shot")

    if not cfg.model_manager.disabled and fabric.is_global_zero:

        def log_models(
            run_id: str, experiment_id: str | None = None, run_name: str | None = None
        ) -> Dict[str, ModelInfo]:
            with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=True) as _:
                model_info = {}
                unwrapped_models = {}
                models_keys = set(cfg.model_manager.models.keys())
                for k in models_keys:
                    if "exploration" not in k and k != "ensembles":
                        unwrapped_models[k] = unwrap_fabric(local_vars[k])
                        model_info[k] = mlflow.pytorch.log_model(unwrapped_models[k], artifact_path=k)
                    else:
                        cfg.model_manager.models.pop(k, None)
                mlflow.log_dict(cfg, "config.json")
            return model_info

        register_model(fabric, log_models, cfg)
