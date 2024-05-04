import copy
import os
import pathlib
import warnings
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_v3.dreamer_v3 import train
from sheeprl.algos.dreamer_v3.utils import Moments, prepare_obs, test
from sheeprl.algos.p2e_dv3.agent import build_agent
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs, unwrap_fabric


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any], exploration_cfg: Dict[str, Any]):
    # Single-device fabric object
    fabric_player = get_single_device_fabric(fabric)

    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    ckpt_path = pathlib.Path(cfg.checkpoint.exploration_ckpt_path)
    resume_from_checkpoint = cfg.checkpoint.resume_from is not None

    # Finetuning that was interrupted for some reason
    if resume_from_checkpoint:
        state = fabric.load(pathlib.Path(cfg.checkpoint.resume_from))
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
    fabric.print(f"Log dir: {log_dir}")

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
    clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
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

    (world_model, _, actor_task, critic_task, target_critic_task, actor_exploration, _, player) = build_agent(
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

    # Optimizers
    world_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
    )
    actor_task_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer, params=actor_task.parameters(), _convert_="all"
    )
    critic_task_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer, params=critic_task.parameters(), _convert_="all"
    )
    world_optimizer.load_state_dict(state["world_optimizer"])
    actor_task_optimizer.load_state_dict(state["actor_task_optimizer"])
    critic_task_optimizer.load_state_dict(state["critic_task_optimizer"])
    world_optimizer, actor_task_optimizer, critic_task_optimizer = fabric.setup_optimizers(
        world_optimizer, actor_task_optimizer, critic_task_optimizer
    )

    moments_task = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments_task.load_state_dict(state["moments_task"])

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * world_size) if not cfg.dry_run else 4
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=SequentialReplayBuffer,
    )
    if resume_from_checkpoint or (cfg.buffer.load_from_exploration and exploration_cfg.buffer.checkpoint):
        if isinstance(state["rb"], list) and world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], EnvIndependentReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {world_size} processes are instantiated")

    # Global variables
    train_step = 0
    last_train = 0
    start_step = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["update"] // fabric.world_size) + 1
        if resume_from_checkpoint
        else 1
    )
    policy_step = state["update"] * cfg.env.num_envs if resume_from_checkpoint else 0
    last_log = state["last_log"] if resume_from_checkpoint else 0
    last_checkpoint = state["last_checkpoint"] if resume_from_checkpoint else 0
    policy_steps_per_update = int(cfg.env.num_envs * fabric.world_size)
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if resume_from_checkpoint:
        cfg.algo.per_rank_batch_size = state["batch_size"] // world_size
        if not cfg.buffer.checkpoint:
            learning_starts += start_step

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

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
    step_data = {}
    obs = envs.reset(seed=cfg.seed)[0]
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["terminated"])
    player.init_states()

    cumulative_per_rank_gradient_steps = 0
    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * world_size

        with torch.inference_mode():
            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                if len(mask) == 0:
                    mask = None
                real_actions = actions = player.get_actions(torch_obs, mask=mask)
                actions = torch.cat(actions, -1).cpu().numpy()
                if is_continuous:
                    real_actions = torch.cat(real_actions, dim=-1).cpu().numpy()
                else:
                    real_actions = (
                        torch.cat([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                    )

                step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
                rb.add(step_data, validate_args=cfg.buffer.validate_args)

                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                dones = np.logical_or(terminated, truncated).astype(np.uint8)

            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:
                for i, agent_roe in enumerate(infos["restart_on_exception"]):
                    if agent_roe and not dones[i]:
                        last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                        rb.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["terminated"][last_inserted_idx]
                        )
                        rb.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                            rb.buffer[i]["truncated"][last_inserted_idx]
                        )
                        rb.buffer[i]["is_first"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["is_first"][last_inserted_idx]
                        )
                        step_data["is_first"][i] = np.ones_like(step_data["is_first"][i])

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
            real_next_obs = copy.deepcopy(next_obs)
            if "final_observation" in infos:
                for idx, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        for k, v in final_obs.items():
                            real_next_obs[k][idx] = v

            for k in obs_keys:
                step_data[k] = next_obs[k][np.newaxis]

            # next_obs becomes the new obs
            obs = next_obs

            rewards = rewards.reshape((1, cfg.env.num_envs, -1))
            step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
            step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
            step_data["rewards"] = clip_rewards_fn(rewards)

            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
                reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

                # Reset already inserted step data
                step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
                player.init_states(dones_idxes)

        # Train the agent
        if update >= learning_starts:
            per_rank_gradient_steps = ratio(policy_step / world_size)
            if per_rank_gradient_steps > 0:
                if player.actor_type != "task":
                    player.actor_type = "task"
                    player.actor = fabric_player.setup_module(unwrap_fabric(actor_task))
                    for agent_p, p in zip(actor_task.parameters(), player.actor.parameters()):
                        p.data = agent_p.data
                local_data = rb.sample_tensors(
                    cfg.algo.per_rank_batch_size,
                    sequence_length=cfg.algo.per_rank_sequence_length,
                    n_samples=per_rank_gradient_steps,
                    dtype=None,
                    device=fabric.device,
                    from_numpy=cfg.buffer.from_numpy,
                )
                # Start training
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    for i in range(per_rank_gradient_steps):
                        if (
                            cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                            == 0
                        ):
                            tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                            for cp, tcp in zip(critic_task.module.parameters(), target_critic_task.parameters()):
                                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                        batch = {k: v[i].float() for k, v in local_data.items()}
                        train(
                            fabric,
                            world_model,
                            actor_task,
                            critic_task,
                            target_critic_task,
                            world_optimizer,
                            actor_task_optimizer,
                            critic_task_optimizer,
                            batch,
                            aggregator,
                            cfg,
                            is_continuous=is_continuous,
                            actions_dim=actions_dim,
                            moments=moments_task,
                        )
                        cumulative_per_rank_gradient_steps += 1
                    train_step += world_size

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or update == num_updates):
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
                "ratio": ratio.state_dict(),
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
    if fabric.is_global_zero and cfg.algo.run_test:
        player.actor_type = "task"
        player.actor = fabric_player.setup_module(unwrap_fabric(actor_task))
        test(player, fabric, cfg, log_dir, "few-shot", greedy=False)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "actor_task": actor_task,
            "critic_task": critic_task,
            "target_critic_task": target_critic_task,
            "moments_task": moments_task,
        }
        register_model(fabric, log_models, cfg, models_to_log)
