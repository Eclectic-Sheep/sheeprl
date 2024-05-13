import copy
import os
import warnings
from datetime import timedelta
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from torch.distributed.algorithms.join import Join
from torch.utils.data import BatchSampler, RandomSampler
from torchmetrics import SumMetric

from sheeprl.algos.ppo.agent import build_agent
from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.utils import normalize_obs, prepare_obs, test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.logger import get_log_dir
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay, save_configs


@torch.inference_mode()
def player(
    fabric: Fabric,
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    cfg: Dict[str, Any],
):
    # Initialize Fabric player-only
    fabric_player = get_single_device_fabric(fabric)
    log_dir = get_log_dir(fabric_player, cfg.root_dir, cfg.run_name, False)
    device = fabric_player.device

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric_player.load(cfg.checkpoint.resume_from)

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg,
                cfg.seed + i,
                0,
                log_dir,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    observation_space = envs.single_observation_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: "
            "`cnn_keys.encoder=[rgb]` or `mlp_keys.encoder=[state]`"
        )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(envs.single_action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        envs.single_action_space.shape
        if is_continuous
        else (envs.single_action_space.nvec.tolist() if is_multidiscrete else [envs.single_action_space.n])
    )

    # Create the actor and critic models
    agent_args = {
        "actions_dim": actions_dim,
        "obs_space": observation_space,
        "encoder_cfg": cfg.algo.encoder,
        "actor_cfg": cfg.algo.actor,
        "critic_cfg": cfg.algo.critic,
        "cnn_keys": cfg.algo.cnn_keys.encoder,
        "mlp_keys": cfg.algo.mlp_keys.encoder,
        "screen_size": cfg.env.screen_size,
        "distribution_cfg": cfg.distribution,
        "is_continuous": is_continuous,
    }
    _, agent = build_agent(
        fabric_player,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        cfg=cfg,
        obs_space=observation_space,
        agent_state=state["agent"] if cfg.checkpoint.resume_from else None,
    )
    del _

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)
    # Send (possibly updated, by the make_env method for example) cfg to the trainers
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // (world_collective.world_size - 1)
    cfg.checkpoint.log_dir = log_dir
    world_collective.broadcast_object_list([cfg], src=0)

    # Broadcast the parameters needed to the trainers to instantiate the PPOAgent
    world_collective.broadcast_object_list([agent_args], src=0)

    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(list(agent.parameters())),
        device=device,
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, list(agent.parameters()))

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    rb = ReplayBuffer(
        cfg.buffer.size,
        cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        obs_keys=obs_keys,
    )

    # Global variables
    start_iter = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        state["iter_num"] + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["iter_num"] * cfg.env.num_envs * cfg.algo.rollout_steps if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_iter = int(cfg.env.num_envs * cfg.algo.rollout_steps)
    total_iters = cfg.algo.total_steps // policy_steps_per_iter if not cfg.dry_run else 1

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
    if policy_steps_per_iter < world_collective.world_size - 1:
        raise RuntimeError(
            "The number of trainers ({}) is greater than the available collected data ({}). ".format(
                world_collective.world_size - 1, policy_steps_per_iter
            )
            + "Consider to lower the number of trainers at least to the size of available collected data"
        )
    chunks_sizes = [
        len(chunk) for chunk in torch.tensor_split(torch.arange(policy_steps_per_iter), world_collective.world_size - 1)
    ]

    # Broadcast total_iters to all the world
    update_t = torch.as_tensor([total_iters], device=device, dtype=torch.float32)
    world_collective.broadcast(update_t, src=0)

    # Get the first environment observation and start the optimization
    step_data = {}
    next_obs = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    for k in obs_keys:
        if k in cfg.algo.cnn_keys.encoder:
            next_obs[k] = next_obs[k].reshape(cfg.env.num_envs, -1, *next_obs[k].shape[-2:])
        step_data[k] = next_obs[k][np.newaxis]

    params = {"iter_num": start_iter, "last_log": last_log, "last_checkpoint": last_checkpoint}
    world_collective.scatter_object_list([None], [params] * world_collective.world_size, src=0)
    for _ in range(start_iter, total_iters + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs

            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                # Sample an action given the observation received by the environment
                torch_obs = prepare_obs(fabric, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                actions, logprobs, values = agent(torch_obs)
                if is_continuous:
                    real_actions = torch.stack(actions, -1).cpu().numpy()
                else:
                    real_actions = torch.stack([act.argmax(dim=-1) for act in actions], dim=-1).cpu().numpy()
                actions = torch.cat(actions, -1).cpu().numpy()

                # Single environment step
                obs, rewards, terminated, truncated, info = envs.step(real_actions.reshape(envs.action_space.shape))
                truncated_envs = np.nonzero(truncated)[0]
                if len(truncated_envs) > 0:
                    real_next_obs = {
                        k: torch.empty(
                            len(truncated_envs),
                            *observation_space[k].shape,
                            dtype=torch.float32,
                            device=device,
                        )
                        for k in obs_keys
                    }
                    for i, truncated_env in enumerate(truncated_envs):
                        for k, v in info["final_observation"][truncated_env].items():
                            torch_v = torch.as_tensor(v, dtype=torch.float32, device=device)
                            if k in cfg.algo.cnn_keys.encoder:
                                torch_v = torch_v.view(-1, *v.shape[-2:])
                                torch_v = torch_v / 255.0 - 0.5
                            real_next_obs[k][i] = torch_v
                    vals = agent.get_values(real_next_obs).cpu().numpy()
                    rewards[truncated_envs] += cfg.algo.gamma * vals.reshape(rewards[truncated_envs].shape)
                dones = np.logical_or(terminated, truncated).reshape(cfg.env.num_envs, -1).astype(np.uint8)
                rewards = rewards.reshape(cfg.env.num_envs, -1)

            # Update the step data
            step_data["dones"] = dones[np.newaxis]
            step_data["values"] = values.cpu().numpy()[np.newaxis]
            step_data["actions"] = actions.reshape(1, cfg.env.num_envs, -1)
            step_data["logprobs"] = logprobs.cpu().numpy()[np.newaxis]
            step_data["rewards"] = rewards[np.newaxis]
            if cfg.buffer.memmap:
                step_data["returns"] = np.zeros_like(rewards, shape=(1, *rewards.shape))
                step_data["advantages"] = np.zeros_like(rewards, shape=(1, *rewards.shape))

            # Append data to buffer
            rb.add(step_data, validate_args=cfg.buffer.validate_args)

            # Update the observation and dones
            next_obs = {}
            for k in obs_keys:
                _obs = obs[k]
                if k in cfg.algo.cnn_keys.encoder:
                    _obs = _obs.reshape(cfg.env.num_envs, -1, *_obs.shape[-2:])
                step_data[k] = _obs[np.newaxis]
                next_obs[k] = _obs

            if cfg.metric.log_level > 0 and "final_info" in info:
                for i, agent_ep_info in enumerate(info["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        if aggregator and not aggregator.disabled:
                            aggregator.update("Rewards/rew_avg", ep_rew)
                            aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Transform the data into PyTorch Tensors
        local_data = rb.to_tensor(dtype=None, device=device, from_numpy=cfg.buffer.from_numpy)

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
        next_values = agent.get_values(torch_obs)
        returns, advantages = gae(
            local_data["rewards"].to(torch.float64),
            local_data["values"],
            local_data["dones"],
            next_values,
            cfg.algo.rollout_steps,
            cfg.algo.gamma,
            cfg.algo.gae_lambda,
        )

        # Add returns and advantages to the buffer
        local_data["returns"] = returns.float()
        local_data["advantages"] = advantages.float()
        local_data["rewards"] = local_data["rewards"].float()

        # Send data to the training agents
        # Split data in an even way, when possible
        perm = torch.randperm(local_data[next(iter(local_data.keys()))].shape[0], device=device)
        # chunks = {k1: [k1_chunk_1, k1_chunk_2, ...], k2: [k2_chunk_1, k2_chunk_2, ...]}
        chunks = {k: v[perm].flatten(0, 1).split(chunks_sizes) for k, v in local_data.items()}
        # chunks = [{k1: k1_chunk_1, k2: k2_chunk_1}, {k1: k1_chunk_2, k2: k2_chunk_2}, ...]
        chunks = [{k: v[i] for k, v in chunks.items()} for i in range(len(chunks[next(iter(chunks.keys()))]))]
        world_collective.scatter_object_list([None], [None] + chunks, src=0)

        # Wait the trainers to finish
        player_trainer_collective.broadcast(flattened_parameters, src=1)

        # Convert back the parameters
        torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, list(agent.parameters()))

        if cfg.metric.log_level > 0 and policy_step - last_log >= cfg.metric.log_every:
            # Gather metrics from the trainers
            metrics = [None]
            player_trainer_collective.broadcast_object_list(metrics, src=1)
            metrics = metrics[0]

            # Log metrics
            fabric.log_dict(metrics, policy_step)
            if aggregator and not aggregator.disabled:
                fabric.log_dict(aggregator.compute(), policy_step)
                aggregator.reset()

            # Sync timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/sps_env_interaction" in timer_metrics and timer_metrics["Time/sps_env_interaction"] > 0:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) * cfg.env.action_repeat) / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step

        # Checkpoint model
        if cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every:
            last_checkpoint = policy_step
            ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_player",
                fabric=fabric,
                player_trainer_collective=player_trainer_collective,
                ckpt_path=ckpt_path,
            )

    world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)

    # Last Checkpoint
    if cfg.checkpoint.save_last:
        ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
        fabric.call(
            "on_checkpoint_player",
            fabric=fabric,
            player_trainer_collective=player_trainer_collective,
            ckpt_path=ckpt_path,
        )

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(agent, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.ppo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"agent": agent}
        register_model(fabric, log_models, cfg, models_to_log)


def trainer(
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    optimization_pg: CollectibleGroup,
    cfg: Dict[str, Any],
):
    global_rank = world_collective.rank
    group_world_size = world_collective.world_size - 1

    # Initialize Fabric
    cfg.fabric.pop("loggers", None)
    cfg.fabric.pop("strategy", None)
    fabric: Fabric = hydra.utils.instantiate(
        cfg.fabric, strategy=DDPStrategy(process_group=optimization_pg), _convert_="all"
    )
    fabric.launch()
    device = fabric.device

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # Receive (possibly updated, by the make_env method for example) cfg from the player
    data = [None]
    world_collective.broadcast_object_list(data, src=0)
    cfg: Dict[str, Any] = data[0]

    # Environment setup
    agent_args = [None]
    world_collective.broadcast_object_list(agent_args, src=0)

    # Define the agent and the optimizer
    agent, _ = build_agent(
        fabric,
        actions_dim=agent_args[0]["actions_dim"],
        is_continuous=agent_args[0]["is_continuous"],
        cfg=cfg,
        obs_space=agent_args[0]["obs_space"],
        agent_state=state["agent"] if cfg.checkpoint.resume_from else None,
    )
    del _
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")

    # Load the state from the checkpoint
    if cfg.checkpoint.resume_from:
        agent.load_state_dict(state["agent"])
        optimizer.load_state_dict(state["optimizer"])

    # Setup agent and optimizer with Fabric
    optimizer = fabric.setup_optimizers(optimizer)

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()),
            src=1,
        )

    # Receive maximum number of updates from the player
    total_iters = torch.zeros(1, device=device)
    world_collective.broadcast(total_iters, src=0)
    total_iters = total_iters.item()

    # Linear learning rate scheduler
    if cfg.algo.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=total_iters, power=1.0)
        if cfg.checkpoint.resume_from:
            scheduler.load_state_dict(state["scheduler"])

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Start training
    last_train = 0
    train_step = 0

    policy_steps_per_iter = cfg.env.num_envs * cfg.algo.rollout_steps
    params = [None]
    world_collective.scatter_object_list(params, [None for _ in range(world_collective.world_size)], src=0)
    params = params[0]
    iter_num = params["iter_num"]
    policy_step = iter_num * policy_steps_per_iter
    last_log = params["last_log"]
    last_checkpoint = params["last_checkpoint"]
    initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)
    initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)
    while True:
        # Wait for data
        data = [None]
        world_collective.scatter_object_list(data, [None for _ in range(world_collective.world_size)], src=0)
        data = data[0]
        if not isinstance(data, dict) and data == -1:
            # Last Checkpoint
            if cfg.checkpoint.save_last:
                state = {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
                    "iter_num": iter_num,
                    "batch_size": cfg.algo.per_rank_batch_size * (world_collective.world_size - 1),
                    "last_log": last_log,
                    "last_checkpoint": last_checkpoint,
                }
                ckpt_path = cfg.checkpoint.log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
                fabric.call(
                    "on_checkpoint_trainer",
                    fabric=fabric,
                    player_trainer_collective=player_trainer_collective,
                    ckpt_path=ckpt_path,
                    state=state,
                )
            return

        train_step += group_world_size

        # Prepare sampler
        indexes = list(range(data[next(iter(data.keys()))].shape[0]))
        sampler = BatchSampler(RandomSampler(indexes), batch_size=cfg.algo.per_rank_batch_size, drop_last=False)

        # Start training
        with timer(
            "Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg
        ):
            # The Join context is needed because there can be the possibility
            # that some ranks receive less data
            with Join(
                [agent.feature_extractor._forward_module, agent.actor._forward_module, agent.critic._forward_module]
            ):
                for _ in range(cfg.algo.update_epochs):
                    for batch_idxes in sampler:
                        batch = {k: data[k][batch_idxes] for k in data.keys()}
                        normalized_obs = normalize_obs(
                            batch, cfg.algo.cnn_keys.encoder, cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder
                        )
                        _, logprobs, entropy, new_values = agent(
                            normalized_obs, torch.split(batch["actions"], agent.actions_dim, dim=-1)
                        )

                        if cfg.algo.normalize_advantages:
                            batch["advantages"] = normalize_tensor(batch["advantages"])

                        # Policy loss
                        pg_loss = policy_loss(
                            logprobs,
                            batch["logprobs"],
                            batch["advantages"],
                            cfg.algo.clip_coef,
                            cfg.algo.loss_reduction,
                        )

                        # Value loss
                        v_loss = value_loss(
                            new_values,
                            batch["values"],
                            batch["returns"],
                            cfg.algo.clip_coef,
                            cfg.algo.clip_vloss,
                            cfg.algo.loss_reduction,
                        )

                        # Entropy loss
                        ent_loss = entropy_loss(entropy, cfg.algo.loss_reduction)

                        # Equation (9) in the paper
                        loss = pg_loss + cfg.algo.vf_coef * v_loss + cfg.algo.ent_coef * ent_loss

                        optimizer.zero_grad(set_to_none=True)
                        fabric.backward(loss)
                        if cfg.algo.max_grad_norm > 0.0:
                            fabric.clip_gradients(agent, optimizer, max_norm=cfg.algo.max_grad_norm)
                        optimizer.step()

                        # Update metrics
                        if aggregator and not aggregator.disabled:
                            aggregator.update("Loss/policy_loss", pg_loss.detach())
                            aggregator.update("Loss/value_loss", v_loss.detach())
                            aggregator.update("Loss/entropy_loss", ent_loss.detach())

        if global_rank == 1:
            player_trainer_collective.broadcast(
                torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()),
                src=1,
            )

        if cfg.metric.log_level > 0 and policy_step - last_log >= cfg.metric.log_every:
            # Sync distributed metrics
            metrics = {}
            if aggregator and not aggregator.disabled:
                metrics.update(aggregator.compute())
                aggregator.reset()

            # Sync distributed timers
            if not timer.disabled:
                timers = timer.compute()
                if "Time/train_time" in timers and timers["Time/train_time"] > 0:
                    metrics.update({"Time/sps_train": (train_step - last_train) / timers["Time/train_time"]})
                timer.reset()

            # Send metrics to the player
            if global_rank == 1:
                if cfg.algo.anneal_lr:
                    metrics["Info/learning_rate"] = scheduler.get_last_lr()[0]
                else:
                    metrics["Info/learning_rate"] = cfg.algo.optimizer.lr
                metrics["Info/clip_coef"] = cfg.algo.clip_coef
                metrics["Info/ent_coef"] = cfg.algo.ent_coef
                player_trainer_collective.broadcast_object_list(
                    [metrics], src=1
                )  # Broadcast metrics: fake send with object list between rank-0 and rank-1

            # Reset counters
            last_log = policy_step
            last_train = train_step

        if cfg.algo.anneal_lr:
            scheduler.step()

        if cfg.algo.anneal_clip_coef:
            cfg.algo.clip_coef = polynomial_decay(
                iter_num, initial=initial_clip_coef, final=0.0, max_decay_steps=total_iters, power=1.0
            )

        if cfg.algo.anneal_ent_coef:
            cfg.algo.ent_coef = polynomial_decay(
                iter_num, initial=initial_ent_coef, final=0.0, max_decay_steps=total_iters, power=1.0
            )

        # Checkpoint model on rank-0: send it everything
        if cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every:
            last_checkpoint = policy_step
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
                "iter_num": iter_num,
                "batch_size": cfg.algo.per_rank_batch_size * (world_collective.world_size - 1),
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = cfg.checkpoint.log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_trainer",
                fabric=fabric,
                player_trainer_collective=player_trainer_collective,
                ckpt_path=ckpt_path,
                state=state,
            )
        iter_num += 1
        policy_step += cfg.env.num_envs * cfg.algo.rollout_steps


@register_algorithm(decoupled=True)
def main(fabric: Fabric, cfg: Dict[str, Any]):
    if fabric.world_size == 1:
        raise RuntimeError(
            "Please run the script with the number of devices greater than 1: "
            "`python sheeprl.py exp=ppo_decoupled fabric.devices=2 ...`"
        )

    if "minedojo" in cfg.env.wrapper._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by PPO agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    if cfg.buffer.share_data:
        warnings.warn(
            "You have called the script with `buffer.share_data=True`: "
            "decoupled scripts splits collected data in an almost-even way between the number of trainers"
        )

    world_collective = TorchCollective()
    player_trainer_collective = TorchCollective()
    world_collective.setup(
        backend="nccl" if os.environ.get("LT_ACCELERATOR", None) in ("gpu", "cuda") else "gloo",
        timeout=timedelta(days=1),
    )

    # Create a global group, assigning it to the collective: used by the player to exchange
    # collected experiences with the trainers
    world_collective.create_group(timeout=timedelta(days=1))
    global_rank = world_collective.rank

    # Create a group between rank-0 (player) and rank-1 (trainer), assigning it to the collective:
    # used by rank-1 to send metrics to be tracked by the rank-0 at the end of a training episode
    player_trainer_collective.create_group(ranks=[0, 1], timeout=timedelta(days=1))

    # Create a new group, without assigning it to the collective: in this way the trainers can
    # still communicate with the player through the global group, but they can optimize the agent
    # between themselves
    optimization_pg = world_collective.new_group(
        ranks=list(range(1, world_collective.world_size)), timeout=timedelta(days=1)
    )
    if global_rank == 0:
        player(fabric, world_collective, player_trainer_collective, cfg)
    else:
        trainer(world_collective, player_trainer_collective, optimization_pg, cfg)
