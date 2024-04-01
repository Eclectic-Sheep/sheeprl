import copy
import os
import warnings
from datetime import timedelta
from math import prod
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data.sampler import BatchSampler
from torchmetrics import SumMetric

from sheeprl.algos.sac.agent import SACAgent, SACCritic, build_agent
from sheeprl.algos.sac.sac import train
from sheeprl.algos.sac.utils import test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.logger import get_log_dir
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs


@torch.inference_mode()
def player(
    fabric: Fabric, world_collective: TorchCollective, player_trainer_collective: TorchCollective, cfg: Dict[str, Any]
):
    # Initialize Fabric player-only
    fabric_player = get_single_device_fabric(fabric)
    log_dir = get_log_dir(fabric_player, cfg.root_dir, cfg.run_name, False)
    device = fabric_player.device
    rank = fabric.global_rank

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric_player.load(cfg.checkpoint.resume_from)

    if len(cfg.algo.cnn_keys.encoder) > 0:
        warnings.warn("SAC algorithm cannot allow to use images as observations, the CNN keys will be ignored")
        cfg.algo.cnn_keys.encoder = []

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
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the SAC agent")
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.algo.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
    for k in cfg.algo.mlp_keys.encoder:
        if len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the SAC agent. "
                f"Provided environment: {cfg.env.id}"
            )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Send (possibly updated, by the make_env method for example) cfg to the trainers
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
    cfg.checkpoint.log_dir = log_dir
    world_collective.broadcast_object_list([cfg], src=0)

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(action_space.shape)
    obs_dim = sum([prod(observation_space[k].shape) for k in cfg.algo.mlp_keys.encoder])
    _, actor = build_agent(
        fabric_player,
        cfg,
        observation_space,
        action_space,
        state["agent"] if cfg.checkpoint.resume_from else None,
    )
    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()), device=device
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    buffer_size = cfg.buffer.size // cfg.env.num_envs if not cfg.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], ReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(
                "The replay buffer in the configs must be of type "
                f"`sheeprl.data.buffers.ReplayBuffer`, got {type(state['rb'])}."
            )

    # Global variables
    first_info_sent = False
    start_step = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        state["update"] + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs)
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from and not cfg.buffer.checkpoint:
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

    step_data = {}
    # Get the first environment observation and start the optimization
    obs = envs.reset(seed=cfg.seed)[0]
    obs = np.concatenate([obs[k] for k in cfg.algo.mlp_keys.encoder], axis=-1)

    per_rank_gradient_steps = 0
    cumulative_per_rank_gradient_steps = 0
    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
            if update <= learning_starts:
                actions = envs.action_space.sample()
            else:
                # Sample an action given the observation received by the environment
                torch_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions = actor(torch_obs)
                actions = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            next_obs = np.concatenate([next_obs[k] for k in cfg.algo.mlp_keys.encoder], axis=-1)
            rewards = rewards.reshape(cfg.env.num_envs, -1)

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
                    real_next_obs[idx] = np.concatenate(
                        [v for k, v in final_obs.items() if k in cfg.algo.mlp_keys.encoder], axis=-1
                    )

        step_data["terminated"] = terminated.reshape(1, cfg.env.num_envs, -1).astype(np.uint8)
        step_data["truncated"] = truncated.reshape(1, cfg.env.num_envs, -1).astype(np.uint8)
        step_data["actions"] = actions[np.newaxis]
        step_data["observations"] = obs[np.newaxis]
        if not cfg.buffer.sample_next_obs:
            step_data["next_observations"] = real_next_obs[np.newaxis]
        step_data["rewards"] = rewards[np.newaxis]
        rb.add(step_data, validate_args=cfg.buffer.validate_args)

        # next_obs becomes the new obs
        obs = next_obs

        # Send data to the training agents
        if update >= learning_starts:
            per_rank_gradient_steps = ratio(policy_step / (fabric.world_size - 1))
            cumulative_per_rank_gradient_steps += per_rank_gradient_steps
            if per_rank_gradient_steps > 0:
                # Send local info to the trainers
                if not first_info_sent:
                    world_collective.broadcast_object_list(
                        [{"update": update, "last_log": last_log, "last_checkpoint": last_checkpoint}], src=0
                    )
                    first_info_sent = True

                # Sample data to be sent to the trainers
                sample = rb.sample_tensors(
                    batch_size=per_rank_gradient_steps * cfg.algo.per_rank_batch_size * (fabric.world_size - 1),
                    sample_next_obs=cfg.buffer.sample_next_obs,
                    dtype=None,
                    device=device,
                    from_numpy=cfg.buffer.from_numpy,
                )
                # chunks = {k1: [k1_chunk_1, k1_chunk_2, ...], k2: [k2_chunk_1, k2_chunk_2, ...]}
                chunks = {
                    k: v.float().split(per_rank_gradient_steps * cfg.algo.per_rank_batch_size)
                    for k, v in sample.items()
                }
                # chunks = [{k1: k1_chunk_1, k2: k2_chunk_1}, {k1: k1_chunk_2, k2: k2_chunk_2}, ...]
                chunks = [{k: v[i] for k, v in chunks.items()} for i in range(len(chunks[next(iter(chunks.keys()))]))]
                world_collective.scatter_object_list([None], [None] + chunks, src=0)

                # Wait the trainers to finish
                player_trainer_collective.broadcast(flattened_parameters, src=1)

                # Convert back the parameters
                torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

                # Logs trainers-only metrics
                if cfg.metric.log_level > 0 and policy_step - last_log >= cfg.metric.log_every:
                    # Gather metrics from the trainers
                    metrics = [None]
                    player_trainer_collective.broadcast_object_list(metrics, src=1)

                    # Log metrics
                    fabric.log_dict(metrics[0], policy_step)

        # Logs player-only metrics
        if cfg.metric.log_level > 0 and policy_step - last_log >= cfg.metric.log_every:
            if aggregator and not aggregator.disabled:
                fabric.log_dict(aggregator.compute(), policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio",
                cumulative_per_rank_gradient_steps * (fabric.world_size - 1) / policy_step,
                policy_step,
            )

            # Sync timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                fabric.log(
                    "Time/sps_env_interaction",
                    ((policy_step - last_log) * cfg.env.action_repeat) / timer_metrics["Time/env_interaction_time"],
                    policy_step,
                )
                timer.reset()

            # Reset counters
            last_log = policy_step

        # Checkpoint model
        if (
            update >= learning_starts  # otherwise the processes end up deadlocked
            and cfg.checkpoint.every > 0
            and policy_step - last_checkpoint >= cfg.checkpoint.every
        ):
            last_checkpoint = policy_step
            ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_player",
                fabric=fabric,
                player_trainer_collective=player_trainer_collective,
                ckpt_path=ckpt_path,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
                ratio_state_dict=ratio.state_dict(),
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
            replay_buffer=rb if cfg.buffer.checkpoint else None,
            ratio_state_dict=ratio.state_dict(),
        )

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(actor, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac.utils import log_models
        from sheeprl.utils.mlflow import register_model

        critics = [
            SACCritic(observation_dim=obs_dim + act_dim, hidden_size=cfg.algo.critic.hidden_size, num_critics=1)
            for _ in range(cfg.algo.critic.n)
        ]
        target_entropy = -act_dim
        agent = SACAgent(
            actor, critics, target_entropy, alpha=cfg.algo.alpha.alpha, tau=cfg.algo.tau, device=fabric.device
        )
        flattened_parameters = torch.empty_like(
            torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()), device=device
        )
        player_trainer_collective.broadcast(flattened_parameters, src=1)
        torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, agent.parameters())
        register_model(fabric, log_models, cfg, {"agent": agent})


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
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env([make_env(cfg, 0, 0, None)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    agent, _ = build_agent(
        fabric,
        cfg,
        envs.single_observation_space,
        envs.single_action_space,
        state["agent"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers
    qf_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=agent.qfs.parameters(), _convert_="all")
    actor_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer, params=agent.actor.parameters(), _convert_="all"
    )
    alpha_optimizer = hydra.utils.instantiate(cfg.algo.alpha.optimizer, params=[agent.log_alpha], _convert_="all")
    if cfg.checkpoint.resume_from:
        qf_optimizer.load_state_dict(state["qf_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        alpha_optimizer.load_state_dict(state["alpha_optimizer"])
    qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        qf_optimizer, actor_optimizer, alpha_optimizer
    )

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters()), src=1
        )

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Receive data from player regarding the:
    # * update
    # * last_log
    # * last_checkpoint
    data = [None]
    world_collective.broadcast_object_list(data, src=0)
    update = data[0]["update"]
    last_log = data[0]["last_log"]
    last_checkpoint = data[0]["last_checkpoint"]

    # Start training
    train_step = 0
    last_train = 0
    policy_steps_per_update = cfg.env.num_envs
    policy_step = update * policy_steps_per_update
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
                    "qf_optimizer": qf_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "alpha_optimizer": alpha_optimizer.state_dict(),
                    "update": update,
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
            if not cfg.model_manager.disabled:
                player_trainer_collective.broadcast(
                    torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()), src=1
                )
            return
        sampler = BatchSampler(
            range(len(data[next(iter(data.keys()))])), batch_size=cfg.algo.per_rank_batch_size, drop_last=False
        )

        # Start training
        with timer(
            "Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg
        ):
            for batch_idxes in sampler:
                train(
                    fabric,
                    agent,
                    actor_optimizer,
                    qf_optimizer,
                    alpha_optimizer,
                    {k: data[k][batch_idxes] for k in data.keys()},
                    aggregator,
                    update,
                    cfg,
                    policy_steps_per_update,
                    group=optimization_pg,
                )
            train_step += group_world_size

        if global_rank == 1:
            player_trainer_collective.broadcast(
                torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters()), src=1
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
                metrics.update({"Time/sps_train": (train_step - last_train) / timers["Time/train_time"]})
                timer.reset()

            if global_rank == 1:
                player_trainer_collective.broadcast_object_list(
                    [metrics], src=1
                )  # Broadcast metrics: fake send with object list between rank-0 and rank-1

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint model on rank-0: send it everything
        if cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every:
            last_checkpoint = policy_step
            state = {
                "agent": agent.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "update": update,
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

        # Update counters
        update += 1
        policy_step += policy_steps_per_update


@register_algorithm(decoupled=True)
def main(fabric: Fabric, cfg: Dict[str, Any]):
    if fabric.world_size == 1:
        raise RuntimeError(
            "Please run the script with the number of devices greater than 1: "
            "`python sheeprl.py exp=sac_decoupled fabric.devices=2 ...`"
        )

    if "minedojo" in cfg.env.wrapper._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by SAC agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
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
