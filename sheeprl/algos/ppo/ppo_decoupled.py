import copy
import os
import warnings
from datetime import timedelta
from typing import Any, Dict

import gymnasium as gym
import hydra
import mlflow
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from mlflow.models.model import ModelInfo
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase, make_tensordict
from torch.distributed.algorithms.join import Join
from torch.utils.data import BatchSampler, RandomSampler
from torchmetrics import SumMetric

from sheeprl.algos.ppo.agent import PPOAgent
from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.utils import normalize_obs, test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay, register_model, unwrap_fabric


@torch.no_grad()
def player(
    fabric: Fabric, cfg: Dict[str, Any], world_collective: TorchCollective, player_trainer_collective: TorchCollective
):
    # Initialize the fabric object
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name, False)
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)
        cfg.algo.per_rank_batch_size = state["batch_size"] // (world_collective.world_size - 1)

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

    # Send (possibly updated, by the make_env method for example) cfg to the trainers
    cfg.checkpoint.log_dir = log_dir
    world_collective.broadcast_object_list([cfg], src=0)

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
    agent = PPOAgent(**agent_args).to(device)

    local_vars = locals()

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
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator).to(device)

    # Local data
    rb = ReplayBuffer(
        cfg.algo.rollout_steps,
        cfg.env.num_envs,
        device=device,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=device)

    # Global variables
    start_step = state["update"] if cfg.checkpoint.resume_from else 1
    policy_step = (state["update"] - 1) * cfg.env.num_envs * cfg.algo.rollout_steps if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * cfg.algo.rollout_steps)
    num_updates = cfg.algo.total_steps // policy_steps_per_update if not cfg.dry_run else 1

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
    if policy_steps_per_update < world_collective.world_size - 1:
        raise RuntimeError(
            "The number of trainers ({}) is greater than the available collected data ({}). ".format(
                world_collective.world_size - 1, policy_steps_per_update
            )
            + "Consider to lower the number of trainers at least to the size of available collected data"
        )
    chunks_sizes = [
        len(chunk)
        for chunk in torch.tensor_split(torch.arange(policy_steps_per_update), world_collective.world_size - 1)
    ]

    # Broadcast num_updates to all the world
    update_t = torch.tensor([num_updates], device=device, dtype=torch.float32)
    world_collective.broadcast(update_t, src=0)

    # Get the first environment observation and start the optimization
    obs = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    next_obs = {}
    for k in obs_keys:
        torch_obs = torch.as_tensor(obs[k]).to(fabric.device)
        if k in cfg.algo.cnn_keys.encoder:
            torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
        elif k in cfg.algo.mlp_keys.encoder:
            torch_obs = torch_obs.float()
        step_data[k] = torch_obs
        next_obs[k] = torch_obs

    params = {"update": start_step, "last_log": last_log, "last_checkpoint": last_checkpoint}
    world_collective.scatter_object_list([None], [params] * world_collective.world_size, src=0)
    for _ in range(start_step, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs

            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
                with torch.no_grad():
                    # Sample an action given the observation received by the environment
                    normalized_obs = normalize_obs(next_obs, cfg.algo.cnn_keys.encoder, obs_keys)
                    actions, logprobs, _, values = agent(normalized_obs)
                    if is_continuous:
                        real_actions = torch.cat(actions, -1).cpu().numpy()
                    else:
                        real_actions = np.concatenate([act.argmax(dim=-1).cpu().numpy() for act in actions], axis=-1)
                    actions = torch.cat(actions, -1)

                # Single environment step
                obs, rewards, dones, truncated, info = envs.step(real_actions)
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
                                torch_v = torch_v.view(len(truncated_envs), -1, *torch_obs.shape[-2:]) / 255.0 - 0.5
                            real_next_obs[k][i] = torch_v
                    with torch.no_grad():
                        vals = agent.get_value(real_next_obs).cpu().numpy()
                        rewards[truncated_envs] += vals.reshape(rewards[truncated_envs].shape)
                dones = np.logical_or(dones, truncated)
                dones = torch.as_tensor(dones, dtype=torch.float32, device=device).view(cfg.env.num_envs, -1)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(cfg.env.num_envs, -1)

            # Update the step data
            step_data["dones"] = dones
            step_data["values"] = values
            step_data["actions"] = actions
            step_data["logprobs"] = logprobs
            step_data["rewards"] = rewards
            if cfg.buffer.memmap:
                step_data["returns"] = torch.zeros_like(rewards)
                step_data["advantages"] = torch.zeros_like(rewards)

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            # Update the observation and dones
            next_obs = {}
            for k in obs_keys:
                if k in cfg.algo.cnn_keys.encoder:
                    torch_obs = torch.as_tensor(obs[k], device=device)
                    torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
                elif k in cfg.algo.mlp_keys.encoder:
                    torch_obs = torch.as_tensor(obs[k], device=device, dtype=torch.float32)
                step_data[k] = torch_obs
                next_obs[k] = torch_obs

            if cfg.metric.log_level > 0 and "final_info" in info:
                for i, agent_ep_info in enumerate(info["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        if aggregator and not aggregator.disabled:
                            aggregator.update("Rewards/rew_avg", ep_rew)
                            aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        normalized_obs = normalize_obs(next_obs, cfg.algo.cnn_keys.encoder, obs_keys)
        next_values = agent.get_value(normalized_obs)
        returns, advantages = gae(
            rb["rewards"],
            rb["values"],
            rb["dones"],
            next_values,
            cfg.algo.rollout_steps,
            cfg.algo.gamma,
            cfg.algo.gae_lambda,
        )

        # Add returns and advantages to the buffer
        rb["returns"] = returns.float()
        rb["advantages"] = advantages.float()

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        # Send data to the training agents
        # Split data in an even way, when possible
        perm = torch.randperm(local_data.shape[0], device=device)
        chunks = local_data[perm].split(chunks_sizes)
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
    if fabric.is_global_zero:
        test(agent, fabric, cfg, log_dir)

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


def trainer(
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    optimization_pg: CollectibleGroup,
):
    global_rank = world_collective.rank
    group_world_size = world_collective.world_size - 1

    # Receive (possibly updated, by the make_env method for example) cfg from the player
    data = [None]
    world_collective.broadcast_object_list(data, src=0)
    cfg: Dict[str, Any] = data[0]

    # Initialize Fabric
    cfg.fabric.pop("loggers", None)
    cfg.fabric.pop("strategy", None)
    fabric: Fabric = hydra.utils.instantiate(
        cfg.fabric, strategy=DDPStrategy(process_group=optimization_pg), _convert_="all"
    )
    fabric.launch()
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # Environment setup
    agent_args = [None]
    world_collective.broadcast_object_list(agent_args, src=0)

    # Define the agent and the optimizer
    agent = PPOAgent(**agent_args[0])
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters())

    # Load the state from the checkpoint
    if cfg.checkpoint.resume_from:
        agent.load_state_dict(state["agent"])
        optimizer.load_state_dict(state["optimizer"])

    # Setup agent and optimizer with Fabric
    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()),
            src=1,
        )

    # Receive maximum number of updates from the player
    num_updates = torch.zeros(1, device=device)
    world_collective.broadcast(num_updates, src=0)
    num_updates = num_updates.item()

    # Linear learning rate scheduler
    if cfg.algo.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)
        if cfg.checkpoint.resume_from:
            scheduler.load_state_dict(state["scheduler"])

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator).to(device)

    # Start training
    last_train = 0
    train_step = 0

    policy_steps_per_update = cfg.env.num_envs * cfg.algo.rollout_steps
    params = [None]
    world_collective.scatter_object_list(params, [None for _ in range(world_collective.world_size)], src=0)
    params = params[0]
    update = params["update"]
    policy_step = update * policy_steps_per_update
    last_log = params["last_log"]
    last_checkpoint = params["last_checkpoint"]
    initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)
    initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)
    while True:
        # Wait for data
        data = [None]
        world_collective.scatter_object_list(data, [None for _ in range(world_collective.world_size)], src=0)
        data = data[0]
        if not isinstance(data, TensorDictBase) and data == -1:
            # Last Checkpoint
            if cfg.checkpoint.save_last:
                state = {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
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
            return
        data = make_tensordict(data, device=device)

        train_step += group_world_size

        # Prepare sampler
        indexes = list(range(data.shape[0]))
        sampler = BatchSampler(RandomSampler(indexes), batch_size=cfg.algo.per_rank_batch_size, drop_last=False)

        # Start training
        with timer(
            "Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg)
        ):
            # The Join context is needed because there can be the possibility
            # that some ranks receive less data
            with Join([agent._forward_module]):
                for _ in range(cfg.algo.update_epochs):
                    for batch_idxes in sampler:
                        batch = data[batch_idxes]
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
                update, initial=initial_clip_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        if cfg.algo.anneal_ent_coef:
            cfg.algo.ent_coef = polynomial_decay(
                update, initial=initial_ent_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        # Checkpoint model on rank-0: send it everything
        if cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every:
            last_checkpoint = policy_step
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
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
        update += 1
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
        player(fabric, cfg, world_collective, player_trainer_collective)
    else:
        trainer(world_collective, player_trainer_collective, optimization_pg)
