import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from math import prod

import gymnasium as gym
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric

from sheeprl.algos.sac.agent import SACActor, SACAgent, SACCritic
from sheeprl.algos.sac.args import SACArgs
from sheeprl.algos.sac.sac import train
from sheeprl.algos.sac.utils import test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import make_env


@torch.no_grad()
def player(args: SACArgs, world_collective: TorchCollective, player_trainer_collective: TorchCollective):
    root_dir = (
        args.root_dir
        if args.root_dir is not None
        else os.path.join("logs", "sac_decoupled", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    run_name = (
        args.run_name if args.run_name is not None else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    )
    logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
    logger.log_hyperparams(asdict(args))

    # Initialize Fabric
    fabric = Fabric(loggers=logger, callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if args.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args.capture_video,
                logger.log_dir,
                "train",
                mask_velocities=False,
                vector_env_idx=i,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = prod(envs.single_observation_space.shape)
    actor = SACActor(
        observation_dim=obs_dim,
        action_dim=act_dim,
        hidden_size=args.actor_hidden_size,
        action_low=envs.single_action_space.low,
        action_high=envs.single_action_space.high,
    ).to(device)
    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()), device=device
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

    # Metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(sync_on_compute=False),
                "Game/ep_len_avg": MeanMetric(sync_on_compute=False),
                "Time/step_per_second": MeanMetric(sync_on_compute=False),
            }
        )

    # Local data
    buffer_size = args.buffer_size // args.num_envs if not args.dry_run else 1
    rb = ReplayBuffer(buffer_size, args.num_envs, device=device, memmap=args.memmap_buffer)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    start_time = time.perf_counter()
    num_updates = int(args.total_steps // args.num_envs) if not args.dry_run else 1
    args.learning_starts = args.learning_starts // args.num_envs if not args.dry_run else 0

    with device:
        # Get the first environment observation and start the optimization
        obs = torch.tensor(envs.reset(seed=args.seed)[0], dtype=torch.float32)  # [N_envs, N_obs]

    for global_step in range(1, num_updates + 1):
        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            # Sample an action given the observation received by the environment
            with torch.no_grad():
                actions, _ = actor(obs)
                actions = actions.cpu().numpy()
        next_obs, rewards, dones, truncated, infos = envs.step(actions)
        dones = np.logical_or(dones, truncated)

        if "final_info" in infos:
            for i, agent_final_info in enumerate(infos["final_info"]):
                if agent_final_info is not None and "episode" in agent_final_info:
                    fabric.print(
                        f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                    )
                    aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                    aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

        # Save the real next observation
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    real_next_obs[idx] = final_obs

        with device:
            next_obs = torch.tensor(real_next_obs, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32).view(args.num_envs, -1)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(args.num_envs, -1)  # [N_envs, 1]
            dones = torch.tensor(dones, dtype=torch.float32).view(args.num_envs, -1)

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        if not args.sample_next_obs:
            step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Send data to the training agents
        if global_step >= args.learning_starts - 1:
            training_steps = args.learning_starts if global_step == args.learning_starts - 1 else 1
            chunks = rb.sample(
                training_steps * args.gradient_steps * args.per_rank_batch_size * (fabric.world_size - 1),
                sample_next_obs=args.sample_next_obs,
            ).split(training_steps * args.gradient_steps * args.per_rank_batch_size)
            world_collective.scatter_object_list([None], [None] + chunks, src=0)

            # Gather metrics from the trainers to be plotted
            metrics = [None]
            player_trainer_collective.broadcast_object_list(metrics, src=1)

            # Wait the trainers to finish
            player_trainer_collective.broadcast(flattened_parameters, src=1)

            # Convert back the parameters
            torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

            fabric.log_dict(metrics[0], global_step)
        aggregator.update("Time/step_per_second", int(global_step / (time.perf_counter() - start_time)))
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint model
        if (args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0) or args.dry_run:
            ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{global_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_player",
                fabric=fabric,
                player_trainer_collective=player_trainer_collective,
                ckpt_path=ckpt_path,
                replay_buffer=rb if args.checkpoint_buffer else None,
            )

    world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)

    # Last Checkpoint
    ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{num_updates}_{fabric.global_rank}.ckpt"
    fabric.call(
        "on_checkpoint_player",
        fabric=fabric,
        player_trainer_collective=player_trainer_collective,
        ckpt_path=ckpt_path,
        replay_buffer=rb if args.checkpoint_buffer else None,
    )

    envs.close()
    if fabric.is_global_zero:
        test_env = make_env(
            args.env_id,
            None,
            0,
            args.capture_video,
            fabric.logger.log_dir,
            "test",
            mask_velocities=False,
            vector_env_idx=0,
        )()
        test(actor, test_env, fabric, args)


def trainer(
    args: SACArgs,
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    optimization_pg: CollectibleGroup,
):
    global_rank = world_collective.rank

    # Initialize Fabric
    fabric = Fabric(strategy=DDPStrategy(process_group=optimization_pg), callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if args.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env([make_env(args.env_id, 0, 0, False, None, mask_velocities=False)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = prod(envs.single_observation_space.shape)
    actor = fabric.setup_module(
        SACActor(
            observation_dim=obs_dim,
            action_dim=act_dim,
            hidden_size=args.actor_hidden_size,
            action_low=envs.single_action_space.low,
            action_high=envs.single_action_space.high,
        )
    )
    critics = [
        fabric.setup_module(
            SACCritic(observation_dim=obs_dim + act_dim, hidden_size=args.critic_hidden_size, num_critics=1)
        )
        for _ in range(args.num_critics)
    ]
    target_entropy = -act_dim
    agent = SACAgent(actor, critics, target_entropy, alpha=args.alpha, tau=args.tau, device=fabric.device)

    # Optimizers
    qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        Adam(agent.qfs.parameters(), lr=args.q_lr, eps=1e-4),
        Adam(agent.actor.parameters(), lr=args.policy_lr, eps=1e-4),
        Adam([agent.log_alpha], lr=args.alpha_lr, eps=1e-4),
    )

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()), src=1
        )

    # Metrics
    with fabric.device:
        aggregator = MetricAggregator(
            {
                "Loss/value_loss": MeanMetric(process_group=optimization_pg),
                "Loss/policy_loss": MeanMetric(process_group=optimization_pg),
                "Loss/alpha_loss": MeanMetric(process_group=optimization_pg),
            }
        )

    # Start training
    global_step = 1
    while True:
        # Wait for data
        data = [None]
        world_collective.scatter_object_list(data, [None for _ in range(world_collective.world_size)], src=0)
        data = data[0]
        if not isinstance(data, TensorDictBase) and data == -1:
            # Last Checkpoint
            if global_rank == 1:
                state = {
                    "agent": agent.state_dict(),
                    "qf_optimizer": qf_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "alpha_optimizer": alpha_optimizer.state_dict(),
                    "args": asdict(args),
                    "global_step": global_step,
                }
                fabric.call("on_checkpoint_trainer", player_trainer_collective=player_trainer_collective, state=state)
            return
        data = make_tensordict(data, device=device)
        sampler = BatchSampler(range(len(data)), batch_size=args.per_rank_batch_size, drop_last=False)
        for batch_idxes in sampler:
            train(
                fabric,
                agent,
                actor_optimizer,
                qf_optimizer,
                alpha_optimizer,
                data[batch_idxes],
                aggregator,
                global_step,
                args,
                group=optimization_pg,
            )
            global_step += 1

        # Send updated weights to the player
        metrics = aggregator.compute()
        aggregator.reset()
        if global_rank == 1:
            player_trainer_collective.broadcast_object_list(
                [metrics], src=1
            )  # Broadcast metrics: fake send with object list between rank-0 and rank-1
            player_trainer_collective.broadcast(
                torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()), src=1
            )

        # Checkpoint model on rank-0: send it everything
        if (args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0) or args.dry_run:
            if global_rank == 1:
                state = {
                    "agent": agent.state_dict(),
                    "qf_optimizer": qf_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "alpha_optimizer": alpha_optimizer.state_dict(),
                    "args": asdict(args),
                    "global_step": global_step,
                }
                fabric.call("on_checkpoint_trainer", player_trainer_collective=player_trainer_collective, state=state)


@register_algorithm(decoupled=True)
def main():
    parser = HfArgumentParser(SACArgs)
    args: SACArgs = parser.parse_args_into_dataclasses()[0]

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

    if world_collective.world_size == 1:
        raise RuntimeError(
            "Please run the script with the number of devices greater than 1: "
            "`lightning run model --devices=2 sheeprl.py ...`"
        )

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
        player(args, world_collective, player_trainer_collective)
    else:
        trainer(args, world_collective, player_trainer_collective, optimization_pg)


if __name__ == "__main__":
    main()
