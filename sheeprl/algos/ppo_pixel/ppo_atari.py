import copy
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from math import prod

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase, make_tensordict
from torch.distributed.algorithms.join import Join
from torch.optim import Adam
from torch.utils.data import BatchSampler, RandomSampler
from torchmetrics import MeanMetric

from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo_pixel.agent import PPOAtariAgent
from sheeprl.algos.ppo_pixel.args import PPOAtariArgs
from sheeprl.algos.ppo_pixel.utils import test_ppo_pixel
from sheeprl.data import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.imports import _IS_ATARI_AVAILABLE, _IS_ATARI_ROMS_AVAILABLE
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay

if not _IS_ATARI_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_ATARI_AVAILABLE))

if not _IS_ATARI_ROMS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_ATARI_ROMS_AVAILABLE))


def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    prefix: str = "",
    vector_env_idx: int = 0,
    frame_stack: int = 4,
    screen_size: int = 64,
    action_repeat: int = 2,
):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and vector_env_idx == 0 and idx == 0:
            env = gym.experimental.wrappers.RecordVideoV0(
                env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
            )
        env = AtariPreprocessing(
            env,
            screen_size=screen_size,
            frame_skip=action_repeat,
            grayscale_obs=True,
            grayscale_newaxis=True,
            scale_obs=True,
        )
        env = gym.wrappers.TransformObservation(env, lambda obs: obs.transpose(2, 0, 1))
        env.observation_space = gym.spaces.Box(
            0, 1, (env.observation_space.shape[2], screen_size, screen_size), np.float32
        )
        if frame_stack > 0:
            env = gym.wrappers.FrameStack(env, frame_stack)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


@torch.no_grad()
def player(args: PPOAtariArgs, world_collective: TorchCollective, player_trainer_collective: TorchCollective):
    root_dir = (
        args.root_dir
        if args.root_dir is not None
        else os.path.join("logs", "ppo_atari", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    run_name = (
        args.run_name if args.run_name is not None else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    )

    logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
    logger.log_hyperparams(asdict(args))

    # Initialize Fabric object
    fabric = Fabric(loggers=logger, callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                0,
                args.capture_video,
                logger.log_dir,
                "train",
                vector_env_idx=i,
                frame_stack=args.frame_stack,
                screen_size=args.screen_size,
                action_repeat=args.action_repeat,
            )
            for i in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action space is supported")

    # Create the actor and critic models
    features_dim = 512
    agent = PPOAtariAgent(
        in_channels=prod(envs.single_observation_space.shape[:-2]),
        features_dim=features_dim,
        action_dim=envs.single_action_space.n,
        screen_size=args.screen_size,
    )

    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()), device=device
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, agent.parameters())

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
    rb = ReplayBuffer(args.rollout_steps, args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_step = int(args.num_envs * args.rollout_steps)
    num_updates = args.total_steps // single_global_step if not args.dry_run else 1
    if single_global_step < world_collective.world_size - 1:
        raise RuntimeError(
            "The number of trainers ({}) is greater than the available collected data ({}). ".format(
                world_collective.world_size - 1, single_global_step
            )
            + "Consider to lower the number of trainers at least to the size of available collected data"
        )
    chunks_sizes = [
        len(chunk) for chunk in torch.tensor_split(torch.arange(single_global_step), world_collective.world_size - 1)
    ]

    # Broadcast num_updates to all the world
    update_t = torch.tensor([num_updates], device=device, dtype=torch.float32)
    world_collective.broadcast(update_t, src=0)

    with device:
        # Get the first environment observation and start the optimization
        next_obs = torch.tensor(np.array(envs.reset(seed=args.seed)[0]), device=device)
        next_done = torch.zeros(args.num_envs, 1).to(device)

    for update in range(1, num_updates + 1):
        for _ in range(0, args.rollout_steps):
            global_step += args.num_envs

            # Sample an action given the observation received by the environment
            next_obs = next_obs.flatten(start_dim=1, end_dim=-3)
            action, logprob, _, value = agent(next_obs)

            # Single environment step
            obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            with device:
                obs = torch.tensor(np.array(obs))  # [N_envs, N_obs]
                rewards = torch.tensor(reward).view(args.num_envs, -1)  # [N_envs, 1]
                done = torch.logical_or(torch.tensor(done), torch.tensor(truncated))  # [N_envs, 1]
                done = done.view(args.num_envs, -1).float()

            # Update the step data
            step_data["dones"] = next_done
            step_data["values"] = value
            step_data["actions"] = action
            step_data["logprobs"] = logprob
            step_data["rewards"] = rewards
            step_data["observations"] = next_obs

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            # Update the observation and done
            next_obs = obs
            next_done = done

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                        )
                        aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                        aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        next_obs = next_obs.flatten(start_dim=1, end_dim=-3)
        next_values = agent.get_value(next_obs)
        returns, advantages = gae(
            rb["rewards"],
            rb["values"],
            rb["dones"],
            next_values,
            next_done,
            args.rollout_steps,
            args.gamma,
            args.gae_lambda,
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

        # Gather metrics from the trainers to be plotted
        metrics = [None]
        player_trainer_collective.broadcast_object_list(metrics, src=1)

        # Wait the trainers to finish
        player_trainer_collective.broadcast(flattened_parameters, src=1)

        # Convert back the parameters
        torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, agent.parameters())

        # Log metrics
        aggregator.update("Time/step_per_second", int(global_step / (time.time() - start_time)))
        fabric.log_dict(metrics[0], global_step)
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint model on rank-0: send it everything
        if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run:
            ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{update}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_player",
                fabric=fabric,
                player_trainer_collective=player_trainer_collective,
                ckpt_path=ckpt_path,
            )

    world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)

    # Last Checkpoint
    if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run:
        ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{update}_{fabric.global_rank}.ckpt"
        fabric.call(
            "on_checkpoint_player",
            fabric=fabric,
            player_trainer_collective=player_trainer_collective,
            ckpt_path=ckpt_path,
        )

    envs.close()
    if fabric.is_global_zero:
        test_env = make_env(
            args.env_id,
            args.seed,
            0,
            args.capture_video,
            fabric.logger.log_dir,
            "test",
            vector_env_idx=0,
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            screen_size=args.screen_size,
        )()
        test_ppo_pixel(agent, test_env, fabric, args, normalize=False)


def trainer(
    args: PPOAtariArgs,
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
    envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                0,
                False,
                None,
                action_repeat=args.action_repeat,
                screen_size=args.screen_size,
                frame_stack=args.frame_stack,
            )
        ]
    )

    # Create the actor and critic models
    features_dim = 512
    agent = PPOAtariAgent(
        in_channels=prod(envs.single_observation_space.shape[:-2]),
        features_dim=features_dim,
        action_dim=envs.single_action_space.n,
        screen_size=args.screen_size,
    )

    # Define the agent and the optimizer and setup them with Fabric
    optimizer = Adam(agent.parameters(), lr=args.lr, eps=1e-4)
    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()), src=1
        )

    # Receive maximum number of updates from the player
    num_updates = torch.zeros(1, device=device)
    world_collective.broadcast(num_updates, src=0)
    num_updates = num_updates.item()

    # Linear learning rate scheduler
    if args.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)

    # Metrics
    with fabric.device:
        aggregator = MetricAggregator(
            {
                "Loss/value_loss": MeanMetric(process_group=optimization_pg),
                "Loss/policy_loss": MeanMetric(process_group=optimization_pg),
                "Loss/entropy_loss": MeanMetric(process_group=optimization_pg),
            }
        )

    # Start training
    update = 0
    initial_ent_coef = copy.deepcopy(args.ent_coef)
    initial_clip_coef = copy.deepcopy(args.clip_coef)
    while True:
        # Wait for data
        data = [None]
        world_collective.scatter_object_list(data, [None for _ in range(world_collective.world_size)], src=0)
        data = data[0]
        if not isinstance(data, TensorDictBase) and data == -1:
            # Last Checkpoint
            if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run:
                if global_rank == 1:
                    state = {
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": asdict(args),
                        "update_step": update,
                        "scheduler": scheduler.state_dict() if args.anneal_lr else None,
                    }
                    fabric.call(
                        "on_checkpoint_trainer", player_trainer_collective=player_trainer_collective, state=state
                    )
            return
        data = make_tensordict(data, device=device)
        update += 1

        # Prepare sampler
        indexes = list(range(data.shape[0]))
        sampler = BatchSampler(RandomSampler(indexes), batch_size=args.per_rank_batch_size, drop_last=False)

        # The Join context is needed because there can be the possibility
        # that some ranks receive less data
        with Join([agent._forward_module]):
            for _ in range(args.update_epochs):
                for batch_idxes in sampler:
                    batch = data[batch_idxes]
                    _, new_logprobs, entropies, new_values = agent(batch["observations"], batch["actions"])

                    if args.normalize_advantages:
                        batch["advantages"] = normalize_tensor(batch["advantages"])

                    # Policy loss
                    pg_loss = policy_loss(
                        new_logprobs,
                        batch["logprobs"],
                        batch["advantages"],
                        args.clip_coef,
                        args.loss_reduction,
                    )

                    # Value loss
                    v_loss = value_loss(
                        new_values,
                        batch["values"],
                        batch["returns"],
                        args.clip_coef,
                        args.clip_vloss,
                        args.loss_reduction,
                    )

                    # Entropy loss
                    entropy = entropy_loss(entropies, reduction=args.loss_reduction)

                    # Equation (9) in the paper
                    loss = pg_loss + args.vf_coef * v_loss + args.ent_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    fabric.backward(loss)
                    if args.max_grad_norm > 0.0:
                        fabric.clip_gradients(agent, optimizer, max_norm=args.max_grad_norm)
                    optimizer.step()

                    # Update metrics
                    aggregator.update("Loss/policy_loss", pg_loss.detach())
                    aggregator.update("Loss/value_loss", v_loss.detach())
                    aggregator.update("Loss/entropy_loss", entropy.detach())

        # Send updated weights to the player
        metrics = aggregator.compute()
        aggregator.reset()
        if global_rank == 1:
            if args.anneal_lr:
                metrics["Info/learning_rate"] = scheduler.get_last_lr()[0]
            else:
                metrics["Info/learning_rate"] = args.lr
            metrics["Info/clip_coef"] = args.clip_coef
            metrics["Info/ent_coef"] = args.ent_coef
            player_trainer_collective.broadcast_object_list(
                [metrics], src=1
            )  # Broadcast metrics: fake send with object list between rank-0 and rank-1
            player_trainer_collective.broadcast(
                torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()), src=1
            )

        if args.anneal_lr:
            scheduler.step()

        if args.anneal_clip_coef:
            args.clip_coef = polynomial_decay(
                update, initial=initial_clip_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        if args.anneal_ent_coef:
            args.ent_coef = polynomial_decay(
                update, initial=initial_ent_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        # Checkpoint Model
        if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run:
            if global_rank == 1:
                state = {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": asdict(args),
                    "update_step": update,
                    "scheduler": scheduler.state_dict() if args.anneal_lr else None,
                }
                fabric.call("on_checkpoint_trainer", player_trainer_collective=player_trainer_collective, state=state)


@register_algorithm(decoupled=True)
def main():
    devices = os.environ.get("LT_DEVICES", None)
    if devices is None or devices == "1":
        raise RuntimeError(
            "Please run the script with the number of devices greater than 1: "
            "`lightning run model --devices=2 sheeprl.py ...`"
        )

    parser = HfArgumentParser(PPOAtariArgs)
    args: PPOAtariArgs = parser.parse_args_into_dataclasses()[0]
    args.frame_stack = 0

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
        player(args, world_collective, player_trainer_collective)
    else:
        trainer(args, world_collective, player_trainer_collective, optimization_pg)


if __name__ == "__main__":
    main()
