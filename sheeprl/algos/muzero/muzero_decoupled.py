import os
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timedelta

import gymnasium as gym
import torch
from lightning import Fabric
from lightning_fabric.fabric import _is_using_cli
from lightning_fabric.loggers import TensorBoardLogger
from lightning_fabric.plugins.collectives import TorchCollective
from tensordict import TensorDict
from torchmetrics import MeanMetric

from sheeprl.algos.muzero.args import MuzeroArgs
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import make_env


@torch.no_grad()
def player(
    args: MuzeroArgs,
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    workers_collective: TorchCollective,
):
    rank = world_collective.rank

    root_dir = (
        args.root_dir
        if args.root_dir is not None
        else os.path.join("logs", "ppo_decoupled", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    run_name = (
        args.run_name if args.run_name is not None else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    )
    logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
    logger.log_hyperparams(asdict(args))

    # Initialize Fabric object
    fabric = Fabric(loggers=logger)
    if not _is_using_cli():
        fabric.launch()
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    envs = make_env(
        args.env_id,
        args.seed + rank,
        0,
        args.capture_video,
        logger.log_dir,
        "train",
        mask_velocities=args.mask_vel,
        vector_env_idx=rank,
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action space is supported")

    # Create the actor and critic models
    network = MuzeroNetwork()

    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(network.parameters()),
        device=device,
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, network.parameters())

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
    rb = MuzeroBuffer(args.rollout_steps, args.num_envs, device=device)
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
        next_obs = torch.tensor(envs.reset(seed=args.seed)[0], device=device)
        next_done = torch.zeros(args.num_envs, 1).to(device)

    for update in range(1, num_updates + 1):
        for _ in range(0, args.rollout_steps):
            global_step += args.num_envs

            # Sample an action given the observation received by the environment
            actions_logits = actor(next_obs)
            dist = Categorical(logits=actions_logits.unsqueeze(-2))
            action = dist.sample()
            logprob = dist.log_prob(action)

            # Compute the value of the current observation
            value = critic(next_obs)

            # Store the current step data
            step_data["dones"] = next_done
            step_data["values"] = value
            step_data["actions"] = action
            step_data["logprobs"] = logprob
            step_data["observations"] = next_obs

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            with device:
                next_obs = torch.tensor(next_obs)
                next_done = (
                    torch.logical_or(torch.tensor(done), torch.tensor(truncated)).view(args.num_envs, -1).float()
                )  # [N_envs, 1]

                # Save reward for the last (observation, action) pair
                step_data["rewards"] = torch.tensor(reward).view(args.num_envs, -1)  # [N_envs, 1]

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                        )
                        aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                        aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        next_values = critic(next_obs)
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
        torch.nn.utils.convert_parameters.vector_to_parameters(
            flattened_parameters, list(actor.parameters()) + list(critic.parameters())
        )

        # Log metrics
        aggregator.update("Time/step_per_second", int(global_step / (time.time() - start_time)))
        fabric.log_dict(metrics[0], global_step)
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint Model
        if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run:
            state = [None]
            player_trainer_collective.broadcast_object_list(state, src=1)
            ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{update}_{fabric.global_rank}.ckpt"
            fabric.save(ckpt_path, state[0])

    world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)
    envs.close()
    if fabric.is_global_zero:
        test_env = make_env(
            args.env_id,
            None,
            0,
            args.capture_video,
            fabric.logger.log_dir,
            "test",
            mask_velocities=args.mask_vel,
            vector_env_idx=0,
        )()
        test(actor, test_env, fabric, args)


@register_algorithm(decoupled=True)
def main():
    devices = os.environ.get("LT_DEVICES", None)
    if devices is None or devices == "1":
        raise RuntimeError(
            "Please run the script with the number of devices greater than 1: "
            "`lightning run model --devices=2 sheeprl.py ...`"
        )

    parser = HfArgumentParser(MuzeroArgs)
    args: MuzeroArgs = parser.parse_args_into_dataclasses()[0]

    if args.share_data:
        warnings.warn(
            "You have called the script with `--share_data=True`: "
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

    # Create a group between rank-0 (trainer) and rank-1 (player), assigning it to the collective:
    # used by rank-1 to send metrics to be tracked by the rank-0 at the end of a training episode
    player_trainer_collective.create_group(ranks=[0, 1], timeout=timedelta(days=1))

    # Create a new group, without assigning it to the collective: in this way the trainers can
    # still communicate with the player through the global group, but they can optimize the agent
    # between themselves

    # NOTE: here we have many players and a single trainer, as opposed to other decoupled scripts
    workers_pg = world_collective.new_group(
        ranks=list(range(1, world_collective.world_size)), timeout=timedelta(days=1)
    )
    if global_rank == 0:
        trainer(args, world_collective, player_trainer_collective)
    else:
        player(args, world_collective, player_trainer_collective, workers_pg)
