import copy
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime

import gymnasium as gym
import torch
from gymnasium.vector import SyncVectorEnv
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase, make_tensordict
from torch.distributed.algorithms.join import Join
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import MeanMetric

from fabricrl.algos.ppo.args import PPOArgs
from fabricrl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from fabricrl.algos.ppo.utils import test
from fabricrl.data import ReplayBuffer
from fabricrl.models.models import MLP
from fabricrl.utils.metric import MetricAggregator
from fabricrl.utils.parser import HfArgumentParser
from fabricrl.utils.utils import gae, make_env, normalize_tensor, polynomial_decay

__all__ = ["main"]


@torch.no_grad()
def player(args: PPOArgs, world_collective: TorchCollective, player_trainer_collective: TorchCollective):
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}"

    logger = TensorBoardLogger(
        root_dir=os.path.join("logs", "ppo_decoupled", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")), name=run_name
    )
    logger.log_hyperparams(asdict(args))

    # Initialize Fabric object
    fabric = Fabric(loggers=logger)
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
                mask_velocities=args.mask_vel,
                vector_env_idx=i,
            )
            for i in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action space is supported")

    # Create the actor and critic models
    actor = MLP(
        input_dims=envs.single_observation_space.shape,
        output_dim=envs.single_action_space.n,
        hidden_sizes=(64, 64),
        activation=torch.nn.ReLU,
    ).to(device)
    critic = MLP(
        input_dims=envs.single_observation_space.shape,
        output_dim=1,
        hidden_sizes=(64, 64),
        activation=torch.nn.ReLU,
    ).to(device)

    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(list(actor.parameters()) + list(critic.parameters())),
        device=device,
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(
        flattened_parameters, list(actor.parameters()) + list(critic.parameters())
    )

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
        next_obs = torch.tensor(envs.reset(seed=args.seed)[0], device=device)
        next_done = torch.zeros(args.num_envs, 1).to(device)

    for _ in range(1, num_updates + 1):
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

    world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)
    envs.close()
    if fabric.is_global_zero:
        test(actor, envs, fabric, args)


def trainer(
    args: PPOArgs,
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    optimization_pg: CollectibleGroup,
):
    global_rank = world_collective.rank
    group_rank = global_rank - 1
    group_world_size = world_collective.world_size - 1

    # Initialize Fabric
    fabric = Fabric(strategy=DDPStrategy(process_group=optimization_pg))
    if not _is_using_cli():
        fabric.launch()
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    envs = SyncVectorEnv([make_env(args.env_id, 0, 0, False, None, mask_velocities=args.mask_vel)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Create the actor and critic models
    actor = MLP(
        input_dims=envs.single_observation_space.shape,
        output_dim=envs.single_action_space.n,
        hidden_sizes=(64, 64),
        activation=torch.nn.ReLU,
    )
    critic = MLP(
        input_dims=envs.single_observation_space.shape, output_dim=1, hidden_sizes=(64, 64), activation=torch.nn.ReLU
    )

    # Define the agent and the optimizer and setup them with Fabric
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-4)
    actor = fabric.setup_module(actor)
    critic = fabric.setup_module(critic)
    optimizer = fabric.setup_optimizers(optimizer)

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(
                list(actor.parameters()) + list(critic.parameters())
            ),
            src=1,
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
            return
        data = make_tensordict(data, device=device)
        update += 1

        # Prepare sampler
        indexes = list(range(data.shape[0]))
        if args.share_data:
            sampler = DistributedSampler(
                indexes, num_replicas=group_world_size, rank=group_rank, shuffle=True, seed=args.seed, drop_last=False
            )
        else:
            sampler = RandomSampler(indexes)
        sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)

        # The Join context is needed because there can be the possibility
        # that some ranks receive less data
        with Join([actor._forward_module, critic._forward_module]) if not args.share_data else nullcontext():
            for epoch in range(args.update_epochs):
                if args.share_data:
                    sampler.sampler.set_epoch(epoch)
                for batch_idxes in sampler:
                    batch = data[batch_idxes]
                    actions_logits = actor(batch["observations"])
                    new_values = critic(batch["observations"])

                    dist = Categorical(logits=actions_logits.unsqueeze(-2))
                    if args.normalize_advantages:
                        batch["advantages"] = normalize_tensor(batch["advantages"])

                    # Policy loss
                    pg_loss = policy_loss(
                        dist.log_prob(batch["actions"]),
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
                    ent_loss = entropy_loss(dist.entropy(), args.loss_reduction)

                    # Equation (9) in the paper
                    loss = pg_loss + args.vf_coef * v_loss + args.ent_coef * ent_loss

                    optimizer.zero_grad(set_to_none=True)
                    fabric.backward(loss)
                    if args.max_grad_norm > 0.0:
                        fabric.clip_gradients(actor, optimizer, max_norm=args.max_grad_norm)
                        fabric.clip_gradients(critic, optimizer, max_norm=args.max_grad_norm)
                    optimizer.step()

                    # Update metrics
                    aggregator.update("Loss/policy_loss", pg_loss.detach())
                    aggregator.update("Loss/value_loss", v_loss.detach())
                    aggregator.update("Loss/entropy_loss", ent_loss.detach())

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
                torch.nn.utils.convert_parameters.parameters_to_vector(
                    list(actor.parameters()) + list(critic.parameters())
                ),
                src=1,
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


def main():
    devices = os.environ.get("LT_DEVICES", None)
    if devices is None or devices == "1":
        raise RuntimeError(
            "Please run the script with the number of devices greater than 1: "
            "`lightning run model --devices=2 main.py ...`"
        )

    parser = HfArgumentParser(PPOArgs)
    args: PPOArgs = parser.parse_args_into_dataclasses()[0]

    if args.share_data:
        warnings.warn(
            "You have called the script with `--share_data=True`: "
            "decoupled scripts splits collected data in an almost-even way between the number of trainers"
        )

    world_collective = TorchCollective()
    player_trainer_collective = TorchCollective()
    world_collective.setup(backend="nccl" if os.environ.get("LT_ACCELERATOR", None) in ("gpu", "cuda") else "gloo")

    # Create a global group, assigning it to the collective: used by the player to exchange
    # collected experiences with the trainers
    world_collective.create_group()
    global_rank = world_collective.rank

    # Create a group between rank-0 (player) and rank-1 (trainer), assigning it to the collective:
    # used by rank-1 to send metrics to be tracked by the rank-0 at the end of a training episode
    player_trainer_collective.create_group(ranks=[0, 1])

    # Create a new group, without assigning it to the collective: in this way the trainers can
    # still communicate with the player through the global group, but they can optimize the agent
    # between themselves
    optimization_pg = world_collective.new_group(ranks=list(range(1, world_collective.world_size)))
    if global_rank == 0:
        player(args, world_collective, player_trainer_collective)
    else:
        trainer(args, world_collective, player_trainer_collective, optimization_pg)


if __name__ == "__main__":
    main()
