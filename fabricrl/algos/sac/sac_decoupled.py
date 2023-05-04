import os
import time
from datetime import datetime
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
from torchmetrics import MeanMetric

from fabricrl.algos.ppo.utils import make_env
from fabricrl.algos.sac.agent import Actor, Critic, SACAgent
from fabricrl.algos.sac.args import SACArgs
from fabricrl.algos.sac.sac import train
from fabricrl.algos.sac.utils import test
from fabricrl.data.buffers import ReplayBuffer
from fabricrl.utils.metric import MetricAggregator
from fabricrl.utils.parser import HfArgumentParser

__all__ = ["main"]


@torch.inference_mode()
def player(args: SACArgs, world_collective: TorchCollective, player_trainer_collective: TorchCollective):
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    logger = TensorBoardLogger(
        root_dir=os.path.join("logs", "sac", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
        name=run_name,
    )

    # Initialize Fabric
    fabric = Fabric(loggers=logger)
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Log hyperparameters
    fabric.logger.experiment.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args.capture_video,
                logger.log_dir,
                "train",
                mask_velocities=False,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    actor = Actor(envs)
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
    rb = ReplayBuffer(args.buffer_size // args.num_envs, args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    start_time = time.time()
    num_updates = args.total_steps // args.num_envs
    args.learning_starts = args.learning_starts // args.num_envs
    if args.learning_starts <= 1:
        args.learning_starts = 2

    with device:
        # Get the first environment observation and start the optimization
        obs = torch.tensor(envs.reset(seed=args.seed)[0])  # [N_envs, N_obs]

    for global_step in range(num_updates):
        # Sample an action given the observation received by the environment
        with torch.inference_mode():
            mean, std = actor(obs)
            actions, _ = actor.get_action_and_log_prob(mean, std)
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
            next_obs = torch.tensor(real_next_obs)
            actions = torch.tensor(actions).view(args.num_envs, -1)
            rewards = torch.tensor(rewards).view(args.num_envs, -1).float()  # [N_envs, 1]
            dones = torch.tensor(dones).view(args.num_envs, -1).float()

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Send data to the training agents
        if global_step > args.learning_starts:
            for _ in range(args.gradient_steps):
                chunks = rb.sample(args.batch_size * (fabric.world_size - 1)).split(args.batch_size)
                world_collective.scatter_object_list([None], [None] + chunks, src=0)

            # Gather metrics from the trainers to be plotted
            metrics = [None]
            player_trainer_collective.broadcast_object_list(metrics, src=1)

            # Wait the trainers to finish
            player_trainer_collective.broadcast(flattened_parameters, src=1)

            # Convert back the parameters
            torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

            fabric.log_dict(metrics[0], global_step)
        aggregator.update("Time/step_per_second", int(global_step / (time.time() - start_time)))
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

    world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)
    envs.close()
    if fabric.is_global_zero:
        test(actor, device, fabric.logger.experiment, args)


def trainer(
    args: SACArgs,
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    optimization_pg: CollectibleGroup,
):
    global_rank = world_collective.rank
    global_rank - 1

    # Initialize Fabric
    fabric = Fabric(strategy=DDPStrategy(process_group=optimization_pg))  # accelerator="cuda" if args.cuda else "cpu"
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, 0, False, None, mask_velocities=False)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    actor = fabric.setup_module(Actor(envs))
    critics = [fabric.setup_module(Critic(envs)) for _ in range(args.num_critics)]
    target_entropy = -prod(envs.single_action_space.shape)
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
    global_step = 0
    while True:
        for _ in range(args.gradient_steps):
            # Wait for data
            data = [None]
            world_collective.scatter_object_list(data, [None for _ in range(world_collective.world_size)], src=0)
            data = data[0]
            if not isinstance(data, TensorDictBase) and data == -1:
                return
            data = make_tensordict(data, device=device)

            train(
                fabric,
                agent,
                actor_optimizer,
                qf_optimizer,
                alpha_optimizer,
                data,
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


def main():
    parser = HfArgumentParser(SACArgs)
    args: SACArgs = parser.parse_args_into_dataclasses()[0]

    world_collective = TorchCollective()
    player_trainer_collective = TorchCollective()
    world_collective.setup(backend="gloo")  # "nccl" if args.player_on_gpu and args.cuda else

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
