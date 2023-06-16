import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning_fabric.fabric import _is_using_cli
from lightning_fabric.loggers import TensorBoardLogger
from lightning_fabric.plugins.collectives import TorchCollective
from lightning_fabric.utilities.rank_zero import rank_zero_only
from torch.optim import Adam
from torchmetrics import MeanMetric

from sheeprl.algos.muzero.agent import RecurrentMuzero
from sheeprl.algos.muzero.args import MuzeroArgs
from sheeprl.algos.muzero.loss import policy_loss, reward_loss, value_loss
from sheeprl.data.buffers import Trajectory, TrajectoryReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import make_env


def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    prefix: str = "",
    vector_env_idx: int = 0,
    frame_stack: int = 1,
    screen_size: int = 64,
    action_repeat: int = 1,
):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and vector_env_idx == 0 and idx == 0:
            env = gym.experimental.wrappers.RecordVideoV0(
                env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
            )
        env = AtariPreprocessing(
            env,
            screen_size=screen_size,
            frame_skip=action_repeat,
            grayscale_obs=False,
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


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self):
        self.maximum = float("inf")
        self.minimum = -float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def visit_softmax_temperature(training_steps: int) -> float:
    """Scale the temperature of the softmax action selection by the number of moves played in the game.

    Args:
        training_steps (int): The number of time the network's weight have been updated (an optimizer.step()).
    """
    if training_steps < 500e3:
        return 1.0
    elif training_steps < 750e3:
        return 0.5
    else:
        return 0.25


class Node:
    """A Node in the MCTS tree.

    It is used to store the statistics of the node and its children, in order to progress the MCTS search and store the
    backpropagation values.

    Attributes:
        prior (float): The prior probability of the node.
        image (torch.Tensor): The image of the node.
        hidden_state (Optional[torch.Tensor]): The hidden state of the node.
        reward (float): The reward of the node.
        value_sum (float): The sum of the values of the node.
        visit_count (int): The number of times the node has been visited.
        children (dict[int, Node]): The children of the node, one for each action where actions are represented as integers
            running from 0 to num_actions - 1.
    """

    def __init__(self, prior: float, image: torch.Tensor = None):
        """A Node in the MCTS tree.

        Args:
            prior (float): The prior probability of the node.
            image (torch.Tensor): The image of the node.
                The image is used to create the initial hidden state for the network in MCTS.
                Hence, it is needed only for the starting node of every MCTS search.
        """
        self.prior: float = prior
        self.image: torch.Tensor = image

        self.hidden_state: Optional[torch.Tensor] = None
        self.reward: float = 0.0
        self.value_sum: torch.tensor = 0.0
        self.visit_count: int = 0
        self.children: dict[int, Node] = {}

    def expanded(self) -> bool:
        """Returns True if the node is already expanded, False otherwise.

        When a node is visited for the first time, the search stops, the node gets expanded to compute its children, and
        then the `backpropagate` phase of MCTS starts. If a node is already expanded, instead, the search continues
        until an unvisited node is reached.
        """
        return len(self.children) > 0

    def value(self) -> float:
        """Returns the value of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """Add exploration noise to the prior probabilities."""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

    def mcts(
        self,
        agent: torch.nn.Module,
        num_simulations: int,
        gamma: float = 0.997,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25,
    ):
        """Runs MCTS for num_simulations"""
        # Initialize the hidden state and compute the actor's policy logits
        hidden_state, policy_logits, _ = agent.initial_inference(self.image)
        self.hidden_state = hidden_state

        # Use the actor's policy to initialize the children of the node.
        # The policy results are used as prior probabilities.
        normalized_policy = torch.exp(policy_logits)
        normalized_policy = normalized_policy / torch.sum(normalized_policy)
        for action in range(normalized_policy.numel()):
            self.children[action] = Node(normalized_policy[:, action].item())
        self.add_exploration_noise(dirichlet_alpha, exploration_fraction)

        # Expand until an unvisited node (i.e. not yet expanded) is reached
        min_max_stats = MinMaxStats()

        for sim_num in range(num_simulations):
            node = self
            search_path = [node]

            while node.expanded():
                # Select the child with the highest UCB score
                _, imagined_action, child = max(
                    (
                        ucb_score(
                            parent=node,
                            child=child,
                            min_max_stats=min_max_stats,
                            gamma=gamma,
                            pb_c_base=19652,
                            pb_c_init=1.25,
                        ),
                        action,
                        child,
                    )
                    for action, child in node.children.items()
                )
                search_path.append(child)
                node = child

        # When a path from the starting node to an unvisited node is found, expand the unvisited node
        parent = search_path[-2]
        hidden_state, reward, policy_logits, value = agent.recurrent_inference(
            torch.tensor([imagined_action]).view(1, 1, 1).to(dtype=torch.float32), parent.hidden_state
        )
        node.hidden_state = hidden_state
        node.reward = reward
        normalized_policy = torch.exp(policy_logits)
        normalized_policy = normalized_policy / torch.sum(normalized_policy)
        for action in range(normalized_policy.numel()):
            self.children[action] = Node(normalized_policy.squeeze()[action].item())

        # Backpropagate the search path to update the nodes' statistics
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + gamma * value


def ucb_score(
    parent: Node, child: Node, min_max_stats: MinMaxStats, gamma: float, pb_c_base: float, pb_c_init: float
) -> float:
    """Computes the UCB score of a child node relative to its parent, using the min-max bounds on the value function to
    normalize the node value."""
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = child.reward + gamma * min_max_stats.normalize(child.value())
    else:
        value_score = 0
    return prior_score + value_score


@torch.no_grad()
def player(
    args: MuzeroArgs,
    world_collective: TorchCollective,
    buffer_players_collective: TorchCollective,
    players_trainer_collective: TorchCollective,
):
    rank_zero_only.rank = 0  # TODO: fix rank_zero_only issue with logger
    rank = world_collective.rank

    root_dir = (
        args.root_dir
        if args.root_dir is not None
        else os.path.join("logs", "muzero_decoupled", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    run_name = (
        args.run_name if args.run_name is not None else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    )
    logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
    logger.log_hyperparams(asdict(args))

    # Initialize Fabric object
    fabric = Fabric(loggers=logger)
    fabric.log("test", 1)
    if not _is_using_cli():
        fabric.launch()
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    env = make_env(
        args.env_id,
        args.seed + rank,
        0,
        args.capture_video,
        logger.log_dir,
        "train",
        frame_stack=1,
        vector_env_idx=rank,
    )()
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action space is supported")

    # Create the model
    agent = RecurrentMuzero(num_actions=env.action_space.n).to(device)

    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()),
        device=device,
    )

    # Receive the first weights from the rank-0, a.k.a. the trainer
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    print("PLAYER WAITING FOR INITIAL PARAMS")
    players_trainer_collective.broadcast(flattened_parameters, src=world_collective.world_size - 1)
    print("OK!! PLAYER RECEIVED INITIAL PARAMS")
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

    # Global variables
    global_step = 0
    start_time = time.time()
    # Global variables
    start_time = time.perf_counter()

    num_updates = int(args.total_steps // args.num_players) if not args.dry_run else 1
    args.learning_starts = args.learning_starts // args.num_envs if not args.dry_run else 0

    for update in range(1, num_updates + 1):
        print("PLAYER STARTING TO PLAY")
        print(f"Player is playing update num {update} / {num_updates + 1}")
        # reset the episode at every update
        with device:
            # Get the first environment observation and start the optimization
            obs: torch.Tensor = torch.tensor(env.reset(seed=args.seed)[0], device=device).view(
                1, 3, 64, 64
            )  # shape (C, H, W)

        steps_data = None
        for step in range(0, args.max_trajectory_len):
            global_step += 1
            node = Node(prior=0, image=obs)

            # start MCTS
            node.mcts(agent, args.num_simulations, args.gamma, args.dirichlet_alpha, args.exploration_fraction)

            # Select action based on the visit count distribution and the temperature
            visits_count = torch.tensor([child.visit_count for child in node.children.values()])
            temperature = visit_softmax_temperature(training_steps=agent.training_steps)
            visits_count = visits_count / temperature
            action = torch.distributions.Categorical(logits=visits_count).sample()

            # Single environment step
            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))

            # Store the current step data
            if steps_data is None:
                steps_data = Trajectory(
                    {
                        "policies": visits_count.reshape(1, 1, -1),
                        "actions": action.reshape(1, 1, -1),
                        "observations": obs.unsqueeze(0),
                        "rewards": torch.tensor([reward]).reshape(1, 1, -1),
                        "values": node.value_sum.reshape(1, 1, -1),
                    },
                    batch_size=(1, 1),
                )
            else:
                steps_data = torch.cat(
                    [
                        steps_data,
                        Trajectory(
                            {
                                "policies": visits_count.reshape(1, 1, -1),
                                "actions": action.reshape(1, 1, -1),
                                "observations": obs.unsqueeze(0),
                                "rewards": torch.tensor([reward]).reshape(1, 1, -1),
                                "values": node.value_sum.reshape(1, 1, -1),
                            },
                            batch_size=(1, 1),
                        ),
                    ],
                )

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-{rank}: global_step={global_step}, reward={agent_final_info['episode']['r'][0]}"
                        )
                        aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                        aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

            if done or truncated:
                break
            else:
                with device:
                    obs = torch.tensor(next_obs).view(1, 3, 64, 64)

        agent.training_steps += 1
        # After episode is done, send data to the buffer
        print("PLAYER SENDING DATA!")
        buffer_players_collective.gather_object(steps_data, None, dst=0)
        print("OK!! PLAYER SENT DATA!")

        # Gather metrics from the trainer to be plotted
        # metrics = [None]
        # players_trainer_collective.broadcast_object_list(metrics, src=world_collective.world_size - 1)

        # Wait the trainer to finish
        print("PLAYER WAITING FOR PARAMS")
        players_trainer_collective.broadcast(flattened_parameters, src=world_collective.world_size - 1)
        print("OK!! PLAYER RECEIVED PARAMS")

        # Convert back the parameters
        torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, agent.parameters())

        # Log metrics
        aggregator.update("Time/step_per_second", int(global_step / (time.time() - start_time)))
        # fabric.log_dict(metrics[0], global_step)
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint Model # TODO move to the trainer
        if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run:
            state = [None]
            players_trainer_collective.broadcast_object_list(state, src=1)
            ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{update}_{fabric.global_rank}.ckpt"
            fabric.save(ckpt_path, state[0])

    # world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)
    env.close()
    # if fabric.is_global_zero:
    #     test_env = make_env(
    #         args.env_id,
    #         None,
    #         0,
    #         args.capture_video,
    #         fabric.logger.log_dir,
    #         "test",
    #         mask_velocities=args.mask_vel,
    #         vector_env_idx=0,
    #     )()
    #     test(agent, test_env, fabric, args)


def trainer(
    args: MuzeroArgs,
    world_collective: TorchCollective,
    buffer_trainers_collective: TorchCollective,
    players_trainer_collective: TorchCollective,
    optimization_pg,
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
    env = make_env(args.env_id, 0, 0, False, None, frame_stack=1)()
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = RecurrentMuzero(num_actions=env.action_space.n)

    # Define the agent and the optimizer and setup them with Fabric
    optimizer = Adam(agent.parameters(), lr=args.lr, eps=1e-4, weight_decay=args.weight_decay)
    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Send weights to rank-1, a.k.a. the first player
    print("TRAINER SENDING INITIAL PARAMS")
    if players_trainer_collective.rank == players_trainer_collective.world_size - 1:
        players_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()),
            src=world_collective.world_size - 1,
        )
        print("OK! TRAINER SENT INITIAL PARAMS")

    # Linear learning rate scheduler
    if args.anneal_lr:
        from torch.optim.lr_scheduler import StepLR

        scheduler = StepLR(optimizer=optimizer, step_size=args.decay_period, gamma=args.decay_factor)

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

    while True:
        # Wait for data
        print("TRAINER WAITING FOR DATA")
        received_batch = [None]
        buffer_trainers_collective.scatter_object_list(received_batch, None, src=0)
        print("OK!! TRAINER RECEIVED DATA")
        batch = received_batch[0]
        # if not isinstance(batch, Trajectory) and batch == -1:
        #     # Last Checkpoint
        #     if global_rank == 0:
        #         state = {
        #             "agent": agent.state_dict(),
        #             "optimizer": optimizer.state_dict(),
        #             "args": asdict(args),
        #             "update_step": update,
        #             "scheduler": scheduler.state_dict() if args.anneal_lr else None,
        #         }
        #         fabric.call("on_checkpoint_trainer", players_trainer_collective=players_trainer_collective, state=state)
        #     return
        # data = make_tensordict(data, device=device)
        update += 1

        for _ in range(args.update_epochs):
            # batch = buffer.sample(batch_size=batch_size, sequence_length=sequence_length)

            target_rewards = batch["rewards"].squeeze()
            target_values = batch["values"].squeeze()
            target_policies = batch["policies"].squeeze()
            observations = batch["observations"].squeeze(2)  # shape should be (L, N, C, H, W)
            actions = batch["actions"].squeeze(2)

            hidden_state_0, policy_0, value_0 = agent.initial_inference(
                observations[0]
            )  # in shape should be (N, C, H, W)
            # Policy loss
            pg_loss = policy_loss(policy_0, target_policies[0])
            # Value loss
            v_loss = value_loss(value_0, target_values[0])
            # Reward loss
            r_loss = torch.tensor(0.0, device=device)

            for sequence_idx in range(1, args.chunk_sequence_len):
                hidden_states, rewards, policies, values = agent.recurrent_inference(
                    actions[sequence_idx].unsqueeze(0).to(dtype=torch.float32), hidden_state_0
                )  # action should be (1, N, 1)
                # Policy loss
                pg_loss += policy_loss(policies.squeeze(), target_policies[sequence_idx])
                # Value loss
                v_loss += value_loss(values.squeeze(), target_values[sequence_idx])
                # Reward loss
                r_loss += reward_loss(rewards.squeeze(), target_rewards[sequence_idx])

            # Equation (1) in the paper, the regularization loss is handled by `weight_decay` in the optimizer
            loss = pg_loss + v_loss + r_loss

            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            print("UPDATING")
            optimizer.step()

            # Update metrics
            aggregator.update("Loss/policy_loss", pg_loss.detach())
            aggregator.update("Loss/value_loss", v_loss.detach())
            aggregator.update("Loss/entropy_loss", r_loss.detach())

        # Send updated weights to the player
        metrics = aggregator.compute()
        aggregator.reset()
        print("TRAINER SENDING UPDATED PARAMS")
        if players_trainer_collective.rank == players_trainer_collective.world_size - 1:
            if args.anneal_lr:
                metrics["Info/learning_rate"] = scheduler.get_last_lr()[0]
            else:
                metrics["Info/learning_rate"] = args.lr

            # players_trainer_collective.broadcast_object_list(
            #    [metrics], src=world_collective.world_size - 1
            # )  # Broadcast metrics: fake send with object list between players and trainer
            players_trainer_collective.broadcast(
                torch.nn.utils.convert_parameters.parameters_to_vector(agent.parameters()),
                src=world_collective.world_size - 1,
            )
        print("OK!! TRAINER SENT UPDATED PARAMS")

        if args.anneal_lr:
            scheduler.step()

        # Checkpoint model on rank-0: send it everything
        if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run:
            if global_rank == 1:
                state = {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": asdict(args),
                    "update_step": update,
                    "scheduler": scheduler.state_dict() if args.anneal_lr else None,
                }
                fabric.call("on_checkpoint_trainer", players_trainer_collective=players_trainer_collective, state=state)


def buffer(
    args: MuzeroArgs,
    world_collective: TorchCollective,
    buffer_players_collective: TorchCollective,
    buffer_trainers_collective: TorchCollective,
):
    """Buffer process.
    Receive data from the players and send it to the trainers.

    Args:
        world_collective: collective for the world group.
        buffer_players_collective: collective for the players group and the buffer, for receiving the collected data.
            The first rank is for the buffer.
        buffer_trainers_collective: collective for the buffer and the trainers, for sending the collected data.
            The first rank is for the buffer.
    """
    rb = TrajectoryReplayBuffer(max_num_trajectories=args.buffer_capacity)
    while True:
        print(f"Buffer {world_collective.rank}: Waiting for collected data from players...")
        steps_data = [None for _ in range(buffer_players_collective.world_size)]
        buffer_players_collective.gather_object(None, steps_data, dst=0)  # gather_object uses global rank
        for traj in steps_data:
            rb.add(traj)

        print(f"Buffer {world_collective.rank}: Sending collected data to trainers...")
        sampled_buffers = [
            rb.sample(args.chunks_per_batch, args.chunk_sequence_len)
            for _ in range(buffer_trainers_collective.world_size - 1)
        ]
        buffer_trainers_collective.scatter_object_list(
            [None], [None] + sampled_buffers, src=0
        )  # scatter_object_list uses global rank


@register_algorithm(decoupled=True)
def main():
    # Ranks semantic:
    # rank-0 -> buffer
    # rank-1, ..., rank-num_players -> players
    # rank-(num_players+1), ..., rank-(num_players+num_trainers) -> trainers

    parser = HfArgumentParser(MuzeroArgs)
    args: MuzeroArgs = parser.parse_args_into_dataclasses()[0]

    devices = os.environ.get("LT_DEVICES", None)
    if devices is None or devices in ("1", "2"):
        raise RuntimeError(
            "Please run the script with the number of devices greater than 2: "
            "`lightning run model --devices=3 sheeprl.py ...`"
        )

    world_collective = TorchCollective()
    world_collective.setup(
        backend="nccl" if os.environ.get("LT_ACCELERATOR", None) in ("gpu", "cuda") else "gloo",
        timeout=timedelta(days=1),
    )

    # Create a global group, assigning it to the collective: used by the player to exchange
    # collected experiences with the trainers
    world_collective.create_group(timeout=timedelta(days=1))
    global_rank = world_collective.rank

    # Trainers collective to train the model in paraller with all the other trainers, needed for the optimization_pg
    trainers_collective = TorchCollective()
    trainers_collective.create_group(
        ranks=list(range(args.num_players + 1, args.num_players + args.num_trainers + 1)),
        timeout=timedelta(days=1),
    )
    optimization_pg = trainers_collective.group

    # Players-Buffer collective, to share data between players and buffer
    buffer_players_collective = TorchCollective()
    buffer_players_collective.create_group(ranks=list(range(0, args.num_players + 1)), timeout=timedelta(days=1))

    # Trainers-Buffer collective, to share data between buffer and trainers
    buffer_trainers_collective = TorchCollective()
    buffer_trainers_collective.create_group(
        ranks=[0] + list(range(args.num_players + 1, args.num_players + args.num_trainers + 1)),
        timeout=timedelta(days=1),
    )

    # Trainer-Players collective, to share the updated parameters between trainer and players
    players_trainer_collective = TorchCollective()
    players_trainer_collective.create_group(
        ranks=list(range(1, args.num_players + 1)) + [args.num_players + args.num_trainers],
        timeout=timedelta(days=1),
    )

    if global_rank == 0:
        buffer(args, world_collective, buffer_players_collective, buffer_trainers_collective)
    elif 1 <= global_rank <= args.num_players:
        player(args, world_collective, buffer_players_collective, players_trainer_collective)
    else:
        trainer(
            args,
            world_collective,
            buffer_trainers_collective,
            players_trainer_collective,
            optimization_pg,
        )
