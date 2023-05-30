import os
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timedelta
import math
from typing import Optional

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
        self.value_sum: float = 0.0
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

    def mcts(self, network: torch.nn.Module, num_simulations: int, discount: float = 0.997) -> dict[int, "Node"]:
        """Runs MCTS for num_simulations"""
        # Initialize the hidden state and compute the actor's policy logits
        hidden_state, policy_logits = network.initial_inference(self.image)

        # Use the actor's policy to initialize the children of the node.
        # The policy results are used as prior probabilities.
        policy = {a: math.exp(policy_logits[a]) for a in range(len(policy_logits))}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            self.children[action] = Node(p / policy_sum)
        self.add_exploration_noise()

        # Expand until an unvisited node (i.e. not yet expanded) is reached
        min_max_stats = MinMaxStats()

        for _ in range(num_simulations):
            node = self
            search_path = [node]

            while node.expanded():
                # Select the child with the highest UCB score
                _, imagined_action, child = max(
                    (ucb_score(
                        parent=node,
                        child=child,
                        min_max_stats=min_max_stats,
                        discount=discount,
                        pb_c_base = 19652,
                        pb_c_init = 1.25
                    ),
                    action, child
                )
                for action, child in node.children.items()
                )
                search_path.append(node)

            # When a path from the starting node to an unvisited node is found, expand the unvisited node
            parent = search_path[-2]
            hidden_state, reward, policy_logits, value = network.recurrent_inference(parent.hidden_state, imagined_action)
            node.hidden_state = hidden_state
            node.reward = reward
            policy = {a_id: math.exp(a_logit) for a_id, a_logit in enumerate(policy_logits)}
            policy_sum = sum(policy.values())
            for action, p in policy.items():
                node.children[action] = Node(p / policy_sum)

            # Backpropagate the search path to update the nodes' statistics
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.value())

                value = node.reward + discount * value


def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats, discount: float, pb_c_base: float, pb_c_init: float) -> float:
    """Computes the UCB score of a child node relative to its parent, using the min-max bounds on the value function to
    normalize the node value."""
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = child.reward + discount * min_max_stats.normalize(child.value())
    else:
        value_score = 0
    return prior_score + value_score


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

    # Receive the first weights from the rank-0, a.k.a. the trainer
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=0)
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
    rb = TrajectoryReplayBuffer(args.rollout_steps, args.num_envs, device=device)
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

    for update in range(1, num_updates + 1):
        # reset the episode at every update
        with device:
            # Get the first environment observation and start the optimization
            obs: torch.Tensor = torch.tensor(envs.reset(seed=args.seed)[0], device=device)

        for step in range(0, args.max_trajectory_len):
            global_step += 1
            node = Node(prior=0, image=obs)

            # start MCTS
            node.mcts(network, num_simulations, discount)

            # Select action based on the visit count distribution and the temperature
            visits_count = torch.tensor([child.visit_count for child in node.children.values()])
            temperature = visit_softmax_temperature(training_steps=network.training_steps())
            visits_count = visits_count / temperature
            action = torch.distributions.Categorical(logits=visits_count).sample()

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            # Store the current step data
            step_data = TensorDict(
                {
                    "visits_counts": visits_count,
                    "actions": action,
                    "observations": obs,
                    "rewards": reward,
                    "node_values": node.value,
                },
                batch_size=1
            )

            # Update observation
            with device:
                obs = torch.tensor(next_obs)

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-{rank}: global_step={global_step}, reward={agent_final_info['episode']['r'][0]}"
                        )
                        aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                        aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))
            if done or truncated:
                break

        # After episode is done
        # TODO Add trajectory to the buffer and broadcast

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        # TODO Send data to the training agent
        # Collecting from all players
        perm = torch.randperm(local_data.shape[0], device=device)
        chunks = local_data[perm].split(chunks_sizes)
        world_collective.scatter_object_list([None], [None] + chunks, src=0)

        # Gather metrics from the trainer to be plotted
        metrics = [None]
        player_trainer_collective.broadcast_object_list(metrics, src=0)

        # Wait the trainer to finish
        player_trainer_collective.broadcast(flattened_parameters, src=0)

        # Convert back the parameters
        torch.nn.utils.convert_parameters.vector_to_parameters(
            flattened_parameters, list(actor.parameters()) + list(critic.parameters())
        )

        # Log metrics
        aggregator.update("Time/step_per_second", int(global_step / (time.time() - start_time)))
        fabric.log_dict(metrics[0], global_step)
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint Model # TODO move to the trainer
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
