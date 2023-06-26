import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from lightning import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
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
                    [
                        ucb_score(
                            parent=node,
                            child=child_node,
                            min_max_stats=min_max_stats,
                            gamma=gamma,
                            pb_c_base=19652,
                            pb_c_init=1.25,
                        ),
                        action,
                        child_node,
                    ]
                    for action, child_node in node.children.items()
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
                node.children[action] = Node(normalized_policy.squeeze()[action].item())

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


@register_algorithm()
def main():
    parser = HfArgumentParser(MuzeroArgs)
    args: MuzeroArgs = parser.parse_args_into_dataclasses()[0]

    # Initialize Fabric
    fabric = Fabric(callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set logger only on rank-0 but share the logger directory: since we don't know
    # what is happening during the `fabric.save()` method, at least we assure that all
    # ranks save under the same named folder.
    # As a plus, rank-0 sets the time uniquely for everyone
    world_collective = TorchCollective()
    if fabric.world_size > 1:
        world_collective.setup()
        world_collective.create_group()
    if rank == 0:
        root_dir = (
            args.root_dir
            if args.root_dir is not None
            else os.path.join("logs", "muzero", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        run_name = (
            args.run_name
            if args.run_name is not None
            else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        )
        logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
        fabric._loggers = [logger]
        log_dir = logger.log_dir
        fabric.logger.log_hyperparams(asdict(args))
        if fabric.world_size > 1:
            world_collective.broadcast_object_list([log_dir], src=0)
    else:
        data = [None]
        world_collective.broadcast_object_list(data, src=0)
        log_dir = data[0]
        os.makedirs(log_dir, exist_ok=True)

    # Environment setup
    env = make_env(
        args.env_id,
        args.seed + rank,
        rank,
        args.capture_video,
        logger.log_dir,
        "train",
        frame_stack=1,
        vector_env_idx=rank,
    )()
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action space is supported"

    # Create the model
    agent = RecurrentMuzero(num_actions=env.action_space.n).to(device)
    optimizer = Adam(agent.parameters(), lr=args.lr, eps=1e-4, weight_decay=args.weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)

    # Metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(),
                "Game/ep_len_avg": MeanMetric(),
                "Time/step_per_second": MeanMetric(),
                "Loss/value_loss": MeanMetric(),
                "Loss/policy_loss": MeanMetric(),
                "Loss/reward_loss": MeanMetric(),
            }
        )

    # Local data
    buffer_size = args.buffer_capacity // int(args.num_envs * fabric.world_size) if not args.dry_run else 1
    rb = TrajectoryReplayBuffer(max_num_trajectories=buffer_size)  # TODO device=device, memmap=args.memmap_buffer)

    # Global variables
    start_time = time.perf_counter()
    num_updates = int(args.total_steps // args.num_players) if not args.dry_run else 1
    args.learning_starts = args.learning_starts // args.num_envs if not args.dry_run else 0

    with torch.no_grad():
        for update_step in range(1, num_updates + 1):
            # reset the episode at every update
            with device:
                # Get the first environment observation and start the optimization
                obs: torch.Tensor = torch.tensor(env.reset(seed=args.seed)[0], device=device).view(
                    1, 3, 64, 64
                )  # shape (1, C, H, W)

            steps_data = None
            for trajectory_step in range(0, args.max_trajectory_len):
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
                        device=device,
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

                if "episode" in info:
                    fabric.print(f"Rank-{rank}: update_step={update_step}, reward={info['episode']['r'][0]}")
                    aggregator.update("Rewards/rew_avg", info["episode"]["r"][0])
                    aggregator.update("Game/ep_len_avg", info["episode"]["l"][0])

                if done or truncated:
                    break
                else:
                    with device:
                        obs = torch.tensor(next_obs).view(1, 3, 64, 64)
            rb.add(trajectory=steps_data)

        if update_step >= args.learning_starts - 1:
            training_steps = args.learning_starts if update_step == args.learning_starts - 1 else 1
            for _ in range(training_steps):
                # We sample one time to reduce the communications between processes
                data = rb.sample(args.chunks_per_batch, args.chunk_sequence_len)

                target_rewards = data["rewards"].squeeze()
                target_values = data["values"].squeeze(-1)
                target_policies = data["policies"].squeeze()
                observations = data["observations"].squeeze(2)  # shape should be (L, N, C, H, W)
                actions = data["actions"].squeeze(2)

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
                    v_loss += value_loss(values, target_values[sequence_idx].unsqueeze(0))
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
                aggregator.update("Loss/reward_loss", r_loss.detach())

        aggregator.update("Time/step_per_second", int(update_step / (time.perf_counter() - start_time)))
        fabric.log_dict(aggregator.compute(), update_step)
        aggregator.reset()

    env.close()


if __name__ == "__main__":
    main()
