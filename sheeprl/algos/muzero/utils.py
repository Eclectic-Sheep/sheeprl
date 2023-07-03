import math
import os
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing
from lightning import Fabric

from sheeprl.algos.muzero.agent import MuzeroAgent
from sheeprl.algos.muzero.args import MuzeroArgs


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
        if self.maximum > self.minimum != -float("inf") and self.maximum != float("inf"):
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

    def __init__(self, prior: float, image: torch.Tensor = None, device=Union["cpu", torch.device]):
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
        self.reward: torch.tensor = torch.tensor(0.0, device=device)
        self.value_sum: torch.tensor = torch.tensor(0.0, device=device)
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
        normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
        for action in range(normalized_policy.numel()):
            self.children[action] = Node(normalized_policy[:, action].item(), device=hidden_state.device)
        self.add_exploration_noise(dirichlet_alpha, exploration_fraction)

        # Expand until an unvisited node (i.e. not yet expanded) is reached
        min_max_stats = MinMaxStats()

        for sim_num in range(num_simulations):
            node = self
            search_path = [node]

            while node.expanded():
                # Select the child with the highest UCB score
                ucb_scores = [
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
                ]
                _, imagined_action, child = max(ucb_scores)
                search_path.append(child)
                node = child

            # When a path from the starting node to an unvisited node is found, expand the unvisited node
            parent = search_path[-2]
            hidden_state, reward, policy_logits, value = agent.recurrent_inference(
                torch.tensor([imagined_action])
                .view(1, 1, 1)
                .to(device=parent.hidden_state.device, dtype=torch.float32),
                parent.hidden_state,
            )
            node.hidden_state = hidden_state
            node.reward = reward
            normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
            for action in range(normalized_policy.numel()):
                node.children[action] = Node(
                    normalized_policy.squeeze()[action].item(), device=normalized_policy.device
                )

            # Backpropagate the search path to update the nodes' statistics
            for node in reversed(search_path):
                node.value_sum += value.squeeze()
                node.visit_count += 1
                min_max_stats.update(node.value())

                value = node.reward + gamma * value


def ucb_score(
    parent: Node, child: Node, min_max_stats: MinMaxStats, gamma: float, pb_c_base: float, pb_c_init: float
) -> float:
    """Computes the UCB score of a child node relative to its parent, using the min-max bounds on the value function to
    normalize the node value."""
    device = parent.hidden_state.device
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = child.reward.squeeze() + gamma * min_max_stats.normalize(child.value())
    else:
        value_score = torch.tensor(0.0, device=device)
    return prior_score + value_score


@torch.no_grad()
def test(agent: MuzeroAgent, env: gym.Env, fabric: Fabric, args: MuzeroArgs):
    agent.eval()
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(
        np.array(env.reset(seed=args.seed)[0]), device=fabric.device, dtype=torch.float32
    ).unsqueeze(0)
    while not done:
        # Act greedly through the environment
        node = Node(prior=0, image=next_obs)

        # start MCTS
        node.mcts(agent, args.num_simulations, args.gamma, args.dirichlet_alpha, args.exploration_fraction)

        # Select action based on the visit count distribution and the temperature
        visits_count = torch.tensor([child.visit_count for child in node.children.values()])
        visits_count = visits_count
        action = torch.distributions.Categorical(logits=visits_count).sample()
        print(f"Mcts completed, action: {action}")
        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        cumulative_rew += reward

        if args.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
