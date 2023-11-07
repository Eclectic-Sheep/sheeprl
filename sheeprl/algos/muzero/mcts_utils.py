import math
from typing import Optional

import numpy as np
import torch


class Node:
    """A Node in the MCTS tree.

    It is used to store the statistics of the node and its children, in order to progress the MCTS search and store the
    backpropagation values.

    Attributes:
        prior (float): The prior probability of the node.
        hidden_state (Optional[torch.Tensor]): The hidden state of the node.
        reward (float): The reward of the node.
        value_sum (float): The sum of the values of the node.
        visit_count (int): The number of times the node has been visited.
        children (list[Node]): The children of the node, one for each action where actions are represented
            as integers running from 0 to num_actions - 1.
    """

    def __init__(self, prior: float):
        """A Node in the MCTS tree.

        Args:
            prior (float): The prior probability of the node.
            image (torch.Tensor): The image of the node.
                The image is used to create the initial hidden state for the network in MCTS.
                Hence, it is needed only for the starting node of every MCTS search.
        """
        self.prior: float = prior
        self.hidden_state: Optional[torch.Tensor] = None
        self.reward: float = 0.0
        self.value_sum: float = 0.0
        self.visit_count: int = 0
        self.children: list[Node] = []
        self.imagined_action: Optional[int] = None

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
            return self.value_sum
        return self.value_sum / self.visit_count

    def expand(self, normalized_policy_list: list[float]):
        """Expands the node by creating all its children.

        Args:
            normalized_policy (torch.Tensor): The policy output of the network, normalized to be a probability
                distribution.
        """
        for prior in normalized_policy_list:
            self.children.append(Node(prior))

    def add_exploration_noise(self, noise: list[float], exploration_fraction: float):
        """Add exploration noise to the prior probabilities."""
        actions = list(range(len(self.children)))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum + 1e-7)
        return value


def rollout(root: Node, pbc_base: float, pbc_init: float, gamma: float, min_max_stats: MinMaxStats) -> list[Node]:
    if not root.expanded():
        raise RuntimeError("Cannot rollout from an unexpanded root!")
    search_path = [root]
    node = root
    while node.expanded():
        # Select the child with the highest UCB score
        ucb_scores = ucb_score(
            parent=node,
            min_max_stats=min_max_stats,
            pbc_base=pbc_base,
            pbc_init=pbc_init,
            gamma=gamma,
        )

        ucb_scores = ucb_scores + 1e-7 * np.random.random(ucb_scores.shape)  # Add tiny bit of randomness for tie break
        imagined_action = np.argmax(ucb_scores)
        node.imagined_action = imagined_action
        child = node.children[imagined_action]
        search_path.append(child)
        node = child
    return search_path


def ucb_score(parent: Node, pbc_base: float, pbc_init: float, gamma: float, min_max_stats: MinMaxStats) -> np.ndarray:
    """Computes the UCB score of a child node relative to its parent,
    using the min-max bounds on the value function to
    normalize the node value."""
    children_visit_counts = np.array([child.visit_count for child in parent.children])
    children_values = np.array([child.value() for child in parent.children])
    children_priors = np.array([child.prior for child in parent.children])
    children_rewards = np.array([child.reward for child in parent.children])
    pb_c = math.log((parent.visit_count + pbc_base + 1) / pbc_base) + pbc_init
    pb_c *= math.sqrt(parent.visit_count) / (children_visit_counts + 1)

    prior_score = pb_c * children_priors
    value_score = children_rewards + gamma * children_values
    value_score = min_max_stats.normalize(value_score)
    return prior_score + value_score


def backpropagate(search_path: list[Node], value: float, gamma: float, min_max_stats: MinMaxStats):
    for visited_node in reversed(search_path):
        visited_node.value_sum += value
        visited_node.visit_count += 1
        min_max_stats.update(visited_node.value())
        value = visited_node.reward + gamma * value
