import math
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric

from sheeprl.algos.muzero.agent import MuzeroAgent
from sheeprl.utils.utils import inverse_symsqrt, symsqrt, two_hot_decoder, two_hot_encoder


def support_to_scalar(support: torch.Tensor, support_range: int) -> np.ndarray:
    """Converts a support representation to a scalar."""
    if isinstance(support, torch.Tensor):
        support = support.numpy()
    return inverse_symsqrt(two_hot_decoder(support, support_range))


def scalar_to_support(scalar: np.ndarray, support_range: int) -> torch.Tensor:
    """Converts a scalar to a support representation."""
    return two_hot_encoder(symsqrt(scalar), support_range)


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
        env = gym.make(env_id, render_mode="rgb_array")  # , frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and vector_env_idx == 0 and idx == 0:
            env = gym.experimental.wrappers.RecordVideoV0(
                env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
            )
        # env = AtariPreprocessing(
        #     env,
        #     screen_size=screen_size,
        #     frame_skip=action_repeat,
        #     grayscale_obs=False,
        #     grayscale_newaxis=True,
        #     scale_obs=True,
        # )
        # env = gym.wrappers.TransformObservation(env, lambda obs: obs.transpose(2, 0, 1))
        # env.observation_space = gym.spaces.Box(
        #     0, 1, (env.observation_space.shape[2], screen_size, screen_size), np.float32
        # )
        # if frame_stack > 0:
        #     env = gym.wrappers.FrameStack(env, frame_stack)
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

    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """Add exploration noise to the prior probabilities."""
        actions = list(range(len(self.children)))
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MCTS:
    def __init__(
        self,
        agent: MuzeroAgent,
        num_simulations: int,
        gamma: float = 0.997,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25,
        support_size: int = 300,
        pb_c_base: float = 19652.0,
        pb_c_init: float = 1.25,
    ):
        self.agent = agent
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.support_size = support_size
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init

    def search(self, root, observation: np.ndarray):
        """Runs MCTS for num_simulations and modifies the root node in place with the result."""
        # Initialize the hidden state and compute the actor's policy logits
        hidden_state, policy_logits, _ = self.agent.initial_inference(torch.as_tensor(observation))
        root.hidden_state = hidden_state.reshape(1, 1, -1)

        # Use the actor's policy to initialize the children of the node.
        # The policy results are used as prior probabilities.
        normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
        root.expand(normalized_policy.squeeze().tolist())
        root.add_exploration_noise(self.dirichlet_alpha, self.exploration_fraction)

        # Expand until an unvisited node (i.e. not yet expanded) is reached
        min_max_stats = MinMaxStats()

        for _ in range(self.num_simulations):
            search_path, imagined_action = self.rollout(root, min_max_stats)

            # When a path from the starting node to an unvisited node is found, expand the unvisited node
            node = search_path[-1]
            parent = search_path[-2]
            hidden_state, reward, policy_logits, value = self.agent.recurrent_inference(
                torch.tensor([imagined_action]).view(1, 1).to(device=parent.hidden_state.device, dtype=torch.float32),
                parent.hidden_state,
            )
            value = support_to_scalar(torch.nn.functional.softmax(value, dim=-1), self.support_size).item()
            node.hidden_state = hidden_state
            node.reward = support_to_scalar(torch.nn.functional.softmax(reward, dim=-1), self.support_size).item()
            normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
            for action in range(normalized_policy.numel()):
                node.children.append(Node(normalized_policy.squeeze()[action].item()))

            # Backpropagate the search path to update the nodes' statistics
            self.backpropagate(search_path, value, min_max_stats)

    def rollout(self, root: Node, min_max_stats: MinMaxStats) -> tuple[list[Node], int]:
        if not root.expanded():
            raise RuntimeError("Cannot rollout from an unexpanded root!")
        search_path = [root]
        node = root
        while node.expanded():
            # Select the child with the highest UCB score
            ucb_scores = self.ucb_score(
                parent=node,
                min_max_stats=min_max_stats,
            )

            ucb_scores = ucb_scores + 1e-7 * np.random.random(
                ucb_scores.shape
            )  # Add tiny bit of randomness for tie break
            imagined_action = np.argmax(ucb_scores)
            child = node.children[imagined_action]
            search_path.append(child)
            node = child
        return search_path, imagined_action

    def ucb_score(self, parent: Node, min_max_stats: MinMaxStats) -> np.ndarray:
        """Computes the UCB score of a child node relative to its parent,
        using the min-max bounds on the value function to
        normalize the node value."""
        children_visit_counts = np.array([child.visit_count for child in parent.children])
        children_values = np.array([child.value() for child in parent.children])
        children_priors = np.array([child.prior for child in parent.children])
        children_rewards = np.array([child.reward for child in parent.children])
        pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (children_visit_counts + 1)

        prior_score = pb_c * children_priors
        value_score = children_rewards + self.gamma * min_max_stats.normalize(children_values)
        min_value = np.min(value_score)
        max_value = np.max(value_score) + 1e-7
        value_score = (value_score - min_value) / (max_value - min_value)  # Normalize to be in [0, 1] range
        return prior_score + value_score

    def backpropagate(self, search_path: list[Node], value: float, min_max_stats: MinMaxStats):
        for visited_node in reversed(search_path):
            visited_node.value_sum += value
            visited_node.visit_count += 1
            min_max_stats.update(visited_node.value())
            value = visited_node.reward + self.gamma * value


@torch.no_grad()
def test(agent: MuzeroAgent, env: gym.Env, fabric: Fabric):  # , args: MuzeroArgs):
    # agent.eval()
    # done = False
    # cumulative_rew = 0
    # next_obs = torch.tensor(
    #     np.array(env.reset(seed=args.seed)[0]), device=fabric.device, dtype=torch.float32
    # ).unsqueeze(0)
    # while not done:
    #     # Act greedly through the environment
    #     node = Node(prior=0, image=next_obs)
    #
    #     # start MCTS
    #     node.mcts(agent, args.num_simulations, args.gamma, args.dirichlet_alpha, args.exploration_fraction)
    #
    #     # Select action based on the visit count distribution and the temperature
    #     visits_count = torch.tensor([child.visit_count for child in node.children.values()])
    #     visits_count = visits_count
    #     action = torch.distributions.Categorical(logits=visits_count).sample()
    #     print(f"Mcts completed, action: {action}")
    #     # Single environment step
    #     next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
    #     cumulative_rew += reward
    #
    #     if args.dry_run:
    #         done = True
    # fabric.print("Test - Reward:", cumulative_rew)
    # fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    # env.close()
    pass
