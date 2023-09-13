import math
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing
from lightning import Fabric
from omegaconf import DictConfig

from sheeprl.algos.muzero.agent import MuzeroAgent
from sheeprl.utils.utils import inverse_symsqrt, two_hot_decoder


def support_to_scalar(support: torch.Tensor, support_range: int) -> torch.Tensor:
    """Converts a support representation to a scalar."""
    return inverse_symsqrt(two_hot_decoder(support, support_range))


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
        children (dict[int, Node]): The children of the node, one for each action where actions are represented
            as integers running from 0 to num_actions - 1.
    """

    def __init__(self, prior: float, image: torch.Tensor = None, device=torch.device):
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

    def value(self) -> torch.Tensor:
        """Returns the value of the node."""
        if self.visit_count == 0:
            return self.value_sum
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
        support_size: int = 300,
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

        for _ in range(num_simulations):
            node = self
            search_path = [node]

            while node.expanded():
                # Select the child with the highest UCB score
                ucb_scores = ucb_score(
                    parent=node,
                    min_max_stats=min_max_stats,
                    gamma=gamma,
                    pb_c_base=19652,
                    pb_c_init=1.25,
                )

                ucb_scores = ucb_scores + 1e-7 * torch.rand(
                    ucb_scores.shape
                )  # Add tiny bit of randomness for tie break
                imagined_action = torch.argmax(ucb_scores)
                child = node.children[imagined_action.item()]
                search_path.append(child)
                node = child

            # When a path from the starting node to an unvisited node is found, expand the unvisited node
            parent = search_path[-2]
            hidden_state, reward, policy_logits, value = agent.recurrent_inference(
                torch.tensor([imagined_action])
                .view(1, 1, -1)
                .to(device=parent.hidden_state.device, dtype=torch.float32),
                parent.hidden_state,
            )
            value = support_to_scalar(torch.nn.functional.softmax(value, dim=-1), support_size)
            node.hidden_state = hidden_state
            node.reward = support_to_scalar(torch.nn.functional.softmax(reward, dim=-1), support_size)
            normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
            for action in range(normalized_policy.numel()):
                node.children[action] = Node(
                    normalized_policy.squeeze()[action].item(), device=normalized_policy.device
                )

            # Backpropagate the search path to update the nodes' statistics
            for visited_node in reversed(search_path):
                visited_node.value_sum += value.squeeze()
                visited_node.visit_count += 1
                min_max_stats.update(visited_node.value())
                value = visited_node.reward + gamma * value


def ucb_score(parent: Node, min_max_stats: MinMaxStats, gamma: float, pb_c_base: float, pb_c_init: float) -> float:
    """Computes the UCB score of a child node relative to its parent, using the min-max bounds on the value function to
    normalize the node value."""
    children_visit_counts = torch.tensor([child.visit_count for child in parent.children.values()], dtype=torch.float32)
    children_values = torch.tensor([child.value() for child in parent.children.values()], dtype=torch.float32)
    children_priors = torch.tensor([child.prior for child in parent.children.values()], dtype=torch.float32)
    children_rewards = torch.tensor([child.reward for child in parent.children.values()], dtype=torch.float32)
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (children_visit_counts + 1)

    prior_score = pb_c * children_priors
    value_score = children_rewards + gamma * min_max_stats.normalize(children_values)
    min_value = torch.min(value_score)
    max_value = torch.max(value_score) + 1e-7  # TODO add parent score and epsilon
    value_score = (value_score - min_value) / (max_value - min_value)  # Normalize to be in [0, 1] range
    return prior_score + value_score


@torch.no_grad()
def test(agent: MuzeroAgent, env: gym.Env, fabric: Fabric, cfg: DictConfig):
    agent.eval()
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(np.array(env.reset(seed=cfg.seed)[0]), device=fabric.device, dtype=torch.float32).unsqueeze(
        0
    )
    while not done:
        # Act greedly through the environment
        node = Node(prior=0, image=next_obs)

        # start MCTS
        node.mcts(agent, cfg.num_simulations, cfg.gamma, cfg.dirichlet_alpha, cfg.exploration_fraction)

        # Select action based on the visit count distribution and the temperature
        visits_count = torch.tensor([child.visit_count for child in node.children.values()])
        visits_count = visits_count
        action = torch.distributions.Categorical(logits=visits_count).sample()
        print(f"Mcts completed, action: {action}")
        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        cumulative_rew += reward

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
