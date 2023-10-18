import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing
from lightning import Fabric
from omegaconf import DictConfig

import sheeprl.algos.muzero.ctree.cytree as tree
from sheeprl.algos.muzero.agent import MuzeroAgent
from sheeprl.utils.utils import inverse_symsqrt, symsqrt, two_hot_decoder, two_hot_encoder


def scalar_to_support(scalar: torch.Tensor, support_size: int):
    """Convert a scalar representation to a support."""
    return two_hot_encoder(symsqrt(scalar), support_size)


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

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, prior: torch.Tensor, image: torch.Tensor = None):
        """A Node in the MCTS tree.

        Args:
            prior (torch.Tensor): The prior probability of the node.
            image (torch.Tensor): The image of the node.
                The image is used to create the initial hidden state for the network in MCTS.
                Hence, it is needed only for the starting node of every MCTS search.
        """
        self.prior: torch.Tensor = prior
        self.image: torch.Tensor = image
        self._device = prior.device
        self._dtype = prior.dtype

        self.hidden_state: Optional[torch.Tensor] = None
        self.reward: torch.tensor = torch.tensor(0.0, dtype=self._dtype, device=self._device)
        self.value_sum: torch.tensor = torch.tensor(0.0, dtype=self._dtype, device=self._device)
        self.visit_count: torch.tensor = torch.tensor(0.0, dtype=self._dtype, device=self._device)
        self.children: dict[int, Node] = {}

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

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


def add_exploration_noise(
    node: Node, noise_distribution: torch.distributions.dirichlet.Dirichlet, exploration_fraction: float
):
    """Add exploration noise to the prior probabilities."""
    actions = list(node.children.keys())
    noise = noise_distribution.rsample()
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - exploration_fraction) + n * exploration_fraction


def mcts(
    root: Node,
    agent: torch.nn.Module,
    num_simulations: int,
    noise_distribution: torch.distributions.dirichlet.Dirichlet,
    gamma: float = 0.997,
    exploration_fraction: float = 0.25,
    support_size: int = 300,
):
    """Runs MCTS for num_simulations"""
    # Initialize the hidden state and compute the actor's policy logits
    with torch.no_grad():
        hidden_state, policy_logits, _ = agent.initial_inference(root.image)
        root.hidden_state = hidden_state

        device = hidden_state.device
        dtype = hidden_state.dtype
        root.image.shape[0]

        # Use the actor's policy to initialize the children of the node.
        # The policy results are used as prior probabilities.
        normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
        root.children = {action: Node(normalized_policy[:, action]) for action in range(normalized_policy.shape[-1])}
        add_exploration_noise(root, noise_distribution, exploration_fraction)

        # Expand until an unvisited node (i.e. not yet expanded) is reached
        min_max_stats = MinMaxStats()

        for _ in range(num_simulations):
            node = root
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

                ucb_scores += 1e-7 * torch.rand(
                    ucb_scores.shape, device=device
                )  # Add tiny bit of randomness for tie break
                imagined_action = torch.argmax(ucb_scores)
                child = node.children[imagined_action.item()]
                search_path.append(child)
                node = child

            # When a path from the starting node to an unvisited node is found, expand the unvisited node
            parent = search_path[-2]
            imagined_action = imagined_action.view(1, 1, -1).to(device=device, dtype=dtype)
            hidden_state, reward, policy_logits, value = agent.recurrent_inference(
                imagined_action,
                parent.hidden_state,
            )
            value = support_to_scalar(torch.nn.functional.softmax(value, dim=-1), support_size)
            node.hidden_state = hidden_state
            node.reward = support_to_scalar(torch.nn.functional.softmax(reward, dim=-1), support_size)
            normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
            for action in range(normalized_policy.numel()):
                node.children[action] = Node(normalized_policy.squeeze()[action])

            # Backpropagate the search path to update the nodes' statistics
            for visited_node in reversed(search_path):
                visited_node.value_sum += value.squeeze()
                visited_node.visit_count += 1
                min_max_stats.update(visited_node.value().item())
                value = visited_node.reward + gamma * value


def ucb_score(
    parent: Node, min_max_stats: MinMaxStats, gamma: float, pb_c_base: float, pb_c_init: float
) -> torch.Tensor:
    """Computes the UCB score of a child node relative to its parent, using the min-max bounds on the value function to
    normalize the node value."""
    device = parent.device
    dtype = parent.dtype
    children = parent.children.values()
    children_visit_counts = torch.tensor([child.visit_count for child in children], dtype=dtype, device=device)
    children_values = torch.tensor([child.value() for child in children], dtype=dtype, device=device)
    children_priors = torch.tensor([child.prior for child in children], dtype=dtype, device=device)
    children_rewards = torch.tensor([child.reward for child in children], dtype=dtype, device=device)

    pb_c = torch.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c = pb_c * torch.div(torch.sqrt(parent.visit_count), (children_visit_counts + 1))

    prior_score = pb_c * children_priors

    min_max_normalized = min_max_stats.normalize(children_values)
    min_value = torch.min(children_rewards + gamma * min_max_normalized)
    max_value = torch.max(children_rewards + gamma * min_max_normalized) + 1e-7
    value_score = (children_rewards + gamma * min_max_normalized - min_value) / (max_value - min_value)
    return prior_score + value_score


@torch.no_grad()
def test(agent: MuzeroAgent, env: gym.Env, fabric: Fabric, cfg: DictConfig):
    agent.eval()
    torch.tensor(np.array(env.reset(seed=cfg.seed)[0]), device=fabric.device, dtype=torch.float32).unsqueeze(0)
    """
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
    """


class MCTS:
    def __init__(self, num_simulations, value_delta_max, device, pb_c_base, pb_c_init, discount, support_range):
        self.num_simulations = num_simulations
        self.value_delta_max = value_delta_max
        self.device = device
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.support_range = support_range

    def search(self, roots, model, hidden_state_roots):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
        Parameters
        ----------
        roots: Any
            a batch of expanded root nodes
        hidden_state_roots: list
            the hidden states of the roots
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self.device
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x embedding_size

            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.value_delta_max)

            for index_simulation in range(self.num_simulations):
                hidden_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.ResultsWrapper(num)
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(
                    roots, self.pb_c_base, self.pb_c_init, self.discount, min_max_stats_lst, results
                )

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()

                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).long()

                # evaluation for leaf nodes
                # if self.config.amp_type == 'torch_amp':
                #     with autocast():
                #         hidden_state_nodes, reward, policy_logits, value =
                #         model.recurrent_inference(last_actions, hidden_states)
                # else:
                hidden_state_nodes, reward, policy_logits, value = model.recurrent_inference(
                    last_actions.view(num, 1), hidden_states
                )

                value_pool = (
                    support_to_scalar(torch.nn.functional.softmax(value, dim=-1), self.support_range)
                    .reshape(-1)
                    .tolist()
                )
                policy_logits_pool = policy_logits.tolist()
                reward_pool = (
                    support_to_scalar(torch.nn.functional.softmax(reward, dim=-1), self.support_range)
                    .reshape(-1)
                    .tolist()
                )

                hidden_state_pool.append(hidden_state_nodes.tolist())
                hidden_state_index_x += 1

                # backpropagation along the search path to update the attributes
                tree.batch_back_propagate(
                    hidden_state_index_x,
                    self.discount,
                    value_pool,
                    reward_pool,
                    policy_logits_pool,
                    min_max_stats_lst,
                    results,
                )
