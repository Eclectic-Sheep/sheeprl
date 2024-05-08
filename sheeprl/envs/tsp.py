import itertools
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import pygame


class TSP(gym.Env):
    """Travelling Salesman Problem environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, n_nodes, render_mode=None, screen_size=600, reward_mode="sparse", seed=None):
        """Create a TSP environment with n_nodes nodes."""
        np.random.seed(seed)
        self.n_nodes = n_nodes
        self.reward_mode = reward_mode

        if render_mode is None:
            render_mode = "rgb_array"
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode: {render_mode}")
        self.render_mode = render_mode
        self.screen_size = screen_size
        self.window = None
        self.clock = None

        nodes_space = gym.spaces.Box(low=0, high=1, shape=(n_nodes, 3))  # x, y, visited
        edge_space = gym.spaces.MultiBinary(n_nodes * (n_nodes - 1))  # visited or not
        edge_links_space = gym.spaces.MultiDiscrete([[n_nodes, n_nodes]] * (n_nodes * (n_nodes - 1)))

        self.mask_space = gym.spaces.MultiBinary(n_nodes)
        self.first_node_space = gym.spaces.MultiDiscrete([n_nodes])
        self.current_node_space = gym.spaces.MultiDiscrete([n_nodes])

        self.observation_space = gym.spaces.Dict(
            {
                "nodes": nodes_space,
                "edges": edge_space,
                "edge_links": edge_links_space,
                "mask": self.mask_space,
                "first_node": self.first_node_space,
                "current_node": self.current_node_space,
            }
        )

        self.action_space = gym.spaces.Discrete(n_nodes)

    def reset(self, seed: int = None, options: Dict[str, Any] = None, add_starting_node=True):
        """Create a graph with self.n_nodes nodes and return the initial observation."""
        self.nodes = np.random.rand(self.n_nodes, 2)
        # add a column for the visited flag
        self.nodes = np.concatenate([self.nodes, np.zeros((self.n_nodes, 1))], axis=1)
        # change self.nodes to float32
        self.nodes = self.nodes.astype(np.float32)
        self.edge_links = np.array(
            [x for x in itertools.product(range(self.n_nodes), repeat=2) if x[0] != x[1]],
            dtype=np.int64,
        )  # avoid self loops
        self.edges = np.zeros(len(self.edge_links), dtype=np.int64)
        self.first_node = None
        self.current_node = None
        self.partial_solution = []  # TODO to array?
        mask = np.zeros(self.n_nodes, dtype=np.int64)

        if add_starting_node:
            self.first_node = np.array([np.random.randint(self.n_nodes)], dtype=np.int64)
            self.current_node = self.first_node
            self.nodes[self.current_node, 2] = 1
            self.partial_solution.append(self.current_node)
            mask[self.current_node] = 1
        return {
            "nodes": self.nodes,
            "edge_links": self.edge_links,
            "edges": self.edges,
            "mask": mask,
            "first_node": self.first_node,
            "current_node": self.current_node,
        }, {"partial_solution": self.partial_solution}

    def step(self, action: int):
        """Mark the edge as visited and return the new observation and the reward."""
        self.nodes[action, 2] = 1
        self.partial_solution.append(action)
        if self.current_node is not None:
            # find the index of the edge_link equal to [current_node, action]
            edge_index = np.nonzero(np.all(self.edge_links == [self.current_node.item(), action.item()], axis=1))[0]
            self.edges[edge_index] = True
        else:
            self.first_node = action
        self.current_node = np.array([action])

        if np.all(self.nodes[:, 2]):
            # close the tour
            edge_index = np.nonzero(
                np.all(self.edge_links == [self.current_node.item(), self.first_node.item()], axis=1)
            )[0]
            self.edges[edge_index] = True
            penalty = (
                0
                if self.reward_mode == "sparse"
                else self._compute_edge_length(self.partial_solution[-2], self.partial_solution[-1])
            )
            self.partial_solution.append(self.first_node)
            penalty += (
                self._compute_tour_length()
                if self.reward_mode == "sparse"
                else self._compute_edge_length(self.partial_solution[-2], self.partial_solution[-1])
            )
            done = True
        else:
            edge_len = 0
            if len(self.partial_solution) >= 2:
                edge_len = self._compute_edge_length(self.partial_solution[-2], self.partial_solution[-1])

            penalty = 0 if self.reward_mode == "sparse" else edge_len
            done = False

        mask = np.zeros(self.n_nodes, dtype=np.int64)
        sol_list = [x.item() for x in self.partial_solution]
        mask[sol_list] = 1

        obs = {
            "nodes": self.nodes,
            "edge_links": self.edge_links,
            "edges": self.edges,
            "mask": mask,
            "first_node": self.first_node,
            "current_node": self.current_node,
        }
        return obs, -penalty, done, False, {"partial_solution": self.partial_solution}

    def render(self, mode=None):
        """Render the environment."""
        if mode is None:
            mode = self.render_mode
        if mode in ["human", "rgb_array"]:
            # create a plot with the nodes and edges visited using pygame
            if self.window is None:
                pygame.init()
                if mode == "human":
                    pygame.display.init()
                    self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface((self.screen_size, self.screen_size))
            canvas.fill((255, 255, 255))
            if mode == "human":
                pygame.display.set_caption("TSP")

            for i in range(self.n_nodes):
                # use red for the first node and green for the current node
                x, y, visited = self.nodes[i]
                match i:
                    case self.first_node:
                        color = (255, 0, 0)
                    case self.current_node:
                        color = (0, 255, 0)
                    case _:
                        color = (0, 0, 0)
                pygame.draw.circle(
                    canvas,
                    color,
                    (
                        int(x * self.screen_size),
                        int(y * self.screen_size),
                    ),
                    10,
                )

            for i in range(len(self.partial_solution) - 1):
                node_idx = self.partial_solution[i]
                x, y, _ = self.nodes[node_idx.item()]
                next_node_idx = self.partial_solution[(i + 1)].item()
                x_next, y_next, _ = self.nodes[next_node_idx]
                pygame.draw.line(
                    canvas,
                    (0, 0, 0),
                    (
                        int(x * self.screen_size),
                        int(y * self.screen_size),
                    ),
                    (
                        int(x_next * self.screen_size),
                        int(y_next * self.screen_size),
                    ),
                    2,
                )

            if mode == "human":
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                # display the canvas
                pygame.display.update()

                # We need to ensure that human-rendering occurs at the predefined framerate.
                # The following line will automatically add a delay to keep the framerate stable.
                self.clock.tick(self.metadata["render_fps"])

            if mode == "rgb_array":
                # return the rendered image
                return pygame.surfarray.array3d(canvas)

        else:
            raise ValueError(f"Invalid render mode: {mode}")

    def close(self):
        """Close the environment."""
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def _compute_tour_length(self):
        """Compute the length of the tour."""
        length = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                edge_index = np.nonzero(np.all(self.edge_links == [i, j], axis=1))[0]
                if edge_index.size > 0 and self.edges[edge_index]:
                    length += self._compute_edge_length(i, j)
        return length

    def _compute_edge_length(self, node1, node2):
        """Compute the length of the node1-node2 edge."""
        return np.linalg.norm(self.nodes[node1, :2] - self.nodes[node2, :2])
