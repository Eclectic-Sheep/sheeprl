import itertools

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Graph, GraphInstance


class TSP(gym.Env):
    """Travelling Salesman Problem environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, n_nodes, render_mode=None, screen_size=600, seed=None):
        """Create a TSP environment with n_nodes nodes."""
        np.random.seed(seed)
        self.n_nodes = n_nodes

        if render_mode is None:
            render_mode = "human"
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode: {render_mode}")
        self.render_mode = render_mode
        self.screen_size = screen_size
        self.window = None
        self.clock = None

        self.observation_space = Graph(
            node_space=gym.spaces.Box(low=0, high=1, shape=(3,)),  # x, y, visited
            edge_space=gym.spaces.Discrete(2),  # visited or not
            seed=seed,
        )

    def reset(self):
        """Create a graph with self.n_nodes nodes and return the initial observation."""
        nodes = np.random.rand(self.n_nodes, 2)
        # add a column for the visited flag
        nodes = np.concatenate([nodes, np.zeros((self.n_nodes, 1))], axis=1)
        edge_links = np.array(
            [x for x in itertools.product(range(self.n_nodes), repeat=2) if x[0] != x[1]]
        )  # avoid self loops
        edges = np.zeros(len(edge_links), dtype=bool)
        self.first_node = None
        self.current_node = None
        self.graph = GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)
        return self.graph

    def step(self, action: int):
        """Mark the edge as visited and return the new observation and the reward."""
        self.graph.nodes[action, 2] = 1
        if self.current_node is not None:
            # find the index of the edge_link equal to [current_node, action]
            edge_index = np.nonzero(np.all(self.graph.edge_links == [self.current_node, action], axis=1))[0]
            self.graph.edges[edge_index] = True
        else:
            self.first_node = action
        self.current_node = action

        if np.all(self.graph.nodes[:, 2]):
            # close the tour
            edge_index = np.nonzero(np.all(self.graph.edge_links == [self.current_node, self.first_node], axis=1))[0]
            self.graph.edges[edge_index] = True
            reward = self._compute_tour_length()
            done = True
        else:
            reward = 0
            done = False
        return self.graph, reward, done, {"first_node": self.first_node, "current_node": self.current_node}

    def render(self, mode=None):
        """Render the environment."""
        if mode is None:
            mode = self.render_mode
        if mode == "human":
            # create a plot with the nodes and edges visited using pygame
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface((self.screen_size, self.screen_size))
            canvas.fill((255, 255, 255))
            pygame.display.set_caption("TSP")

            for i in range(self.n_nodes):
                # use red for the first node and green for the current node
                if i == self.first_node:
                    pygame.draw.circle(
                        canvas,
                        (255, 0, 0),
                        (
                            int(self.graph.nodes[i, 0] * self.screen_size),
                            int(self.graph.nodes[i, 1] * self.screen_size),
                        ),
                        10,
                    )
                elif i == self.current_node:
                    pygame.draw.circle(
                        canvas,
                        (0, 255, 0),
                        (
                            int(self.graph.nodes[i, 0] * self.screen_size),
                            int(self.graph.nodes[i, 1] * self.screen_size),
                        ),
                        10,
                    )
                else:
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 0),
                        (
                            int(self.graph.nodes[i, 0] * self.screen_size),
                            int(self.graph.nodes[i, 1] * self.screen_size),
                        ),
                        10,
                    )
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j and self.graph.edges[np.nonzero(np.all(self.graph.edge_links == [i, j], axis=1))[0]]:
                        pygame.draw.line(
                            canvas,
                            (0, 0, 0),
                            (
                                int(self.graph.nodes[i, 0] * self.screen_size),
                                int(self.graph.nodes[i, 1] * self.screen_size),
                            ),
                            (
                                int(self.graph.nodes[j, 0] * self.screen_size),
                                int(self.graph.nodes[j, 1] * self.screen_size),
                            ),
                            2,
                        )

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        elif mode == "rgb_array":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid render mode: {mode}")

    def _compute_tour_length(self):
        """Compute the length of the tour."""
        length = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                edge_index = np.nonzero(np.all(self.graph.edge_links == [i, j], axis=1))[0]
                if edge_index.size > 0 and self.graph.edges[edge_index]:
                    length += np.linalg.norm(self.graph.nodes[i, :2] - self.graph.nodes[j, :2])
        return length


if __name__ == "__main__":
    n_nodes = 10
    env = TSP(n_nodes=n_nodes, seed=9)
    obs = env.reset()
    print(obs)
    obs.nodes[0, 2] = 1

    # use 0 as the first action
    obs, reward, done, info = env.step(0)

    # use the nearest neighbor heuristic to solve the problem
    while not done:
        distances = np.zeros(n_nodes)
        for i in range(n_nodes):
            if not obs.nodes[i, 2] and i != info["current_node"]:
                distances[i] = np.linalg.norm(obs.nodes[info["current_node"], :2] - obs.nodes[i, :2])
            else:
                distances[i] = np.inf

        action = np.argmin(distances)
        obs, reward, done, info = env.step(action)
        env.render()
    print(f"Tour length: {reward}")
