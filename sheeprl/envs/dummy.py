from typing import List, Tuple

import gymnasium as gym
import numpy as np


class ContinuousDummyEnv(gym.Env):
    def __init__(self, action_dim: int = 2, size: Tuple[int, int, int] = (3, 64, 64), n_steps: int = 128):
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(action_dim,))
        self.observation_space = gym.spaces.Box(0, 256, shape=size, dtype=np.uint8)
        self.reward_range = (-np.inf, np.inf)
        self._current_step = 0
        self._n_steps = n_steps

    def step(self, action):
        done = self._current_step == self._n_steps
        self._current_step += 1
        return (
            np.zeros(self.observation_space.shape, dtype=np.uint8),
            np.zeros(1, dtype=np.float32).item(),
            done,
            False,
            {},
        )

    def reset(self, seed=None, options=None):
        self._current_step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class DiscreteDummyEnv(gym.Env):
    def __init__(self, action_dim: int = 2, size: Tuple[int, int, int] = (3, 64, 64), n_steps: int = 4):
        self.action_space = gym.spaces.Discrete(action_dim)
        self.observation_space = gym.spaces.Box(0, 256, shape=size, dtype=np.uint8)
        self.reward_range = (-np.inf, np.inf)
        self._current_step = 0
        self._n_steps = n_steps

    def step(self, action):
        done = self._current_step == self._n_steps
        self._current_step += 1
        return (
            np.random.randint(0, 256, self.observation_space.shape, dtype=np.uint8),
            np.zeros(1, dtype=np.float32).item(),
            done,
            False,
            {},
        )

    def reset(self, seed=None, options=None):
        self._current_step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class MultiDiscreteDummyEnv(gym.Env):
    def __init__(self, action_dims: List[int] = [2, 2], size: Tuple[int, int, int] = (3, 64, 64), n_steps: int = 128):
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        self.observation_space = gym.spaces.Box(0, 256, shape=size, dtype=np.uint8)
        self.reward_range = (-np.inf, np.inf)
        self._current_step = 0
        self._n_steps = n_steps

    def step(self, action):
        done = self._current_step == self._n_steps
        self._current_step += 1
        return (
            np.zeros(self.observation_space.shape, dtype=np.uint8),
            np.zeros(1, dtype=np.float32).item(),
            done,
            False,
            {},
        )

    def reset(self, seed=None, options=None):
        self._current_step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
