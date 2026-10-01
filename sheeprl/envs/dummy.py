from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np


class BaseDummyEnv(gym.Env, ABC):
    @abstractmethod
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (3, 64, 64),
        n_steps: int = 128,
        vector_shape: Tuple[int] = (10,),
        dict_obs_space: bool = True,
    ):
        self._dict_obs_space = dict_obs_space
        if self._dict_obs_space:
            self.observation_space = gym.spaces.Dict(
                {
                    "rgb": gym.spaces.Box(0, 255, shape=image_size, dtype=np.uint8),
                    "state": gym.spaces.Box(-20, 20, shape=vector_shape, dtype=np.float32),
                }
            )
        else:
            self.observation_space = gym.spaces.Box(-20, 20, shape=vector_shape, dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self._current_step = 0
        self._n_steps = n_steps

    def step(self, action):
        done = self._current_step == self._n_steps
        self._current_step += 1
        return (
            self.get_obs(),
            np.zeros(1, dtype=np.float32).item(),
            done,
            False,
            {},
        )

    def get_obs(self) -> Dict[str, np.ndarray]:
        if self._dict_obs_space:
            return {
                # da sostituire con np.random.rand
                "rgb": np.full(self.observation_space["rgb"].shape, self._current_step % 256, dtype=np.uint8),
                "state": np.full(self.observation_space["state"].shape, self._current_step, dtype=np.uint8),
            }
        else:
            return np.full(self.observation_space.shape, self._current_step, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self._current_step = 0
        return self.get_obs(), {}

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class ContinuousDummyEnv(BaseDummyEnv):
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (3, 64, 64),
        n_steps: int = 128,
        vector_shape: Tuple[int] = (10,),
        action_dim: int = 2,
        dict_obs_space: bool = True,
    ):
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(action_dim,))
        super().__init__(
            image_size=image_size, n_steps=n_steps, vector_shape=vector_shape, dict_obs_space=dict_obs_space
        )


class DiscreteDummyEnv(BaseDummyEnv):
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (3, 64, 64),
        n_steps: int = 4,
        vector_shape: Tuple[int] = (10,),
        action_dim: int = 2,
        dict_obs_space: bool = True,
    ):
        self.action_space = gym.spaces.Discrete(action_dim)
        super().__init__(
            image_size=image_size, n_steps=n_steps, vector_shape=vector_shape, dict_obs_space=dict_obs_space
        )


class MultiDiscreteDummyEnv(BaseDummyEnv):
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (3, 64, 64),
        n_steps: int = 128,
        vector_shape: Tuple[int] = (10,),
        action_dims: List[int] = [2, 2],
        dict_obs_space: bool = True,
    ):
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        super().__init__(
            image_size=image_size, n_steps=n_steps, vector_shape=vector_shape, dict_obs_space=dict_obs_space
        )
