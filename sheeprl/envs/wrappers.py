from collections import deque
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium.core import Env


class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observable.
    Adapted from https://github.com/LiuWenlin595/FinalProject.
    """

    # Supported envs
    velocity_indices = {
        "CartPole-v0": np.array([1, 3]),
        "CartPole-v1": np.array([1, 3]),
        "MountainCar-v0": np.array([1]),
        "MountainCarContinuous-v0": np.array([1]),
        "Pendulum-v1": np.array([2]),
        "LunarLander-v2": np.array([2, 3, 5]),
        "LunarLanderContinuous-v2": np.array([2, 3, 5]),
    }

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert env.unwrapped.spec is not None
        env_id: str = env.unwrapped.spec.id
        # By default no masking
        self.mask = np.ones_like(env.observation_space.sample())
        try:
            # Mask velocity
            self.mask[self.velocity_indices[env_id]] = 0.0
        except KeyError as e:
            raise NotImplementedError(f"Velocity masking not implemented for {env_id}") from e

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation * self.mask


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, amount: int = 1):
        super().__init__(env)
        if amount <= 0:
            raise ValueError("`amount` should be a positive integer")
        self._env = env
        self._amount = amount

    @property
    def action_repeat(self) -> int:
        return self._amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        truncated = False
        current_step = 0
        total_reward = 0.0
        while current_step < self._amount and not (done or truncated):
            obs, reward, done, truncated, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, truncated, info


class FrameStack(gym.Wrapper):
    def __init__(self, env: Env, num_stack: int, cnn_keys: Sequence[str]):
        super().__init__(env)
        if num_stack <= 0:
            raise ValueError(f"Invalid value for num_stack, expected a value greater than zero, got {num_stack}")
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise RuntimeError(
                f"Expected an observation space of type gym.spaces.Dict, got: {type(env.observation_space)}"
            )
        self._env = env
        self._num_stack = num_stack
        self._cnn_keys = []
        self.observation_space = self._env.observation_space
        for k, v in self._env.observation_space.spaces.items():
            if (
                cnn_keys
                and (k in cnn_keys or (len(cnn_keys) == 1 and cnn_keys[0].lower() == "all"))
                and len(v.shape) == 3
            ):
                self._cnn_keys.append(k)
                self._observation_space[k] = gym.spaces.Box(
                    np.repeat(self._env.observation_space[k].low[None, ...], 4, axis=0),
                    np.repeat(self._env.observation_space[k].high[None, ...], 4, axis=0),
                    (self._num_stack, *self._env.observation_space[k].shape),
                    self._env.observation_space[k].dtype,
                )

        if self._cnn_keys is None or len(self._cnn_keys) == 0:
            raise RuntimeError(f"Specify at least one valid cnn key")
        self._frames = {k: deque(maxlen=num_stack) for k in self._cnn_keys}

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        stacked_obs = obs
        for k in self._cnn_keys:
            self._frames[k].append(obs[k])
            stacked_obs[k] = np.stack(list(self._frames[k]), axis=0)
        return stacked_obs, reward, done, truncated, info

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)

        [self._frames[k].clear() for k in self._cnn_keys]
        for k in self._cnn_keys:
            [self._frames[k].append(obs[k]) for _ in range(self._num_stack)]
            obs[k] = np.stack(list(self._frames[k]), axis=0)
        return obs, info
