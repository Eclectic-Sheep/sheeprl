import copy
import time
from collections import deque
from typing import Any, Callable, Dict, Optional, Sequence, SupportsFloat, Tuple

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


class RestartOnException(gym.Wrapper):
    def __init__(self, env_fn: Callable[..., gym.Env], exceptions=(Exception,), window=300, maxfails=2, wait=20):
        if not isinstance(exceptions, (tuple, list)):
            exceptions = [exceptions]
        self._env_fn = env_fn
        self._exceptions = tuple(exceptions)
        self._window = window
        self._maxfails = maxfails
        self._wait = wait
        self._last = time.time()
        self._fails = 0
        super().__init__(self._env_fn())

    def step(self, action) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        try:
            return self.env.step(action)
        except self._exceptions as e:
            if time.time() > self._last + self._window:
                self._last = time.time()
                self._fails = 1
            else:
                self._fails += 1
            if self._fails > self._maxfails:
                raise RuntimeError(f"The env crashed too many times: {self._fails}")
            gym.logger.warn(f"STEP - Restarting env after crash with {type(e).__name__}: {e}")
            time.sleep(self._wait)
            self.env = self._env_fn()
            new_obs, info = self.env.reset()
            info.update({"restart_on_exception": True})
            return new_obs, 0.0, False, False, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        try:
            return self.env.reset(seed=seed, options=options)
        except self._exceptions as e:
            if time.time() > self._last + self._window:
                self._last = time.time()
                self._fails = 1
            else:
                self._fails += 1
            if self._fails > self._maxfails:
                raise RuntimeError(f"The env crashed too many times: {self._fails}")
            gym.logger.warn(f"RESET - Restarting env after crash with {type(e).__name__}: {e}")
            time.sleep(self._wait)
            self.env = self._env_fn()
            new_obs, info = self.env.reset()
            info.update({"restart_on_exception": True})
            return new_obs, info


class FrameStack(gym.Wrapper):
    def __init__(self, env: Env, num_stack: int, cnn_keys: Sequence[str], dilation: int = 1) -> None:
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
        self._dilation = dilation
        self.observation_space = copy.deepcopy(self._env.observation_space)
        for k, v in self._env.observation_space.spaces.items():
            if cnn_keys and len(v.shape) == 3:
                self._cnn_keys.append(k)
                self.observation_space[k] = gym.spaces.Box(
                    np.repeat(self._env.observation_space[k].low[None, ...], num_stack, axis=0),
                    np.repeat(self._env.observation_space[k].high[None, ...], num_stack, axis=0),
                    (self._num_stack, *self._env.observation_space[k].shape),
                    self._env.observation_space[k].dtype,
                )

        if self._cnn_keys is None or len(self._cnn_keys) == 0:
            raise RuntimeError(f"Specify at least one valid cnn key to be stacked")
        self._frames = {k: deque(maxlen=num_stack * dilation) for k in self._cnn_keys}

    def _get_obs(self, key):
        frames_subset = list(self._frames[key])[self._dilation - 1 :: self._dilation]
        assert len(frames_subset) == self._num_stack
        return np.stack(list(frames_subset), axis=0)

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, truncated, infos = self._env.step(action)
        for k in self._cnn_keys:
            self._frames[k].append(obs[k])
            if (
                "env_domain" in infos
                and infos["env_domain"] == "DIAMBRA"
                and len(set(["round_done", "stage_done", "game_done"]).intersection(infos.keys())) == 3
                and (infos["round_done"] or infos["stage_done"] or infos["game_done"])
                and not (done or truncated)
            ):
                for _ in range(self._num_stack * self._dilation - 1):
                    self._frames[k].append(obs[k])
            obs[k] = self._get_obs(k)
        return obs, reward, done, truncated, infos

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, infos = self._env.reset(seed=seed, **kwargs)
        [self._frames[k].clear() for k in self._cnn_keys]
        for k in self._cnn_keys:
            [self._frames[k].append(obs[k]) for _ in range(self._num_stack * self._dilation)]
            obs[k] = self._get_obs(k)
        return obs, infos
