from __future__ import annotations

import copy
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import Env, RenderFrame

from sheeprl.transforms.base import Pipeline


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
        "LunarLander-v3": np.array([2, 3, 5]),
        "LunarLanderContinuous-v3": np.array([2, 3, 5]),
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


class _BoxNormalizer:
    """The bounded dimensions of the `Box` space `space` exposed as [-1, 1] (`self.space`): `__call__` clips an action
    of the exposed space and rescales it to the bounds of `space`."""

    def __init__(self, space: gym.spaces.Box) -> None:
        self.env_dtype = space.dtype
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)
        # The dimensions already in [-1, 1] are not recomputed: they get the same actions, bit for bit
        self._rescale = self._mask & ((self._low != -1) | (self._high != 1))
        self.space = gym.spaces.Box(
            low=np.where(self._mask, -1, space.low),
            high=np.where(self._mask, 1, space.high),
            shape=space.shape,
            dtype=space.dtype,
        )

    def __call__(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, self.space.low, self.space.high)
        env_action = (action + 1) / 2 * (self._high - self._low) + self._low
        return np.where(self._rescale, env_action, action).astype(self.env_dtype, copy=False)


def has_box_actions(space: gym.Space) -> bool:
    """Whether the action space `space` is a `Box`, or a tuple or a dictionary of spaces with a `Box` among its
    parts."""
    if isinstance(space, gym.spaces.Tuple):
        return any(has_box_actions(part) for part in space.spaces)
    if isinstance(space, gym.spaces.Dict):
        return any(has_box_actions(part) for part in space.spaces.values())
    return isinstance(space, gym.spaces.Box)


class NormalizeAction(gym.ActionWrapper):
    """Exposes the bounded dimensions of a `Box` action space as [-1, 1], rescaled to the bounds of the environment,
    and clips the actions to the exposed space. The unbounded dimensions are left as they are. In a tuple or a
    dictionary of action spaces, the `Box` parts are normalized and the other parts left as they are.

    The `NormalizeAction` and `ClipAction` wrappers of DreamerV3
    (https://github.com/danijar/dreamerv3/blob/bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01/embodied/core/wrappers.py).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not has_box_actions(env.action_space):
            raise ValueError(f"Only `Box` actions can be normalized, got: {env.action_space}")
        self._normalizers = self._build(env.action_space)
        self.action_space = self._exposed(env.action_space, self._normalizers)

    @classmethod
    def _build(cls, space: gym.Space) -> Any:
        if isinstance(space, gym.spaces.Box):
            return _BoxNormalizer(space)
        if isinstance(space, gym.spaces.Tuple):
            return tuple(cls._build(part) for part in space.spaces)
        if isinstance(space, gym.spaces.Dict):
            return {key: cls._build(part) for key, part in space.spaces.items()}
        return None

    @classmethod
    def _exposed(cls, space: gym.Space, normalizers: Any) -> gym.Space:
        if isinstance(space, gym.spaces.Box):
            return normalizers.space
        if isinstance(space, gym.spaces.Tuple):
            return gym.spaces.Tuple([cls._exposed(part, n) for part, n in zip(space.spaces, normalizers)])
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({key: cls._exposed(part, normalizers[key]) for key, part in space.spaces.items()})
        return space

    @classmethod
    def _normalize(cls, action: Any, normalizers: Any) -> Any:
        if isinstance(normalizers, _BoxNormalizer):
            return normalizers(action)
        if isinstance(normalizers, tuple):
            return tuple(cls._normalize(part, n) for part, n in zip(action, normalizers))
        if isinstance(normalizers, dict):
            return {key: cls._normalize(action[key], n) for key, n in normalizers.items()}
        return action

    def action(self, action: Any) -> Any:
        return self._normalize(action, self._normalizers)


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, amount: int = 1):
        super().__init__(env)
        if amount <= 0:
            raise ValueError("`amount` should be a positive integer")
        self._amount = amount

    @property
    def action_repeat(self) -> int:
        return self._amount

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        done = False
        truncated = False
        current_step = 0
        total_reward = 0.0
        while current_step < self._amount and not (done or truncated):
            obs, reward, done, truncated, info = self.env.step(action)
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
            new_obs, info = self.env.reset(seed=seed, options=options)
            info.update({"restart_on_exception": True})
            return new_obs, info


# The separator of the parts of the nested observations in their flat keys (`agent.pos`): not `/`, since the buffers
# name their memory-mapped files after the keys
NESTED_KEY_SEPARATOR = "."


def _leaves(space: gym.Space, path: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], gym.Space]]:
    """The parts of `space` that are not `Dict` or `Tuple` spaces, with their paths (keys and indices)."""
    if isinstance(space, gym.spaces.Dict):
        return [leaf for key, part in space.spaces.items() for leaf in _leaves(part, (*path, key))]
    if isinstance(space, gym.spaces.Tuple):
        return [leaf for i, part in enumerate(space.spaces) for leaf in _leaves(part, (*path, i))]
    return [(path, space)]


def has_nested_observations(space: gym.spaces.Dict) -> bool:
    """Whether some observations of the dictionary `space` are dictionaries or tuples."""
    return any(isinstance(part, (gym.spaces.Dict, gym.spaces.Tuple)) for part in space.spaces.values())


class FlattenNestedKeys(gym.ObservationWrapper):
    """The parts of the nested observations as flat keys: the part `pos` of the dictionary `agent` is the key
    `agent.pos`, the parts of a tuple are numbered (`sensors.0`, `sensors.1`). The observations that are neither
    dictionaries nor tuples keep their keys. Two parts with the same flat key are an error."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError(f"FlattenNestedKeys takes a dictionary of observations, got {env.observation_space}")
        self._paths: Dict[str, Tuple[Any, ...]] = {}
        spaces: Dict[str, gym.Space] = {}
        for path, space in _leaves(env.observation_space):
            key = NESTED_KEY_SEPARATOR.join(str(part) for part in path)
            if key in spaces:
                raise ValueError(f"Two parts of the observations have the flat key `{key}`: rename one of them")
            self._paths[key] = path
            spaces[key] = space
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        flat = {}
        for key, path in self._paths.items():
            value: Any = observation
            for part in path:
                value = value[part]
            flat[key] = value
        return flat


class TransformsWrapper(gym.Wrapper):
    """The `frame` stage of the observation keys (`sheeprl.transforms`): every key the agent stores goes through the
    `frame` transforms of its pipeline, once per step of the environment; the other keys are dropped. The transforms get
    the `info` of the step with its `terminated` and `truncated`, and are reset (and seeded) with the environment."""

    def __init__(self, env: Env, pipelines: Dict[str, Pipeline]) -> None:
        super().__init__(env)
        self.pipelines = pipelines
        self.observation_space = gym.spaces.Dict({k: p.frame_space for k, p in pipelines.items()})
        self._needs_info = any(p.needs_info for p in pipelines.values())

    def _transform(
        self, obs: Dict[str, Any], info: Dict[str, Any], terminated: bool, truncated: bool
    ) -> Dict[str, Any]:
        info = {**info, "terminated": terminated, "truncated": truncated} if self._needs_info else None
        return {k: p.frame(obs[k], info) for k, p in self.pipelines.items()}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        for pipeline in self.pipelines.values():
            pipeline.reset()
            if seed is not None:
                pipeline.seed(seed)
        return self._transform(obs, info, False, False), info

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs, info, terminated, truncated), reward, terminated, truncated, info


class RewardAsObservationWrapper(gym.Wrapper):
    """It adds the reward to the observations.
    The reward is assumed to be a float scalar, so the reward converted as observation is a gymnasium.spaces.Box
    with shape (1,). Moreover, it tries to get the reward range from the environment to wrap, otherwise it assumes
    that the reward has no range (-inf, +inf).

    It converts the observation space (if it is not already) in gymnasium.spaces.Dict. In case the observation
    space of the environment is a dictionary, then it simply adds the `reward` key in the dictionary, otherwise,
    it creates a dictionary with two keys:
    1. `obs`: the observations of the environment.
    2. `reward`: the reward obtained by the agent.

    Args:
        env (Env): the environment to wrap.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        reward_range = (
            self.env.get_wrapper_attr("reward_range") or (-np.inf, np.inf)
            if self.env.has_wrapper_attr("reward_range")
            else (-np.inf, np.inf)
        )
        # The reward is assumed to be a scalar
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict(
                {
                    "reward": gym.spaces.Box(*reward_range, (1,), np.float32),
                    **{k: v for k, v in self.env.observation_space.items()},
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {"obs": self.env.observation_space, "reward": gym.spaces.Box(*reward_range, (1,), np.float32)}
            )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _convert_obs(self, obs: Any, reward: Union[float, np.ndarray]) -> Dict[str, Any]:
        reward_obs = (np.array(reward) if not isinstance(reward, np.ndarray) else reward).reshape(-1)
        if isinstance(obs, dict):
            obs["reward"] = reward_obs
        else:
            obs = {
                "obs": obs,
                "reward": reward_obs,
            }
        return obs

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, truncated, infos = self.env.step(action)
        return self._convert_obs(obs, copy.deepcopy(reward)), reward, done, truncated, infos

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._convert_obs(obs, 0), infos


class GrayscaleRenderWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        frame = super().render()
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 2:
                frame = frame[..., np.newaxis]
            if len(frame.shape) == 3 and frame.shape[-1] == 1:
                frame = frame.repeat(3, axis=-1)
        return frame


class ActionsAsObservationWrapper(gym.Wrapper):
    def __init__(self, env: Env, num_stack: int, noop: float | int | List[int], dilation: int = 1):
        super().__init__(env)
        if num_stack < 1:
            raise ValueError(
                "The number of actions to the `action_stack` observation "
                f"must be greater or equal than 1, got: {num_stack}"
            )
        if dilation < 1:
            raise ValueError(f"The actions stack dilation argument must be greater than zero, got: {dilation}")
        if not isinstance(noop, (int, float, list)):
            raise ValueError(f"The noop action must be an integer or float or list, got: {noop} ({type(noop)})")
        self._num_stack = num_stack
        self._dilation = dilation
        self._actions = deque(maxlen=num_stack * dilation)
        if isinstance(self.env.action_space, (gym.spaces.Tuple, gym.spaces.Dict)):
            raise NotImplementedError(
                f"The actions as observations of a tuple or a dictionary of action spaces are not supported, got: "
                f"{self.env.action_space}"
            )
        self._is_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        self._is_multidiscrete = isinstance(self.env.action_space, gym.spaces.MultiDiscrete)
        self.observation_space = copy.deepcopy(self.env.observation_space)
        if self._is_continuous:
            self._action_shape = self.env.action_space.shape[0]
            low = np.resize(self.env.action_space.low, self._action_shape * num_stack)
            high = np.resize(self.env.action_space.high, self._action_shape * num_stack)
        elif self._is_multidiscrete:
            low = 0
            high = 1  # one-hot encoding
            # one one-hot for each action
            self._action_shape = sum(self.env.action_space.nvec)
        else:
            low = 0
            high = 1  # one-hot encoding
            self._action_shape = self.env.action_space.n
        self.observation_space["action_stack"] = gym.spaces.Box(
            low=low, high=high, shape=(self._action_shape * num_stack,), dtype=np.float32
        )
        if self._is_continuous:
            if isinstance(noop, list):
                raise ValueError(f"The noop actions must be a float for continuous action spaces, got: {noop}")
            self.noop = np.full((self._action_shape,), noop, dtype=np.float32)
        elif self._is_multidiscrete:
            if not isinstance(noop, list):
                raise ValueError(f"The noop actions must be a list for multi-discrete action spaces, got: {noop}")
            if len(self.env.action_space.nvec) != len(noop):
                raise RuntimeError(
                    "The number of noop actions must be equal to the number of actions of the environment. "
                    f"Got env_action_space = {self.env.action_space.nvec} and {noop =}"
                )
            noops = []
            for act, n in zip(noop, self.env.action_space.nvec):
                noops.append(np.zeros((n,), dtype=np.float32))
                noops[-1][noop[act]] = 1.0
            self.noop = np.concatenate(noops, axis=-1)
        else:
            if isinstance(noop, (list, float)):
                raise ValueError(f"The noop actions must be an integer for discrete action spaces, got: {noop}")
            self.noop = np.zeros((self._action_shape,), dtype=np.float32)
            self.noop[noop] = 1.0

    def step(self, action: Any) -> Tuple[Any | SupportsFloat | bool | Dict[str, Any]]:
        if self._is_continuous:
            self._actions.append(action)
        elif self._is_multidiscrete:
            one_hot_actions = []
            for act, n in zip(action, self.env.action_space.nvec):
                one_hot_actions.append(np.zeros((n,), dtype=np.float32))
                one_hot_actions[-1][act] = 1.0
            self._actions.append(np.concatenate(one_hot_actions, axis=-1))
        else:
            one_hot_action = np.zeros((self._action_shape,), dtype=np.float32)
            one_hot_action[action] = 1.0
            self._actions.append(one_hot_action)
        obs, reward, done, truncated, info = super().step(action)
        obs["action_stack"] = self._get_actions_stack()
        return obs, reward, done, truncated, info

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Any | Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self._actions.clear()
        [self._actions.append(self.noop) for _ in range(self._num_stack * self._dilation)]
        obs["action_stack"] = self._get_actions_stack()
        return obs, info

    def _get_actions_stack(self) -> np.ndarray:
        actions_stack = list(self._actions)[self._dilation - 1 :: self._dilation]
        actions = np.concatenate(actions_stack, axis=-1)
        return actions.astype(np.float32)
