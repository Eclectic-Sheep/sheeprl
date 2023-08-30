"""Adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py"""

from sheeprl.utils.imports import _IS_DMC_AVAILABLE

if not _IS_DMC_AVAILABLE:
    raise ModuleNotFoundError(_IS_DMC_AVAILABLE)

from typing import Any, Dict, Optional, Tuple

import numpy as np
from dm_control import suite
from dm_env import specs
from gymnasium import core, spaces


def _spec_to_box(spec, dtype) -> spaces.Space:
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            raise ValueError(f"Unrecognized spec: {type(s)}")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs: Dict[Any, Any]) -> np.ndarray:
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
        self,
        id: str,
        from_pixels: bool = False,
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
        frame_skip: int = 1,
        task_kwargs: Optional[Dict[Any, Any]] = None,
        environment_kwargs: Optional[Dict[Any, Any]] = None,
        channels_first: bool = True,
        visualize_reward: bool = False,
        seed: Optional[int] = None,
    ):
        domain_name, task_name = id.split("_")

        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32)

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        else:
            self._observation_space = _spec_to_box(self._env.observation_spec().values(), np.float64)

        # state space
        self._state_space = _spec_to_box(self._env.observation_spec().values(), np.float64)
        self.current_state = None

        # render
        self._render_mode: str = "rgb_array"

        # set seed
        self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(camera_id=self._camera_id)
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def seed(self, seed: Optional[int] = None):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, done, False, extra

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs, {}

    def render(self, camera_id: Optional[int] = None):
        return self._env.physics.render(height=self._height, width=self._width, camera_id=camera_id or self._camera_id)

    def close(self):
        self._env.close()
        return super().close()
