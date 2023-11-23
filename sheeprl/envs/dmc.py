"""Adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py"""

from sheeprl.utils.imports import _IS_DMC_AVAILABLE

if not _IS_DMC_AVAILABLE:
    raise ModuleNotFoundError(_IS_DMC_AVAILABLE)

from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_env import specs
from gymnasium import spaces


def _spec_to_box(spec, dtype) -> spaces.Box:
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


class DMCWrapper(gym.Wrapper):
    def __init__(
        self,
        id: str,
        from_pixels: bool = False,
        from_vectors: bool = True,
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
        task_kwargs: Optional[Dict[Any, Any]] = None,
        environment_kwargs: Optional[Dict[Any, Any]] = None,
        channels_first: bool = True,
        visualize_reward: bool = False,
        seed: Optional[int] = None,
    ):
        """DeepMind Control Suite wrapper,
        adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py.
        The observation space is:

        * If both `from_pixels` and `from_vectors` are True, then the observation space
        will be a `gymnasium.spaces.Dict` with two keys: 'rgb' and 'state' for the
        image and vector observation respectively
        * If only `from_vectors` is True, then the observation will be a single numpy array
        containing the concatenation of all the vector observations defined by the task
        * If only `from_pixels` is True, then the observation will be the image rendered
        by the camera specified through the `camera_id` parameter

        Args:
            id (str): the task id, e.g. 'walker_walk'. The id must be 'underscore' separated.
            from_pixels (bool, optional): whether to return the image observation.
                If both 'from_pixels' and 'from_vectors' are True, then the observation space
                will be a `gymnasium.spaces.Dict` with two keys: 'rgb' and 'state' for the
                image and vector observation respectively.
                Defaults to False.
            from_vectors (bool, optional): whether to return the vector observation. This will
                be a flattened observation made up by the concatenation of the multiple
                vector observations defined by the task.
                If both 'from_pixels' and 'from_vectors' are True, then the observation space
                will be a `gymnasium.spaces.Dict` with two keys: 'rgb' and 'state' for the
                image and vector observation respectively.
                Defaults to False.
            height (int, optional): image observation height.
                Defaults to 84.
            width (int, optional): image observation width.
                Defaults to 84.
            camera_id (int, optional): the id of the camera from where to take the image observation.
                Defaults to 0.
            task_kwargs (Optional[Dict[Any, Any]], optional): Optional dict of keyword arguments for the task.
                Defaults to None.
            environment_kwargs (Optional[Dict[Any, Any]], optional): Optional dict specifying
                keyword arguments for the environment.
                Defaults to None.
            channels_first (bool, optional): whether to return a channel-first image.
                Defaults to True.
            visualize_reward (bool, optional): If True, object colours in rendered
                frames are set to indicate the reward at each step.
                Defaults to False.
            seed (Optional[int], optional): the random seed.
                Defaults to None.
        """
        if not (from_vectors or from_pixels):
            raise ValueError(
                "'from_vectors' and 'from_pixels' must not be both False: "
                f"got {from_vectors} and {from_pixels} respectively."
            )

        domain_name, task_name = id.split("_")
        self._from_pixels = from_pixels
        self._from_vectors = from_vectors
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._channels_first = channels_first

        # create task
        env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )
        super().__init__(env)

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self.env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32)

        # set the reward range
        reward_space = _spec_to_box([self.env.reward_spec()], np.float32)
        self._reward_range = (reward_space.low.item(), reward_space.high.item())

        # create observation space
        # at least one between from_pixels and from_vectors is True
        obs_space = {}
        if from_pixels:
            shape = (3, height, width) if channels_first else (height, width, 3)
            obs_space["rgb"] = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        if from_vectors:
            obs_space["state"] = _spec_to_box(self.env.observation_spec().values(), np.float64)
        self._observation_space = spaces.Dict(obs_space)

        # state space
        self._state_space = _spec_to_box(self.env.observation_spec().values(), np.float64)
        self.current_state = None
        # render
        self._render_mode: str = "rgb_array"
        # metadata
        self._metadata = {}
        # set seed
        self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _get_obs(self, time_step) -> Dict[str, np.ndarray]:
        obs = {}
        # at least one between from_pixels and from_vectors is True
        if self._from_pixels:
            rgb_obs = self.render(camera_id=self._camera_id)
            if self._channels_first:
                rgb_obs = rgb_obs.transpose(2, 0, 1).copy()
            obs["rgb"] = rgb_obs
        if self._from_vectors:
            vec_obs = _flatten_obs(time_step.observation)
            obs["state"] = vec_obs
        return obs

    def _convert_action(self, action) -> np.ndarray:
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self) -> Union[spaces.Dict, spaces.Box]:
        return self._observation_space

    @property
    def state_space(self) -> spaces.Box:
        return self._state_space

    @property
    def action_space(self) -> spaces.Box:
        return self._norm_action_space

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self._reward_range

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def seed(self, seed: Optional[int] = None):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(
        self, action: Any
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], SupportsFloat, bool, bool, Dict[str, Any]]:
        action = self._convert_action(action)
        time_step = self.env.step(action)
        reward = time_step.reward or 0.0
        done = time_step.last()
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra = {}
        extra["discount"] = time_step.discount
        extra["internal_state"] = self.env.physics.get_state().copy()
        return obs, reward, done, False, extra

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Dict[str, Any]]:
        time_step = self.env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs, {}

    def render(self, camera_id: Optional[int] = None) -> np.ndarray:
        return self.env.physics.render(height=self._height, width=self._width, camera_id=camera_id or self._camera_id)
