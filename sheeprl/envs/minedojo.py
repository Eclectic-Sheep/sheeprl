"""Adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py"""

from typing import Any, Dict, Optional, Tuple

import minedojo
import numpy as np
from gymnasium import core


class MineDojoWrapper(core.Env):
    def __init__(
        self,
        task_id: str,
        height: int = 64,
        width: int = 64,
        pitch_limits: Tuple[int, int] = (-60, 60),
        seed: Optional[int] = None,
        kwargs: Optional[Dict[Any, Any]] = None,
    ):
        self._height = height
        self._width = width
        self._pitch_limits = pitch_limits
        self._pos = kwargs.pop("start_position", None)
        # TODO: control pitch limits

        # create task
        self._env = minedojo.make(
            task_id=task_id,
            image_size=(height, width),
            world_seed=seed,
            start_position=self._pos,
            **kwargs,
        )
        print(kwargs)
        # render
        self._render_mode: str = "rgb_array"
        # set seed
        self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _convert_action(self, action):
        return action

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def seed(self, seed: Optional[int] = None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, action):
        action = self._convert_action(action)

        obs, reward, done, info = self._env.step(action)
        # TODO: pitch control
        self._pos = {
            "x": ...,
            "y": ...,
            "z": ...,
            "pitch": ...,
            "yaw": ...,
        }
        return obs, reward, done, False, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = self._env.reset()
        if self._pos is None:
            self._pos = {
                "x": ...,
                "y": ...,
                "z": ...,
                "pitch": ...,
                "yaw": ...,
            }
        return obs, {}

    def close(self):
        self._env.close()
        return super().close()
