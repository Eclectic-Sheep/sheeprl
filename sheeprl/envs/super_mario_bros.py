from __future__ import annotations

from sheeprl.utils.imports import _IS_SUPER_MARIO_BROS_AVAILABLE

if not _IS_SUPER_MARIO_BROS_AVAILABLE:
    raise ModuleNotFoundError(_IS_SUPER_MARIO_BROS_AVAILABLE)


from typing import Any, Dict, SupportsFloat, Tuple

import gym_super_mario_bros as gsmb
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from gymnasium.core import RenderFrame
from nes_py.wrappers import JoypadSpace

ACTIONS_SPACE_MAP = {"simple": SIMPLE_MOVEMENT, "right_only": RIGHT_ONLY, "complex": COMPLEX_MOVEMENT}


class JoypadSpaceCustomReset(JoypadSpace):
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        return self.env.reset(seed=seed, options=options)


class SuperMarioBrosWrapper(gym.Wrapper):
    def __init__(self, id: str, action_space: str = "simple", render_mode: str = "rgb_array"):
        env = gsmb.make(id)
        env = JoypadSpaceCustomReset(env, ACTIONS_SPACE_MAP[action_space])
        super().__init__(env)

        self._render_mode = render_mode
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    env.observation_space.low,
                    env.observation_space.high,
                    env.observation_space.shape,
                    env.observation_space.dtype,
                )
            }
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)

    @property
    def render_mode(self) -> str:
        return self._render_mode

    @render_mode.setter
    def render_mode(self, render_mode: str):
        self._render_mode = render_mode

    def step(self, action: np.ndarray | int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = action.squeeze().item()
        obs, reward, done, info = self.env.step(action)
        converted_obs = {"rgb": obs.copy()}
        is_timelimit = info.get("time", False)
        return converted_obs, reward, done and not is_timelimit, done and is_timelimit, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        rendered_frame: np.ndarray | None = self.env.render(mode=self.render_mode)
        if self.render_mode == "rgb_array" and rendered_frame is not None:
            return rendered_frame.copy()
        return

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Any, Dict[str, Any]]:
        obs = self.env.reset(seed=seed, options=options)
        converted_obs = {"rgb": obs.copy()}
        return converted_obs, {}
