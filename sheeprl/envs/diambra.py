from __future__ import annotations

import warnings

from sheeprl.utils.imports import _IS_DIAMBRA_ARENA_AVAILABLE, _IS_DIAMBRA_AVAILABLE

if not _IS_DIAMBRA_AVAILABLE:
    raise ModuleNotFoundError(_IS_DIAMBRA_AVAILABLE)
if not _IS_DIAMBRA_ARENA_AVAILABLE:
    raise ModuleNotFoundError(_IS_DIAMBRA_ARENA_AVAILABLE)

from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import diambra
import diambra.arena
import gymnasium as gym
import numpy as np
from diambra.arena import EnvironmentSettings, WrappersSettings
from gymnasium.core import RenderFrame


class DiambraWrapper(gym.Wrapper):
    def __init__(
        self,
        id: str,
        action_space: str = "diambra.arena.SpaceTypes.DISCRETE",
        screen_size: Union[int, Tuple[int, int]] = 64,
        grayscale: bool = False,
        repeat_action: int = 1,
        rank: int = 0,
        diambra_settings: Dict[str, Any] = {},
        diambra_wrappers: Dict[str, Any] = {},
        render_mode: str = "rgb_array",
        log_level: int = 0,
        increase_performance: bool = True,
    ) -> None:
        if isinstance(screen_size, int):
            screen_size = (screen_size,) * 2

        if diambra_settings.pop("frame_shape", None) is not None:
            warnings.warn("The DIAMBRA frame_shape setting is disabled")
        if diambra_settings.pop("n_players", None) is not None:
            warnings.warn("The DIAMBRA n_players setting is disabled")

        role = diambra_settings.pop("role", None)
        self._action_type = "discrete" if "diambra.arena.SpaceTypes.DISCRETE" == action_space else "multi-discrete"
        settings = EnvironmentSettings(
            **diambra_settings,
            **{
                "game_id": id,
                "action_space": eval(action_space),
                "n_players": 1,
                "role": eval(role) if role is not None else None,
                "render_mode": render_mode,
            },
        )
        if repeat_action > 1:
            if "step_ratio" not in settings or settings["step_ratio"] > 1:
                warnings.warn(
                    f"step_ratio parameter modified to 1 because the sticky action is active ({repeat_action})"
                )
            settings["step_ratio"] = 1
        if diambra_wrappers.pop("frame_shape", None) is not None:
            warnings.warn("The DIAMBRA frame_shape wrapper is disabled")
        if diambra_wrappers.pop("stack_frames", None) is not None:
            warnings.warn("The DIAMBRA stack_frames wrapper is disabled")
        if diambra_wrappers.pop("dilation", None) is not None:
            warnings.warn("The DIAMBRA dilation wrapper is disabled")
        if diambra_wrappers.pop("flatten", None) is not None:
            warnings.warn("The DIAMBRA flatten wrapper is disabled")
        wrappers = WrappersSettings(
            **diambra_wrappers,
            **{
                "flatten": True,
                "repeat_action": repeat_action,
            },
        )
        if increase_performance:
            settings.frame_shape = screen_size + (int(grayscale),)
        else:
            wrappers.frame_shape = screen_size + (int(grayscale),)
        env = diambra.arena.make(id, settings, wrappers, rank=rank, render_mode=render_mode, log_level=log_level)
        super().__init__(env)

        # Observation and action space
        self.action_space = self.env.action_space
        obs = {}
        for k in self.env.observation_space.spaces.keys():
            if isinstance(self.env.observation_space[k], gym.spaces.Discrete):
                low = 0
                high = self.env.observation_space[k].n - 1
                shape = (1,)
                dtype = np.int32
            elif isinstance(self.env.observation_space[k], gym.spaces.MultiDiscrete):
                low = np.zeros_like(self.env.observation_space[k].nvec)
                high = self.env.observation_space[k].nvec - 1
                shape = (len(high),)
                dtype = np.int32
            elif not isinstance(self.env.observation_space[k], gym.spaces.Box):
                raise RuntimeError(f"Invalid observation space, got: {type(self.env.observation_space[k])}")
            obs[k] = (
                self.env.observation_space[k]
                if isinstance(self.env.observation_space[k], gym.spaces.Box)
                else gym.spaces.Box(low, high, shape, dtype)
            )
        self.observation_space = gym.spaces.Dict(obs)
        self._render_mode = render_mode

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _convert_obs(self, obs: Dict[str, Union[int, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            k: (np.array(v) if not isinstance(v, np.ndarray) else v).reshape(self.observation_space[k].shape)
            for k, v in obs.items()
        }

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self._action_type == "discrete" and isinstance(action, np.ndarray):
            action = action.squeeze()
            action = action.item()
        obs, reward, done, truncated, infos = self.env.step(action)
        infos["env_domain"] = "DIAMBRA"
        return self._convert_obs(obs), reward, done or infos.get("env_done", False), truncated, infos

    def render(self, mode: str = "rgb_array", **kwargs) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, infos = self.env.reset(seed=seed, options=options)
        infos["env_domain"] = "DIAMBRA"
        return self._convert_obs(obs), infos
