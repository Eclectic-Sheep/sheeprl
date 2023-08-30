import warnings

from sheeprl.utils.imports import _IS_DIAMBRA_ARENA_AVAILABLE, _IS_DIAMBRA_AVAILABLE

if not _IS_DIAMBRA_AVAILABLE:
    raise ModuleNotFoundError(_IS_DIAMBRA_AVAILABLE)
if not _IS_DIAMBRA_ARENA_AVAILABLE:
    raise ModuleNotFoundError(_IS_DIAMBRA_ARENA_AVAILABLE)

from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union

import diambra
import diambra.arena
import gym
import gymnasium
import numpy as np
from gymnasium import core


class DiambraWrapper(core.Env):
    def __init__(
        self,
        id: str,
        action_space: str = "discrete",
        screen_size: Union[int, Tuple[int, int]] = 64,
        grayscale: bool = False,
        attack_but_combination: bool = True,
        actions_stack: int = 1,
        noop_max: int = 0,
        sticky_actions: int = 1,
        seed: Optional[int] = None,
        rank: int = 0,
        diambra_settings: Dict[str, Any] = {},
        diambra_wrappers: Dict[str, Any] = {},
    ) -> None:
        super().__init__()

        if isinstance(screen_size, int):
            screen_size = (screen_size,) * 2

        settings = {
            **diambra_settings,
            "action_space": action_space,
            "attack_but_combination": attack_but_combination,
            "frame_shape": (*screen_size, int(1 * grayscale)),
        }
        if sticky_actions > 1:
            if "step_ratio" not in settings or settings["step_ratio"] > 1:
                warnings.warn(
                    f"step_ratio parameter modified to 1 because the sticky action is active ({sticky_actions})"
                )
            settings["step_ratio"] = 1
        if diambra_wrappers.pop("frame_stack", None) is not None:
            warnings.warn("the DIAMBRA frame_stack wrapper is disabled")
        if diambra_wrappers.pop("dilation", None) is not None:
            warnings.warn("the DIAMBRA dilation wrapper is disabled")
        wrappers = {
            **diambra_wrappers,
            "no_op_max": noop_max,
            "flatten": True,
            "actions_stack": actions_stack,
            "sticky_actions": sticky_actions,
        }
        self._env = diambra.arena.make(id, settings, wrappers, seed=seed, rank=rank)

        # Observation and action space
        self.action_space = (
            gymnasium.spaces.Discrete(self._env.action_space.n)
            if action_space == "discrete"
            else gymnasium.spaces.MultiDiscrete(self._env.action_space.nvec)
        )
        obs = {}
        for k in self._env.observation_space.spaces.keys():
            if isinstance(self._env.observation_space[k], gym.spaces.Box):
                low = self._env.observation_space[k].low
                high = self._env.observation_space[k].high
                shape = self._env.observation_space[k].shape
                dtype = self._env.observation_space[k].dtype
            elif isinstance(self._env.observation_space[k], gym.spaces.Discrete):
                low = 0
                high = self._env.observation_space[k].n - 1
                shape = (1,)
                dtype = np.int32
            else:
                raise RuntimeError(f"Invalid observation space, got: {type(self._env.observation_space[k])}")
            obs[k] = gymnasium.spaces.Box(low, high, shape, dtype)
        self.observation_space = gymnasium.spaces.Dict(obs)

    def _convert_obs(self, obs: Dict[str, Union[int, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            k: (np.array(v) if not isinstance(v, np.ndarray) else v).reshape(self.observation_space[k].shape)
            for k, v in obs.items()
        }

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, infos = self._env.step(action)
        infos["env_domain"] = "DIAMBRA"
        return self._convert_obs(obs), reward, done, False, infos

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        return self._convert_obs(self._env.reset()), {"env_domain": "DIAMBRA"}
