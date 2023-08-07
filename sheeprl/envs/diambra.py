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
        env_id: str,
        action_space: str = "discrete",
        screen_size: Union[int, Tuple[int, int]] = 64,
        grayscale: bool = False,
        attack_but_combination: str = True,
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
            "action_space": action_space,
            "attack_but_combination": attack_but_combination,
            **diambra_settings,
        }
        wrappers = {
            "no_op_max": noop_max,
            "hwc_obs_resize": (*screen_size, (1 if grayscale else 3)),
            "flatten": True,
            "sticky_actions": sticky_actions,
            **diambra_wrappers,
        }
        self._env = diambra.arena.make(env_id, settings, wrappers, seed=seed, rank=rank)

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
            if isinstance(self._env.observation_space[k], gym.spaces.Discrete):
                low = 0
                high = self._env.observation_space[k].n - 1
                shape = (1,)
                dtype = np.int32
            else:
                raise RuntimeError(f"Invalid observation space, got: {type(self._env.observation_space[k])}")
            obs[k] = gymnasium.spaces.Box(low, high, shape, dtype)
        self.observation_space = gymnasium.spaces.Dict(obs)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, infos = self._env.step(action)
        obs = {
            k: (np.array(v) if not isinstance(v, np.ndarray) else v).reshape(self.observation_space[k].shape)
            for k, v in obs.items()
        }
        return obs, reward, done, False, infos

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, Dict[str, Any]]:
        return self._env.reset(seed=seed, options=options), {}
