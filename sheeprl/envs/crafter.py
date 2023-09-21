from sheeprl.utils.imports import _IS_CRAFTER_AVAILABLE

if not _IS_CRAFTER_AVAILABLE:
    raise ModuleNotFoundError(_IS_CRAFTER_AVAILABLE)

from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import crafter
import numpy as np
from gymnasium import core, spaces
from gymnasium.core import RenderFrame


class CrafterWrapper(core.Env):
    def __init__(self, id: str, screen_size: Union[int, Tuple[int, int]] = 64, seed: Optional[int] = None) -> None:
        assert id in {"reward", "nonreward"}
        if isinstance(screen_size, int):
            screen_size = (screen_size,) * 2

        self._env = crafter.Env(size=screen_size, seed=seed, reward=(id == "reward"))
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    self._env.observation_space.low,
                    self._env.observation_space.high,
                    self._env.observation_space.shape,
                    self._env.observation_space.dtype,
                )
            }
        )
        self.action_space = spaces.Discrete(self._env.action_space.n)
        self.reward_range = self._env.reward_range or (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        # render
        self._render_mode: str = "rgb_array"

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"rgb": obs}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        return self._convert_obs(obs), reward, done, False, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs = self._env.reset()
        return self._convert_obs(obs), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self._env.render()

    def close(self) -> None:
        return super().close()
