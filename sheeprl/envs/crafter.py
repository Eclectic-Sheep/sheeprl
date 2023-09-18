from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import crafter
import numpy as np
from gymnasium import core, spaces
from gymnasium.core import RenderFrame


class CrafterWrapper(core.Env):
    def __init__(self, id: str, screen_size: Union[int, Tuple[int, int]] = 64, seed: Optional[int] = None) -> None:
        super().__init__()
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
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"rgb": obs}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        return self._convert_obs(obs), reward, done, False, info

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Any, Dict[str, Any]]:
        obs = self._env.reset()
        return self._convert_obs(obs), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
        return super().close()
