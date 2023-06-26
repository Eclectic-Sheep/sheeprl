"""Adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py"""

import copy
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import minedojo
import numpy as np
from gymnasium import core

ACTION_MAP = {
    0: np.array([0, 0, 0, 12, 12, 0, 0, 0]),  # no-op
    1: np.array([1, 0, 0, 12, 12, 0, 0, 0]),  # forward
    2: np.array([2, 0, 0, 12, 12, 0, 0, 0]),  # back
    3: np.array([0, 1, 0, 12, 12, 0, 0, 0]),  # left
    4: np.array([0, 2, 0, 12, 12, 0, 0, 0]),  # right
    5: np.array([1, 0, 1, 12, 12, 0, 0, 0]),  # jump + forward
    6: np.array([1, 0, 2, 12, 12, 0, 0, 0]),  # sneak + forward
    7: np.array([1, 0, 3, 12, 12, 0, 0, 0]),  # sprint + forward
    8: np.array([0, 0, 0, 11, 12, 0, 0, 0]),  # pitch down (-15)
    9: np.array([0, 0, 0, 13, 12, 0, 0, 0]),  # pitch up (+15)
    10: np.array([0, 0, 0, 12, 11, 0, 0, 0]),  # yaw down (-15)
    11: np.array([0, 0, 0, 12, 13, 0, 0, 0]),  # yaw up (+15)
    12: np.array([0, 0, 0, 12, 12, 3, 0, 0]),  # attack
}


class MineDojoWrapper(core.Env):
    def __init__(
        self,
        task_id: str,
        height: int = 64,
        width: int = 64,
        pitch_limits: Tuple[int, int] = (-60, 60),
        seed: Optional[int] = None,
        sticky_attack: Optional[int] = 30,
        sticky_jump: Optional[int] = 10,
        **kwargs: Optional[Dict[Any, Any]],
    ):
        self._height = height
        self._width = width
        self._pitch_limits = pitch_limits
        self._pos = kwargs.pop("start_position", None)
        self._start_pos = copy.deepcopy(self._pos)
        self._action_space = gym.spaces.Discrete(len(ACTION_MAP.keys()))
        self._sticky_attack = sticky_attack
        self._sticky_jump = sticky_jump
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0

        if self._pos is not None and not (self._pitch_limits[0] <= self._pos["pitch"] <= self._pitch_limits[1]):
            raise ValueError(
                f"The initial position must respect the pitch limits {self._pitch_limits}, given {self._pos['pitch']}"
            )

        # create task
        self._env = minedojo.make(
            task_id=task_id,
            image_size=(height, width),
            world_seed=seed,
            start_position=self._pos,
            generate_world_type="default",
            allow_mob_spawn=False,
            allow_time_passage=False,
            fast_reset=True,
            **kwargs,
        )
        # render
        self._render_mode: str = "rgb_array"
        # set seed
        self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        action = copy.deepcopy(ACTION_MAP[int(action)])
        if self._sticky_attack:
            if action[5] == 3:
                self._sticky_attack_counter = self._sticky_attack
            if self._sticky_attack_counter > 0:
                action[5] = 3
                action[2] = 0
                self._sticky_attack_counter -= 1
        if self._sticky_jump:
            if action[2] == 1:
                self._sticky_jump_counter = self._sticky_jump
            if self._sticky_jump_counter > 0:
                action[2] = 1
                if action[0] == action[1] == 0:
                    action[0] = 1
                self._sticky_jump_counter -= 1
        return action

    @property
    def render_mode(self) -> str:
        return self._render_mode

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._env.observation_space["rgb"]

    def seed(self, seed: Optional[int] = None) -> None:
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        a = action
        action = self._convert_action(action)
        next_pitch = self._pos["pitch"] + (action[3] - 12) * 15
        if not (self._pitch_limits[0] <= next_pitch <= self._pitch_limits[1]):
            action[3] = 12

        obs, reward, done, info = self._env.step(action)
        self._pos = {
            "x": float(obs["location_stats"]["pos"][0]),
            "y": float(obs["location_stats"]["pos"][1]),
            "z": float(obs["location_stats"]["pos"][2]),
            "pitch": float(obs["location_stats"]["pitch"].item()),
            "yaw": float(obs["location_stats"]["yaw"].item()),
        }
        info = {
            "life_stats": {
                "life": float(obs["life_stats"]["life"].item()),
                "oxygen": float(obs["life_stats"]["oxygen"].item()),
                "food": float(obs["life_stats"]["food"].item()),
            },
            "location_stats": copy.deepcopy(self._pos),
            "action": int(a.item()),
            "biomeid": float(obs["location_stats"]["biome_id"].item()),
        }
        return obs["rgb"], reward, done, False, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = self._env.reset()
        self._pos = {
            "x": float(obs["location_stats"]["pos"][0]),
            "y": float(obs["location_stats"]["pos"][1]),
            "z": float(obs["location_stats"]["pos"][2]),
            "pitch": float(obs["location_stats"]["pitch"].item()),
            "yaw": float(obs["location_stats"]["yaw"].item()),
        }
        return obs["rgb"], {
            "life_stats": {
                "life": float(obs["life_stats"]["life"].item()),
                "oxygen": float(obs["life_stats"]["oxygen"].item()),
                "food": float(obs["life_stats"]["food"].item()),
            },
            "location_stats": copy.deepcopy(self._pos),
        }

    def close(self):
        self._env.close()
        return super().close()
