"""Adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py"""

import copy
from typing import Any, Dict, Optional, Tuple

import gym
import gymnasium
import minerl
import numpy as np
from gymnasium import core

ALL_ITEMS = minerl.herobraine.hero.mc.ALL_ITEMS
ACTION_MAP = {
    0: {
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # no-op
    1: {
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "forward": 1,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # forward
    2: {
        "attack": 0,
        "back": 1,
        "camera": (0, 0),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # back
    3: {
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "forward": 0,
        "jump": 0,
        "left": 1,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # left
    4: {
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 1,
        "sneak": 0,
        "sprint": 0,
    },  # right
    5: {
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "forward": 1,
        "jump": 1,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # jump + forward
    6: {
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "forward": 1,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 1,
        "sprint": 0,
    },  # sneak + forward
    7: {
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "forward": 1,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 1,
    },  # sprint + forward
    8: {
        "attack": 0,
        "back": 0,
        "camera": np.array([-15, 0]),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # pitch down (-15)
    9: {
        "attack": 0,
        "back": 0,
        "camera": np.array([15, 0]),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # pitch up (+15)
    10: {
        "attack": 0,
        "back": 0,
        "camera": np.array([0, -15]),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # yaw down (-15)
    11: {
        "attack": 0,
        "back": 0,
        "camera": np.array([0, 15]),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "none",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # yaw up (+15)
    12: {
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "dirt",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # place
    13: {
        "attack": 1,
        "back": 0,
        "camera": (0, 0),
        "forward": 0,
        "jump": 0,
        "left": 0,
        "place": "dirt",
        "right": 0,
        "sneak": 0,
        "sprint": 0,
    },  # attack
}


class MineRLWrapper(core.Env):
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
        self._sticky_attack = sticky_attack
        self._sticky_jump = sticky_jump
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0

        # create task
        self._env = gym.make(id=task_id)
        # inventory
        self._inventory = {}
        self._inventory_names = None
        # action and observations space
        self._action_space = gymnasium.spaces.Discrete(len(ACTION_MAP.keys()))
        self._observation_space = gymnasium.spaces.Dict(
            {
                "rgb": gymnasium.spaces.Box(0, 255, (3, 64, 64), np.uint8),
                "inventory": gymnasium.spaces.Dict({"dirt": gymnasium.spaces.Box(low=0, high=2304, shape=())}),
                "compass": gymnasium.spaces.Dict({"angle": gymnasium.spaces.Box(low=-180.0, high=180.0, shape=())}),
            }
        )
        self._pos = {
            "pitch": 0.0,
            "yaw": 0.0,
        }
        # render
        self._render_mode: str = "rgb_array"
        # set seed
        self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def render_mode(self) -> str:
        return self._render_mode

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        converted_action = copy.deepcopy(ACTION_MAP[int(action.item())])
        if self._sticky_attack:
            if converted_action["attack"] == 1:
                self._sticky_attack_counter = self._sticky_attack
            if self._sticky_attack_counter > 0:
                converted_action["attack"] = 1
                converted_action["jump"] = 0
                self._sticky_attack_counter -= 1
        if self._sticky_jump:
            if converted_action["jump"] == 1:
                self._sticky_jump_counter = self._sticky_jump
            if self._sticky_jump_counter > 0:
                converted_action["jump"] = 1
                if converted_action["forward"] == converted_action["back"] == 0:
                    converted_action["forward"] = 1
                self._sticky_jump_counter -= 1
        return converted_action

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {
            "rgb": obs["pov"].copy().transpose(1, 2, 0),
            "inventory": obs["inventory"],
            "compass": obs["compass"],
        }

    def seed(self, seed: Optional[int] = None) -> None:
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        action = self._convert_action(action)
        next_pitch = self._pos["pitch"] + action["camera"][0]
        next_yaw = ((self._pos["yaw"] + action["camera"][1]) + 180) % 360 - 180
        if not (self._pitch_limits[0] <= next_pitch <= self._pitch_limits[1]):
            action[3] = action["camera"][0] = 0
            next_pitch = self._pos["pitch"]

        obs, reward, done, info = self._env.step(action)
        self._pos = {
            "pitch": next_pitch,
            "yaw": next_yaw,
        }
        info = {}
        return self._convert_obs(obs), reward, done, False, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = self._env.reset()
        self._pos = {
            "pitch": 0.0,
            "yaw": 0.0,
        }
        return self._convert_obs(obs), {}

    def close(self):
        self._env.close()
        return super().close()
