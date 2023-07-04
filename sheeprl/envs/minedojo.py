"""Adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py"""

import copy
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
import minedojo
import numpy as np
from gymnasium import core
from minedojo.sim import ALL_CRAFT_SMELT_ITEMS, ALL_ITEMS

N_ALL_ITEMS = len(ALL_ITEMS)
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
    12: np.array([0, 0, 0, 12, 12, 1, 0, 0]),  # use
    13: np.array([0, 0, 0, 12, 12, 2, 0, 0]),  # drop
    14: np.array([0, 0, 0, 12, 12, 3, 0, 0]),  # attack
    15: np.array([0, 0, 0, 12, 12, 4, 0, 0]),  # craft
    16: np.array([0, 0, 0, 12, 12, 5, 0, 0]),  # equip
    17: np.array([0, 0, 0, 12, 12, 6, 0, 0]),  # place
    18: np.array([0, 0, 0, 12, 12, 7, 0, 0]),  # destroy
}
ITEM_ID_TO_NAME = dict(enumerate(ALL_ITEMS))
ITEM_NAME_TO_ID = dict(zip(ALL_ITEMS, range(N_ALL_ITEMS)))


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
        self._break_speed_multiplier = kwargs.pop("start_position", 100)
        self._start_pos = copy.deepcopy(self._pos)
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
            break_speed_multiplier=self._break_speed_multiplier,
            **kwargs,
        )
        self._inventory = {}
        self._inventory_names = None
        self.action_space = gym.spaces.MultiDiscrete(
            np.array([len(ACTION_MAP.keys()), len(ALL_CRAFT_SMELT_ITEMS), N_ALL_ITEMS])
        )
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, self._env.observation_space["rgb"].shape, np.uint8),
                "inventory": gym.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float32),
                "equipment": gym.spaces.Box(0.0, 1.0, (N_ALL_ITEMS,), np.int32),
                "life_stats": gym.spaces.Box(0.0, np.array([20.0, 20.0, 300.0]), (3,), np.float32),
                "masks": gym.spaces.Dict(
                    {
                        "action_type": gym.spaces.Box(0, 1, (len(ACTION_MAP),), np.bool_),
                        "equip/place": gym.spaces.Box(0, 1, (N_ALL_ITEMS,), np.bool_),
                        "desrtoy": gym.spaces.Box(0, 1, (N_ALL_ITEMS,), np.bool_),
                        "craft_smelt": gym.spaces.Box(0, 1, (len(ALL_CRAFT_SMELT_ITEMS),), np.bool_),
                    }
                ),
            }
        )
        self.render_mode: str = "rgb_array"
        self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _convert_inventory(self, inventory: Dict[str, Any]) -> np.ndarray:
        converted_inventory = np.zeros(N_ALL_ITEMS)
        self._inventory = {}
        self._inventory_names = inventory["name"].copy()
        for i, (item, quantity) in enumerate(zip(inventory["name"], inventory["quantity"])):
            if item not in self._inventory:
                self._inventory[item] = [i]
            else:
                self._inventory[item].append(i)
            if item == "air":
                converted_inventory[ITEM_NAME_TO_ID[item]] += 1
            else:
                converted_inventory[ITEM_NAME_TO_ID[item]] += quantity
        return converted_inventory

    def _convert_equipment(self, equipment: Dict[str, Any]) -> np.ndarray:
        equip = np.zeros(N_ALL_ITEMS, dtype=np.int32)
        equip[ITEM_NAME_TO_ID[equipment["name"][0]]] = 1
        return equip

    def _convert_masks(self, masks: Dict[str, Any]) -> Dict[str, np.ndarray]:
        equip_mask = np.array([False] * N_ALL_ITEMS)
        destroy_mask = np.array([False] * N_ALL_ITEMS)
        for item, eqp_mask, dst_mask in zip(self._inventory_names, masks["equip"], masks["destroy"]):
            idx = ITEM_NAME_TO_ID[item]
            equip_mask[idx] = eqp_mask
            destroy_mask[idx] = dst_mask
        return {
            "action_type": np.concatenate((np.array([True] * 12), masks["action_type"][1:])),
            "equip/place": equip_mask,
            "desrtoy": destroy_mask,
            "craft_smelt": masks["craft_smelt"],
        }

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        converted_action = ACTION_MAP[int(action[0])].copy()
        if self._sticky_attack:
            if converted_action[5] == 3:
                self._sticky_attack_counter = self._sticky_attack
            if self._sticky_attack_counter > 0:
                converted_action[5] = 3
                converted_action[2] = 0
                self._sticky_attack_counter -= 1
        if self._sticky_jump:
            if converted_action[2] == 1:
                self._sticky_jump_counter = self._sticky_jump
            if self._sticky_jump_counter > 0:
                converted_action[2] = 1
                if converted_action[0] == converted_action[1] == 0:
                    converted_action[0] = 1
                self._sticky_jump_counter -= 1

        converted_action[6] = int(action[1]) if converted_action[5] == 4 else 0
        if converted_action[5] == 5 or converted_action[5] == 6 or converted_action[5] == 7:
            converted_action[7] = self._inventory[ITEM_ID_TO_NAME[int(action[2])]][0]
        else:
            converted_action[7] = 0
        return converted_action

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {
            "rgb": obs["rgb"].copy(),
            "inventory": self._convert_inventory(obs["inventory"]),
            "equipment": self._convert_equipment(obs["equipment"]),
            "life_stats": np.concatenate(
                (obs["life_stats"]["life"], obs["life_stats"]["food"], obs["life_stats"]["oxygen"])
            ),
            "masks": self._convert_masks(obs["masks"]),
        }

    def seed(self, seed: Optional[int] = None) -> None:
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, action: np.ndarray) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
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
            "action": a.tolist(),
            "biomeid": float(obs["location_stats"]["biome_id"].item()),
        }
        return self._convert_obs(obs), reward, done, False, info

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
        return self._convert_obs(obs), {
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
