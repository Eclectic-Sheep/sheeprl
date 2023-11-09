from __future__ import annotations

from sheeprl.utils.imports import _IS_MINEDOJO_AVAILABLE

if not _IS_MINEDOJO_AVAILABLE:
    raise ModuleNotFoundError(_IS_MINEDOJO_AVAILABLE)

import copy
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
import minedojo
import numpy as np
from gymnasium.core import RenderFrame
from minedojo.sim import ALL_CRAFT_SMELT_ITEMS, ALL_ITEMS
from minedojo.sim.wrappers.ar_nn import ARNNWrapper

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

# Minedojo functional actions:
# 0: noop
# 1: use
# 2: drop
# 3: attack
# 4: craft
# 5: equip
# 6: place
# 7: destroy


class MineDojoWrapper(gym.Wrapper):
    def __init__(
        self,
        id: str,
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
        self._break_speed_multiplier = kwargs.pop("break_speed_multiplier", 100)
        self._start_pos = copy.deepcopy(self._pos)
        self._sticky_attack = sticky_attack
        self._sticky_jump = sticky_jump
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0

        if self._pos is not None and not (self._pitch_limits[0] <= self._pos["pitch"] <= self._pitch_limits[1]):
            raise ValueError(
                f"The initial position must respect the pitch limits {self._pitch_limits}, given {self._pos['pitch']}"
            )

        env: ARNNWrapper = minedojo.make(
            task_id=id,
            image_size=(height, width),
            world_seed=seed,
            start_position=self._pos,
            generate_world_type="default",
            fast_reset=True,
            break_speed_multiplier=self._break_speed_multiplier,
            **kwargs,
        )
        super().__init__(env)
        self._inventory = {}
        self._inventory_names = None
        self._inventory_max = np.zeros(N_ALL_ITEMS)
        self.action_space = gym.spaces.MultiDiscrete(
            np.array([len(ACTION_MAP.keys()), len(ALL_CRAFT_SMELT_ITEMS), N_ALL_ITEMS])
        )
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, self.env.observation_space["rgb"].shape, np.uint8),
                "inventory": gym.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float32),
                "inventory_max": gym.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float32),
                "inventory_delta": gym.spaces.Box(-np.inf, np.inf, (N_ALL_ITEMS,), np.float32),
                "equipment": gym.spaces.Box(0.0, 1.0, (N_ALL_ITEMS,), np.int32),
                "life_stats": gym.spaces.Box(0.0, np.array([20.0, 20.0, 300.0]), (3,), np.float32),
                "mask_action_type": gym.spaces.Box(0, 1, (len(ACTION_MAP),), bool),
                "mask_equip_place": gym.spaces.Box(0, 1, (N_ALL_ITEMS,), bool),
                "mask_destroy": gym.spaces.Box(0, 1, (N_ALL_ITEMS,), bool),
                "mask_craft_smelt": gym.spaces.Box(0, 1, (len(ALL_CRAFT_SMELT_ITEMS),), bool),
            }
        )
        self._render_mode: str = "rgb_array"
        self.seed(seed=seed)

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _convert_inventory(self, inventory: Dict[str, Any]) -> np.ndarray:
        # the inventory counts, as a vector with one entry for each Minecraft item
        converted_inventory = np.zeros(N_ALL_ITEMS)
        self._inventory = {}  # map for each item the position in the inventory
        self._inventory_names = np.array(
            ["_".join(item.split(" ")) for item in inventory["name"].copy().tolist()]
        )  # names of the objects in the inventory
        for i, (item, quantity) in enumerate(zip(inventory["name"], inventory["quantity"])):
            item = "_".join(item.split(" "))
            # save all the position of the items in the inventory
            if item not in self._inventory:
                self._inventory[item] = [i]
            else:
                self._inventory[item].append(i)
            # count the items in the inventory
            if item == "air":
                converted_inventory[ITEM_NAME_TO_ID[item]] += 1
            else:
                converted_inventory[ITEM_NAME_TO_ID[item]] += quantity
        self._inventory_max = np.maximum(converted_inventory, self._inventory_max)
        return converted_inventory

    def _convert_inventory_delta(self, inventory_delta: Dict[str, Any]) -> np.ndarray:
        # the inventory counts, as a vector with one entry for each Minecraft item
        converted_inventory_delta = np.zeros(N_ALL_ITEMS)
        for item, quantity in zip(inventory_delta["inc_name_by_craft"], inventory_delta["inc_quantity_by_craft"]):
            item = "_".join(item.split(" "))
            converted_inventory_delta[ITEM_NAME_TO_ID[item]] += quantity
        for item, quantity in zip(inventory_delta["dec_name_by_craft"], inventory_delta["dec_quantity_by_craft"]):
            item = "_".join(item.split(" "))
            converted_inventory_delta[ITEM_NAME_TO_ID[item]] -= quantity
        for item, quantity in zip(inventory_delta["inc_name_by_other"], inventory_delta["inc_quantity_by_other"]):
            item = "_".join(item.split(" "))
            converted_inventory_delta[ITEM_NAME_TO_ID[item]] += quantity
        for item, quantity in zip(inventory_delta["dec_name_by_other"], inventory_delta["dec_quantity_by_other"]):
            item = "_".join(item.split(" "))
            converted_inventory_delta[ITEM_NAME_TO_ID[item]] -= quantity
        return converted_inventory_delta

    def _convert_equipment(self, equipment: Dict[str, Any]) -> np.ndarray:
        equip = np.zeros(N_ALL_ITEMS, dtype=np.int32)
        equip[ITEM_NAME_TO_ID["_".join(equipment["name"][0].split(" "))]] = 1
        return equip

    def _convert_masks(self, masks: Dict[str, Any]) -> Dict[str, np.ndarray]:
        equip_mask = np.array([False] * N_ALL_ITEMS)
        destroy_mask = np.array([False] * N_ALL_ITEMS)
        for item, eqp_mask, dst_mask in zip(self._inventory_names, masks["equip"], masks["destroy"]):
            idx = ITEM_NAME_TO_ID[item]
            equip_mask[idx] = eqp_mask
            destroy_mask[idx] = dst_mask
        masks["action_type"][5:7] *= np.any(equip_mask).item()
        masks["action_type"][7] *= np.any(destroy_mask).item()
        return {
            "mask_action_type": np.concatenate((np.array([True] * 12), masks["action_type"][1:])),
            "mask_equip_place": equip_mask,
            "mask_destroy": destroy_mask,
            "mask_craft_smelt": masks["craft_smelt"],
        }

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        converted_action = ACTION_MAP[int(action[0])].copy()
        if self._sticky_attack:
            # 5 is the index of the functional actions (e.g., use, attack, equip, ...)
            # 3 is the value for the attack action
            if converted_action[5] == 3:
                self._sticky_attack_counter = self._sticky_attack - 1
            # if sticky attack is greater than zero and the agent did not select a functional action
            # (0 in the position 5 of the converted actions), then the attack action is repeated
            if self._sticky_attack_counter > 0 and converted_action[5] == 0:
                converted_action[5] = 3
                self._sticky_attack_counter -= 1
            # it the selected action is not attack, then the agent stops the sticky attack
            elif converted_action[5] != 3:
                self._sticky_attack = 0
        if self._sticky_jump:
            # 2 is the index of the jump/sneak/sprint actions, 1 is the value for the jump action
            if converted_action[2] == 1:
                self._sticky_jump_counter = self._sticky_jump - 1
            # if sticky jump is greater than zero and the agent did not select a jump/sneak/sprint action
            # (0 in the position 2 of the converted actions), then the jump action is repeated
            if self._sticky_jump_counter > 0 and converted_action[0] == 0:
                converted_action[2] = 1
                # if the agent jumps because of the sticky action, then it goes forward if only if
                # it has not chosen another movement action (all 0 in the indices 0 and 1 of the converted actions)
                if converted_action[0] == converted_action[1] == 0:
                    converted_action[0] = 1
                self._sticky_jump_counter -= 1
            # it the selected action is not jump, then the agent stops the sticky jump
            elif converted_action[2] != 1:
                self._sticky_jump_counter = 0
        # if the agent selects the craft action (value 4 in index 5 of the converted actions),
        # then it also selects the element to craft
        converted_action[6] = int(action[1]) if converted_action[5] == 4 else 0
        # if the agent selects the equip/place/destroy action (value 5 or 6 or 7 in index 5 of the converted actions),
        # then it also selects the element to equip/place/destroy
        if converted_action[5] in {5, 6, 7}:
            converted_action[7] = self._inventory[ITEM_ID_TO_NAME[int(action[2])]][0]
        else:
            converted_action[7] = 0
        return converted_action

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {
            "rgb": obs["rgb"].copy(),
            "inventory": self._convert_inventory(obs["inventory"]),
            "inventory_max": self._inventory_max,
            "inventory_delta": self._convert_inventory_delta(obs["delta_inv"]),
            "equipment": self._convert_equipment(obs["equipment"]),
            "life_stats": np.concatenate(
                (obs["life_stats"]["life"], obs["life_stats"]["food"], obs["life_stats"]["oxygen"])
            ),
            **self._convert_masks(obs["masks"]),
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

        obs, reward, done, info = self.env.step(action)
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
        obs = self.env.reset()
        self._pos = {
            "x": float(obs["location_stats"]["pos"][0]),
            "y": float(obs["location_stats"]["pos"][1]),
            "z": float(obs["location_stats"]["pos"][2]),
            "pitch": float(obs["location_stats"]["pitch"].item()),
            "yaw": float(obs["location_stats"]["yaw"].item()),
        }
        self._sticky_jump_counter = 0
        self._sticky_attack_counter = 0
        self._inventory_max = np.zeros(N_ALL_ITEMS)
        return self._convert_obs(obs), {
            "life_stats": {
                "life": float(obs["life_stats"]["life"].item()),
                "oxygen": float(obs["life_stats"]["oxygen"].item()),
                "food": float(obs["life_stats"]["food"].item()),
            },
            "location_stats": copy.deepcopy(self._pos),
            "biomeid": float(obs["location_stats"]["biome_id"].item()),
        }

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "human":
            super().render()
        elif self.render_mode == "rgb_array":
            if self.env.unwrapped._prev_obs is None:
                return None
            else:
                return self.env.unwrapped._prev_obs["rgb"]
        return None
