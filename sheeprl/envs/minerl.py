from sheeprl.utils.imports import _IS_MINERL_0_4_4_AVAILABLE

if not _IS_MINERL_0_4_4_AVAILABLE:
    raise ModuleNotFoundError(_IS_MINERL_0_4_4_AVAILABLE)

import copy
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium
import minerl
import numpy as np
from gymnasium import core
from minerl.herobraine.hero import mc

from sheeprl.envs.minerl_envs.navigate import CustomNavigate
from sheeprl.envs.minerl_envs.obtain import CustomObtainDiamond, CustomObtainIronPickaxe

# In order to use the environment as a gym you need to register it with gym
CUSTOM_ENVS = {
    "custom_navigate": CustomNavigate,
    "custom_obtain_diamond": CustomObtainDiamond,
    "custom_obtain_iron_pickaxe": CustomObtainIronPickaxe,
}


N_ALL_ITEMS = len(mc.ALL_ITEMS)
NOOP = {
    "camera": (0, 0),
    "forward": 0,
    "back": 0,
    "left": 0,
    "right": 0,
    "attack": 0,
    "sprint": 0,
    "jump": 0,
    "sneak": 0,
    "craft": "none",
    "nearbyCraft": "none",
    "nearbySmelt": "none",
    "place": "none",
    "equip": "none",
}
ITEM_ID_TO_NAME = dict(enumerate(mc.ALL_ITEMS))
ITEM_NAME_TO_ID = dict(zip(mc.ALL_ITEMS, range(N_ALL_ITEMS)))


class MineRLWrapper(core.Env):
    def __init__(
        self,
        id: str,
        height: int = 64,
        width: int = 64,
        pitch_limits: Tuple[int, int] = (-60, 60),
        seed: Optional[int] = None,
        sticky_attack: Optional[int] = 30,
        sticky_jump: Optional[int] = 10,
        break_speed_multiplier: Optional[int] = 100,
        countvect_obs: bool = True,
        **kwargs: Optional[Dict[Any, Any]],
    ):
        self._height = height
        self._width = width
        self._pitch_limits = pitch_limits
        self._sticky_attack = sticky_attack
        self._sticky_jump = sticky_jump
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._break_speed_multiplier = break_speed_multiplier
        self._countvect_obs = countvect_obs
        if "navigate" not in id.lower():
            kwargs.pop("extreme", None)

        self._env = CUSTOM_ENVS[id.lower()](break_speed=break_speed_multiplier, **kwargs).make()
        self.ACTIONS_MAP = {0: {}}
        act_idx = 1
        for act in self._env.action_space:
            if isinstance(self._env.action_space[act], minerl.herobraine.hero.spaces.Enum):
                act_val = set(self._env.action_space[act].values.tolist()) - {"none"}
                act_len = len(act_val)
            elif act != "camera":
                act_len = 1
                act_val = [1]
            else:
                act_len = 4
                act_val = [
                    np.array([-15, 0]),
                    np.array([15, 0]),
                    np.array([0, -15]),
                    np.array([0, 15]),
                ]
            action = dict(zip((np.arange(act_len) + act_idx).tolist(), [{act: v} for v in act_val]))
            if act in {"jump", "sneak", "sprint"}:
                action[act_idx]["forward"] = 1
            self.ACTIONS_MAP.update(action)
            act_idx += act_len

        # action and observations space
        self.action_space = gymnasium.spaces.Discrete(len(self.ACTIONS_MAP))

        obs_space = {
            "rgb": gymnasium.spaces.Box(0, 255, (3, 64, 64), np.uint8),
            "life_stats": gymnasium.spaces.Box(0.0, np.array([20.0, 20.0, 300.0]), (3,), np.float32),
            "inventory": (
                gymnasium.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float32)
                if countvect_obs
                else gymnasium.spaces.Box(0.0, np.inf, (len(self._env.observation_space["inventory"]),), np.float32)
            ),
            "max_inventory": (
                gymnasium.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float32)
                if countvect_obs
                else gymnasium.spaces.Box(0.0, np.inf, (len(self._env.observation_space["inventory"]),), np.float32)
            ),
        }
        if "compass" in self._env.observation_space.spaces:
            obs_space["compass"] = gymnasium.spaces.Box(-180, 180, (1,), np.float32)
        if "equipped_items" in self._env.observation_space.spaces:
            obs_space["equipment"] = (
                gymnasium.spaces.Box(0.0, 1.0, (N_ALL_ITEMS,), np.int32)
                if countvect_obs
                else gymnasium.spaces.Box(
                    0.0,
                    1.0,
                    (len(self._env.observation_space["equipped_items"]["mainhand"]["type"].values.tolist()),),
                    np.int32,
                )
            )

        # Mapping from names to ids (index in the vector)
        if not countvect_obs:
            self.inventory_size = obs_space["inventory"].shape[0]
            self.inventory_item_to_id = dict(zip(self._env.observation_space["inventory"], range(self.inventory_size)))
            if "equipment" in obs_space:
                self.equip_size = obs_space["equipment"].shape[0]
                self.equip_item_to_id = dict(
                    zip(
                        self._env.observation_space["equipped_items"]["mainhand"]["type"].values.tolist(),
                        range(self.equip_size),
                    )
                )
        else:
            self.inventory_item_to_id = ITEM_NAME_TO_ID
            self.inventory_size = N_ALL_ITEMS
            if "equipment" in obs_space:
                self.equip_item_to_id = ITEM_NAME_TO_ID
                self.equip_size = N_ALL_ITEMS
        self.observation_space = gymnasium.spaces.Dict(obs_space)
        self._pos = {
            "pitch": 0.0,
            "yaw": 0.0,
        }
        self._max_inventory = np.zeros(self.inventory_size)
        self.render_mode: str = "rgb_array"
        self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _convert_actions(self, action: np.ndarray) -> Dict[str, Any]:
        converted_actions = copy.deepcopy(NOOP)
        converted_actions.update(self.ACTIONS_MAP[action.item()])
        if self._sticky_attack:
            if converted_actions["attack"]:
                self._sticky_attack_counter = self._sticky_attack
            if self._sticky_attack_counter > 0:
                converted_actions["attack"] = 1
                converted_actions["jump"] = 0
                self._sticky_attack_counter -= 1
        if self._sticky_jump:
            if converted_actions["jump"]:
                self._sticky_jump_counter = self._sticky_jump
            if self._sticky_jump_counter > 0:
                converted_actions["jump"] = 1
                converted_actions["forward"] = 1
                self._sticky_jump_counter -= 1
        return converted_actions

    def _convert_equipment(self, equipment: Dict[str, Any]) -> np.ndarray:
        equip = np.zeros(self.equip_size, dtype=np.int32)
        equip[self.equip_item_to_id[equipment["mainhand"]["type"]]] = 1
        return equip

    def _convert_inventory(self, inventory: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # the inventory counts, as a vector with one entry for each Minecraft item
        converted_inventory = {"inventory": np.zeros(self.inventory_size)}
        for i, (item, quantity) in enumerate(inventory.items()):
            # count the items in the inventory
            if item == "air":
                converted_inventory["inventory"][self.inventory_item_to_id[item]] += 1
            else:
                converted_inventory["inventory"][self.inventory_item_to_id[item]] += quantity
        converted_inventory["max_inventory"] = np.maximum(converted_inventory["inventory"], self._max_inventory)
        self._max_inventory = converted_inventory["max_inventory"].copy()
        return converted_inventory

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        converted_obs = {
            "rgb": obs["pov"].copy().transpose(2, 0, 1),
            "life_stats": np.array(
                [obs["life_stats"]["life"], obs["life_stats"]["food"], obs["life_stats"]["air"]], dtype=np.float32
            ),
            **self._convert_inventory(obs["inventory"]),
        }
        if "equipment" in self.observation_space.spaces:
            converted_obs["equipment"] = self._convert_equipment(obs["equipped_items"])
        if "compass" in self.observation_space.spaces:
            converted_obs["compass"] = obs["compass"]["angle"].reshape(-1)
        return converted_obs

    def seed(self, seed: Optional[int] = None) -> None:
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]]:
        converted_actions = self._convert_actions(actions)
        next_pitch = self._pos["pitch"] + converted_actions["camera"][0]
        next_yaw = ((self._pos["yaw"] + converted_actions["camera"][1]) + 180) % 360 - 180
        if not (self._pitch_limits[0] <= next_pitch <= self._pitch_limits[1]):
            converted_actions["camera"] = np.array([0, converted_actions["camera"][1]])
            next_pitch = self._pos["pitch"]

        obs, reward, done, info = self._env.step(converted_actions)
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
        self._max_inventory = np.zeros(self.inventory_size)
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pos = {
            "pitch": 0.0,
            "yaw": 0.0,
        }
        return self._convert_obs(obs), {}

    def render(self, mode: Optional[str] = "rgb_array"):
        return self._env.render(self.render_mode)

    def close(self):
        self._env.close()
        return super().close()
