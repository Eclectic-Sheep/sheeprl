# adapted from https://github.com/minerllabs/minerl

from sheeprl.utils.imports import _IS_MINERL_0_4_4_AVAILABLE

if not _IS_MINERL_0_4_4_AVAILABLE:
    raise ModuleNotFoundError(_IS_MINERL_0_4_4_AVAILABLE)

from abc import ABC
from typing import List

from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler, handlers
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.mc import INVERSE_KEYMAP

SIMPLE_KEYBOARD_ACTION = ["forward", "back", "left", "right", "jump", "sneak", "sprint", "attack"]


class CustomSimpleEmbodimentEnvSpec(EnvSpec, ABC):
    """
    A simple base environment from which all other simple envs inherit.
    """

    def __init__(self, name, *args, resolution=(64, 64), break_speed: int = 100, **kwargs):
        self.resolution = resolution
        self.break_speed = break_speed
        super().__init__(name, *args, **kwargs)

    def create_agent_start(self):
        return [BreakSpeedMultiplier(self.break_speed)]

    def create_observables(self) -> List[TranslationHandler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromLifeStats(),
        ]

    def create_actionables(self) -> List[TranslationHandler]:
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        return [
            handlers.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items() if k in SIMPLE_KEYBOARD_ACTION
        ] + [handlers.CameraAction()]

    def create_monitors(self) -> List[TranslationHandler]:
        return []  # No base monitor needed


# adapted from https://github.com/danijar/diamond_env
class BreakSpeedMultiplier(handler.Handler):
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def to_string(self):
        return f"break_speed({self.multiplier})"

    def xml_template(self):
        return "<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>"
