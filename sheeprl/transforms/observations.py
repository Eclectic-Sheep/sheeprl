"""The pipelines of the observation keys a configuration declares (`algo.inputs`, `algo.extra_keys`): their `frame`
stages, run in the environments (`TransformsWrapper`), and their `envs` stages (`EnvsTransforms`), run by
`GymEnvironment` and by the test."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from torch import nn

from sheeprl.transforms.base import Pipeline, instantiate_transforms

if TYPE_CHECKING:
    from lightning import Fabric

    from sheeprl.utils.utils import dotdict


def input_keys(cfg: dotdict) -> Tuple[str, ...]:
    """The keys of the inputs (`algo.inputs`), in the order of the configuration, each once."""
    keys: List[str] = []
    for spec in (cfg.algo.get("inputs") or {}).values():
        keys += [k for k in (spec or {}).get("keys") or [] if k not in keys]
    return tuple(keys)


def stored_keys(cfg: dotdict) -> Tuple[str, ...]:
    """The keys the environments keep and the store holds: the ones of the inputs, then the extra keys."""
    keys = list(input_keys(cfg))
    return tuple(keys + [k for k in cfg.algo.get("extra_keys") or [] if k not in keys])


def _stored_stages(transforms_cfg: List[Any]) -> List[Any]:
    """The configurations of the transforms that run before the store (`frame` and `envs`)."""
    return [cfg for cfg, t in zip(transforms_cfg, instantiate_transforms(transforms_cfg)) if t.where != "batch"]


def key_transforms(cfg: dotdict) -> Dict[str, List[Any]]:
    """The configured transforms of every stored key: the ones of the first input that lists it (none for an extra key).

    A key is stored once: listed in several inputs, its `frame` and `envs` transforms must be the same in all of them,
    only the `batch` ones may differ.
    """
    transforms: Dict[str, List[Any]] = {}
    owners: Dict[str, str] = {}
    for name, spec in (cfg.algo.get("inputs") or {}).items():
        spec = spec or {}
        for k in spec.get("keys") or []:
            key_cfg = list(spec.get("transforms") or [])
            if k not in transforms:
                transforms[k], owners[k] = key_cfg, name
            elif _stored_stages(transforms[k]) != _stored_stages(key_cfg):
                raise ValueError(
                    f"The key `{k}` is listed in the inputs `{owners[k]}` and `{name}` with different `frame` or "
                    "`envs` transforms: it is stored once, so only their `batch` transforms may differ"
                )
    for k in cfg.algo.get("extra_keys") or []:
        transforms.setdefault(k, [])
    return transforms


def check_keys(cfg: dotdict, space: gym.spaces.Dict) -> None:
    """Every key the configuration declares must be an observation of the environment."""
    missing = [k for k in stored_keys(cfg) if k not in space.spaces]
    if not stored_keys(cfg):
        raise ValueError(
            "No observation key is declared: set the keys of an input, e.g. `algo.inputs.vector.keys=[state]` or "
            f"`algo.inputs.image.keys=[rgb]`. The observations of the environment are {list(space.spaces)}"
        )
    if missing:
        raise ValueError(
            f"The observation keys {missing} declared by `algo.inputs` or `algo.extra_keys` are not observations of "
            f"the environment, whose keys are {list(space.spaces)}. A bare observation is the key `state`; the "
            "render is added under `env.render_key` when set"
        )


def frame_pipelines(cfg: dotdict, space: gym.spaces.Dict) -> Dict[str, Pipeline]:
    """The `frame` stage of every stored key, from the observation space of the environment."""
    check_keys(cfg, space)
    return {k: Pipeline.from_config(t, space[k], "frame", "frame") for k, t in key_transforms(cfg).items()}


class EnvsTransforms(nn.Module):
    """The `envs` stage of every stored key: what `GymEnvironment` applies to the observations of its environments
    (with `update`, adding them to the statistics) and the test to its own (frozen). Its state is the one of the
    transforms with statistics, by key."""

    def __init__(self, pipelines: Dict[str, Pipeline]) -> None:
        super().__init__()
        self.keys = tuple(pipelines)
        # A list: the observation keys are any strings, not all valid names of modules
        self.pipelines = nn.ModuleList(pipelines.values())

    @classmethod
    def from_config(cls, cfg: dotdict, frame_space: gym.spaces.Dict, off_policy: bool = False) -> EnvsTransforms:
        """The `envs` stages of the stored keys, from the space the environments report (after their `frame` stage).
        A transform that learns from the data (e.g. `Standardize`) is an error for an off-policy algorithm, whose
        replay buffer would keep data transformed with statistics that change during the training."""
        pipelines = {
            k: Pipeline.from_config(t, frame_space[k], "envs", "envs")
            for k, t in key_transforms(cfg).items()
            if k in frame_space.spaces
        }
        learning = [k for k, p in pipelines.items() if p.learns_from_data]
        if off_policy and learning:
            raise ValueError(
                f"The `envs` transforms of the keys {learning} learn statistics from the data (e.g. `Standardize`): "
                "they are supported only by on-policy algorithms, as the replay buffer of an off-policy algorithm "
                "would keep observations transformed with statistics that change during the training"
            )
        return cls(pipelines)

    def _pipelines(self) -> Dict[str, Pipeline]:
        return dict(zip(self.keys, self.pipelines))

    @property
    def space(self) -> gym.spaces.Dict:
        """The spaces of the stored keys after their `envs` stage."""
        return gym.spaces.Dict({k: p.envs_space for k, p in self._pipelines().items()})

    def envs(self, obs: Dict[str, np.ndarray], update: bool = True) -> Dict[str, np.ndarray]:
        """`obs` (one row per environment) through the `envs` stage of every key, as a new dictionary."""
        obs = dict(obs)
        for k, pipeline in self._pipelines().items():
            if not pipeline.is_identity("envs"):
                obs[k] = pipeline.envs(obs[k], update=update)
        return obs

    def one(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """The observation of one environment, without the dimension of the environments, through the `envs` stage,
        frozen."""
        return {k: v[0] for k, v in self.envs({k: v[np.newaxis] for k, v in obs.items()}, update=False).items()}

    def sync(self, fabric: Fabric) -> None:
        for pipeline in self.pipelines:
            pipeline.sync(fabric)

    def key_state_dict(self) -> Dict[str, Dict[str, Any]]:
        """The state of the transforms of every key that has one (e.g. the statistics of `Standardize`)."""
        states = {k: p.state_dict() for k, p in self._pipelines().items()}
        return {k: state for k, state in states.items() if state}

    def load_key_state_dict(self, state: Optional[Dict[str, Dict[str, Any]]]) -> None:
        """Restore what `key_state_dict` saved: every key with a state must find its own."""
        state = state or {}
        for k, pipeline in self._pipelines().items():
            if pipeline.state_dict():
                if k not in state:
                    raise KeyError(f"The state of the transforms of the key `{k}` is not in the checkpoint")
                pipeline.load_state_dict(state[k])
