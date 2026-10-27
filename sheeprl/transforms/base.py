"""The transforms of the observation keys and their pipelines.

A transform is one step of what happens to the data of a key between the environment and the network. It runs at one
of three placements, named after the unit of data it sees:

- `frame`: in the environment process, once per step of one environment, on one observation (NumPy): the steps that
  change the shape or lose information, done once and stored compactly (resize, frame stack, flatten, one-hot);
- `envs`: in the main process, inside `GymEnvironment.step`, on the observations (or the rewards) of all the
  environments of the process, one row each (NumPy): the steps with running statistics, which are synced across the
  processes and saved in the checkpoints;
- `batch`: on the device, inside `Input.forward` (so inside the compiled step), on a tensor of any leading shape: the
  cheap elementwise steps that must not inflate the storage (to float, divide by 255, symlog) and the learned ones.

A pipeline holds the transforms of one key, in the order of the placements: the space after its `frame` stage is the
one the environments report, after its `envs` stage the one the store holds, after its `batch` stage the one the
network is sized from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import hydra
import numpy as np
from torch import Tensor, nn

if TYPE_CHECKING:
    from lightning import Fabric

PLACEMENTS = ("frame", "envs", "batch")
# `act`: the policy choosing the actions in the environments (and the test); `train`: a training batch
PHASES = ("act", "train")


class Transform(nn.Module):
    """One step of the pipeline of an observation key (or of the rewards). `where` is the placement it runs at,
    `supports` the ones it implements (the configuration picks another one among them with `where: ...`). A `batch`
    transform runs in the phases `phases` (e.g. a data augmentation only in `train`).

    A transform is an `nn.Module` whatever its placement: its parameters and buffers are its state (e.g. the
    statistics of an `envs` transform, the weights of a learned `batch` one), saved with `state_dict`. `forward` is not
    used: `frame`, `envs` and `batch` are named for the data they take.
    """

    where: str = "batch"
    supports: Tuple[str, ...] = ()
    phases: Tuple[str, ...] = PHASES
    # Whether `frame` reads the `info` of the step
    needs_info: bool = False
    # Whether the transform learns statistics from the data it sees (e.g. a running mean): an error for the
    # off-policy algorithms, whose replay buffer would keep data transformed with statistics that change
    learns_from_data: bool = False

    def __init__(self, where: Optional[str] = None, phases: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        supports = self.supports or (type(self).where,)
        if where is not None:
            if where not in supports:
                raise ValueError(f"{type(self).__name__} runs at {' or '.join(supports)}, not at `{where}`")
            self.where = where
        if phases is not None:
            unknown = set(phases) - set(PHASES)
            if unknown or not phases:
                raise ValueError(f"The phases of {type(self).__name__} must be among {PHASES}, got {list(phases)}")
            self.phases = tuple(phases)

    def space(self, space: gym.Space) -> gym.Space:
        """The space after the transform (shape, dtype and bounds), called once per key when the pipeline is built:
        the transform may keep what its other methods need (e.g. the `n` of a `Discrete`)."""
        return space

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """The observation `x` of one environment, transformed. `info` is the one of the step, with its `terminated`
        and `truncated` (both false after a reset), for the transforms that read it (`needs_info`)."""
        raise NotImplementedError(f"{type(self).__name__} doesn't run at `frame`")

    def envs(self, x: np.ndarray, update: bool = True, dones: Optional[np.ndarray] = None) -> np.ndarray:
        """The observations (or rewards) `x` of the environments, one row each, transformed. With `update`, they are
        added to the statistics of the transform (not for the last observations of the ended episodes, nor in the
        test). `dones` marks the environments whose episode ended in the step."""
        raise NotImplementedError(f"{type(self).__name__} doesn't run at `envs`")

    def batch(self, x: Tensor) -> Tensor:
        """The tensor `x`, of any leading shape, transformed: tensor operations only, to be compiled."""
        raise NotImplementedError(f"{type(self).__name__} doesn't run at `batch`")

    def inverse(self, x: Tensor) -> Tensor:
        """The opposite of `batch`: back to the domain of the store (e.g. the reconstructions to log)."""
        raise NotImplementedError(f"{type(self).__name__} has no inverse")

    def reset(self) -> None:
        """Forget the state of the episode (`frame` transforms, e.g. the frames of `FrameStack`)."""

    def seed(self, seed: int) -> None:
        """Seed the randomness of the transform (random `frame` transforms)."""

    def sync(self, fabric: Fabric) -> None:
        """Merge the statistics of all the processes (`envs` transforms)."""

    def extra_repr(self) -> str:
        return f"where={self.where}"


class Pipeline(nn.Module):
    """The transforms of one key, in the order of the placements (`frame`, then `envs`, then `batch`: validated).

    The pipeline holds the stages from `from_stage` to `to_stage`, and `space` is the space of the data before
    `from_stage`: e.g. the environments hold the `frame` stage, `GymEnvironment` the `envs` one, and an input builds the
    `batch` one from the space of the store. `frame_space`, `envs_space` and `batch_space` are the spaces after every
    stage (for a stage it doesn't hold, the space after the last one before it).
    """

    def __init__(
        self, transforms: Sequence[Transform], space: gym.Space, from_stage: str = "frame", to_stage: str = "batch"
    ) -> None:
        super().__init__()
        if from_stage not in PLACEMENTS or to_stage not in PLACEMENTS:
            raise ValueError(
                f"`from_stage` and `to_stage` must be among {PLACEMENTS}, got `{from_stage}`, `{to_stage}`"
            )
        order = [PLACEMENTS.index(t.where) for t in transforms]
        if order != sorted(order):
            raise ValueError(
                "The transforms of a pipeline run in the order of their placements (frame, then envs, then batch), "
                f"got: {', '.join(f'{type(t).__name__} ({t.where})' for t in transforms)}"
            )
        first, last = PLACEMENTS.index(from_stage), PLACEMENTS.index(to_stage)
        self.stages = nn.ModuleDict(
            {p: nn.ModuleList([t for t in transforms if t.where == p]) for p in PLACEMENTS[first : last + 1]}
        )
        self.spaces: Dict[str, gym.Space] = {}
        for placement in PLACEMENTS:
            for transform in self.stages[placement] if placement in self.stages else ():
                space = transform.space(space)
            self.spaces[placement] = space

    @classmethod
    def from_config(
        cls, transforms_cfg: Sequence[Any], space: gym.Space, from_stage: str = "frame", to_stage: str = "batch"
    ) -> Pipeline:
        """The pipeline of the transforms configured as Hydra objects (`_target_` and arguments)."""
        return cls(instantiate_transforms(transforms_cfg), space, from_stage, to_stage)

    @property
    def frame_space(self) -> gym.Space:
        return self.spaces["frame"]

    @property
    def envs_space(self) -> gym.Space:
        return self.spaces["envs"]

    @property
    def batch_space(self) -> gym.Space:
        return self.spaces["batch"]

    def transforms(self, placement: str) -> Sequence[Transform]:
        """The transforms of the stage `placement` (none if the pipeline doesn't hold it)."""
        return self.stages[placement] if placement in self.stages else ()

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        for transform in self.transforms("frame"):
            x = transform.frame(x, info) if transform.needs_info else transform.frame(x)
        return x

    def envs(self, x: np.ndarray, update: bool = True, dones: Optional[np.ndarray] = None) -> np.ndarray:
        for transform in self.transforms("envs"):
            x = transform.envs(x, update=update, dones=dones)
        return x

    def batch(self, x: Tensor, phase: str = "train") -> Tensor:
        """`x` through the `batch` transforms that run in `phase`."""
        for transform in self.transforms("batch"):
            if phase in transform.phases:
                x = transform.batch(x)
        return x

    def inverse(self, x: Tensor) -> Tensor:
        """`x`, in the domain of the training batches, back to the domain of the store."""
        for transform in reversed(list(self.transforms("batch"))):
            if "train" in transform.phases:
                x = transform.inverse(x)
        return x

    def reset(self) -> None:
        for transform in self.transforms("frame"):
            transform.reset()

    def seed(self, seed: int) -> None:
        for transform in self.transforms("frame"):
            transform.seed(seed)

    def sync(self, fabric: Fabric) -> None:
        for transform in self.transforms("envs"):
            transform.sync(fabric)

    @property
    def needs_info(self) -> bool:
        return any(t.needs_info for t in self.transforms("frame"))

    @property
    def learns_from_data(self) -> bool:
        return any(t.learns_from_data for placement in PLACEMENTS for t in self.transforms(placement))

    def is_identity(self, placement: str) -> bool:
        """Whether the stage `placement` has no transform."""
        return len(self.transforms(placement)) == 0


def instantiate_transforms(transforms_cfg: Sequence[Any]) -> List[Transform]:
    """The transforms configured as Hydra objects (`_target_` and arguments)."""
    transforms = [hydra.utils.instantiate(cfg, _convert_="all") for cfg in transforms_cfg or ()]
    for transform in transforms:
        if not isinstance(transform, Transform):
            raise TypeError(f"A transform must be a `sheeprl.transforms.Transform`, got {type(transform)}")
    return transforms
