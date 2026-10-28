# Add a transform

A transform is one step of what happens to the data of an observation key (or to the rewards) between the
environment and the network. This page explains the contract a transform follows, then writes three of them step by
step: a one-hot encoding, a data augmentation that runs only on the training batches, and the clipping of the rewards
to their sign. Read [inputs and outputs](./inputs_and_outputs.md) first for the vocabulary (placement, pipeline,
phase).

## The contract

A transform is a subclass of `sheeprl.transforms.Transform`, which is an `nn.Module` whatever its placement: its
parameters and buffers are its state, saved with `state_dict` (e.g. the statistics of an `envs` transform, the weights
of a learned `batch` one). Its class attributes:

- `where`: the placement it runs at by default, `frame`, `envs` or `batch`.
- `supports`: the placements it implements, if more than one; the configuration picks one with `where: ...`.
- `phases`: for a `batch` transform, the phases it runs in, `("act", "train")` by default. The configuration may set
  them with `phases: [...]`.
- `needs_info`: whether its `frame` method reads the `info` of the step.
- `learns_from_data`: whether it learns statistics from the data. Such a transform is an error for the off-policy
  algorithms, whose replay buffer would keep data transformed with statistics that change.

Its methods:

- `space(space) -> space`: the space after the transform (shape, dtype and bounds). It is called once per key when
  the pipeline is built, before any data, so it is where the transform keeps what its other methods need (e.g. the
  `n` of a `Discrete` space) and raises a clear error for a space it cannot take.
- One method per placement it supports, named for the data it takes:
  - `frame(x, info=None)`: the observation of one environment (NumPy);
  - `envs(x, update=True, dones=None)`: the observations of all the environments of the process, one row each
    (NumPy). With `update`, they are added to the statistics of the transform; `dones` marks the environments whose
    episode ended in the step;
  - `batch(x)`: a tensor of any leading shape, on the device. Tensor operations only (no `.item()`, no shapes that
    depend on the values): it is compiled with the network.
- Optional: `inverse(x)` (the opposite of `batch`, to bring reconstructions back to the stored domain), `reset()`
  (forget the state of the episode, `frame` transforms), `seed(seed)` (random `frame` transforms) and `sync(fabric)`
  (merge the statistics of all the processes, `envs` transforms).

`forward` is not used. A transform is configured by its import path, like any Hydra object, so it can live in your
own package.

`sheeprl.transforms.check_transform(transform, space, n=8, compile=False)` draws `n` samples from `space`, runs the
transform at its placement and asserts that its outputs have the shape, the dtype and the bounds its `space` method
promises. With `compile=True` it also asserts that `batch` compiles in one graph. Run it on every transform you write.

## Example 1: a one-hot encoding (`frame`)

The question: an environment returns a `Discrete(n)` observation (an integer), and the network wants a vector. The
one-hot vector has `n` values, all 0 but the one at the index of the integer. It changes the shape, so it runs at
`frame`, once per step in the process of the environment, and the store keeps the vector.

```python
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from sheeprl.transforms import Transform


class OneHot(Transform):
    """A `Discrete(n)` observation as a one-hot float32 vector of `n` values."""

    where = "frame"

    def space(self, space: gym.Space) -> gym.Space:
        if not isinstance(space, gym.spaces.Discrete):
            raise TypeError(f"OneHot takes a `Discrete` space, got {space}")
        # Kept for `frame`: the size of the vector and the first value of the space
        self.n, self.start = int(space.n), int(space.start)
        return gym.spaces.Box(0, 1, (self.n,), np.float32)

    def frame(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        one_hot = np.zeros(self.n, dtype=np.float32)
        one_hot[int(x) - self.start] = 1
        return one_hot
```

This is `sheeprl.transforms.OneHot`. Check it, then put it in the vector input:

```python
from sheeprl.transforms import check_transform

check_transform(OneHot(), gym.spaces.Discrete(4))
```

```yaml
algo:
  inputs:
    vector:
      keys: [level]
      transforms: [{_target_: sheeprl.transforms.OneHot}]
```

## Example 2: a data augmentation on the training batches only (`batch`)

The question: add Gaussian noise of standard deviation `std` to the vectors the network trains on, but not to the
ones the policy acts on. The noise must be drawn anew for every batch, so it runs at `batch`, on the device, and the
store keeps the clean observations. It runs only in the phase `train`.

```python
import torch
from torch import Tensor


class GaussianNoise(Transform):
    """`x` plus Gaussian noise of standard deviation `std`, on the training batches."""

    where = "batch"
    phases = ("train",)

    def __init__(self, std: float = 0.01, where=None, phases=None) -> None:
        super().__init__(where, phases)
        self.std = std

    def space(self, space: gym.Space) -> gym.Space:
        if not isinstance(space, gym.spaces.Box) or not np.issubdtype(space.dtype, np.floating):
            raise TypeError(f"GaussianNoise takes a floating-point `Box` space, got {space}")
        # The noise is unbounded: so are the values after it
        return gym.spaces.Box(-np.inf, np.inf, space.shape, space.dtype)

    def batch(self, x: Tensor) -> Tensor:
        return x + self.std * torch.randn_like(x)
```

Check that it compiles in one graph, then add it to the vector input after the other transforms (the order of the
placements is validated: `frame`, then `envs`, then `batch`). Your module must be importable by the run, e.g. in your
own package:

```python
check_transform(GaussianNoise(std=0.1), gym.spaces.Box(-1, 1, (4,), np.float32), compile=True)
```

```yaml
algo:
  inputs:
    vector:
      keys: [state]
      transforms: [{_target_: my_package.transforms.GaussianNoise, std: 0.1}]
```

The policy reads the vectors without the noise (phase `act`), the training step with it (phase `train`). The targets
of a decoder are computed in the phase `train`, so they would get the noise too: use `transforms` in the output to give
its targets their own transforms.

## Example 3: the rewards clipped to their sign (`envs`)

The question: train on the sign of the rewards, -1, 0 or 1, as DQN does on the Atari games. The rewards have their own
pipeline, `algo.rewards.transforms`, of `envs` transforms only: it runs in the main process on the rewards of all the
environments, before they are stored. The space of the rewards of one environment is
`Box(-inf, inf, (), float64)`.

```python
class Sign(Transform):
    """`sign(x)`: -1, 0 or 1."""

    where = "envs"
    supports = ("envs", "batch")

    def space(self, space: gym.Space) -> gym.Space:
        return gym.spaces.Box(-1.0, 1.0, space.shape, space.dtype)

    def envs(self, x: np.ndarray, update: bool = True, dones: Optional[np.ndarray] = None) -> np.ndarray:
        return np.sign(x)

    def batch(self, x: Tensor) -> Tensor:
        return torch.sign(x)
```

`sheeprl.transforms.Sign` is this transform (it shares its code with `Tanh` and `Clip`). Check it on the space of the
rewards and configure it:

```python
from sheeprl.transforms.rewards import REWARD_SPACE

check_transform(Sign(), REWARD_SPACE)
```

```yaml
algo:
  rewards:
    transforms: [{_target_: sheeprl.transforms.Sign}]
```

The rewards logged as the episode returns stay the ones of the environment: the transformed ones are what the
algorithm trains on.

## A transform with statistics

An `envs` transform with running statistics (e.g. `sheeprl.transforms.Standardize`) keeps them as buffers (the
`RunningMeanStd` module of `sheeprl.transforms.stats`), sets `learns_from_data = True`, adds the rows to the
statistics only when `update` is true (not for the last observations of the ended episodes, nor in the test) and
merges the statistics of the processes in `sync`. Its buffers are saved in `checkpoint["env"]` and restored on resume
and in the evaluation.

## Not a transform

A step that creates a key from other signals (the actions or the rewards as observations) or reads several keys at
once is a gymnasium wrapper, set in `env.wrapper`. What the algorithms do to the rewards and the returns on the device
(e.g. the two-hot reward model of the Dreamers) is part of their losses.
