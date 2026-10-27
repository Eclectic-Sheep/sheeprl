"""Running statistics of a stream of batches, synced across the processes: the state of the `envs` transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple, cast

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from lightning import Fabric


@dataclass(frozen=True)
class SampleMoments:
    """The mean, the variance and the number of a set of samples."""

    mean: np.ndarray
    var: np.ndarray
    count: float

    def merge(self, other: SampleMoments) -> SampleMoments:
        """The moments of the union of the two sets of samples
        (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm)."""
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / count
        m2 = self.var * self.count + other.var * other.count + delta**2 * self.count * other.count / count
        return SampleMoments(mean, m2 / count, count)


class RunningMeanStd(nn.Module):
    """The running mean and variance of a stream of batches of arrays of shape `shape`, merged batch by batch.

    The statistics start from a mean of 0 and a variance of 1 with a tiny count (`epsilon`): the first batch replaces
    them almost entirely. They are float64 CPU buffers (`mean`, `var`, `count`, saved with `state_dict`); the
    arithmetic is NumPy float64 on views of them.

    With several processes, every process adds its own batches, and `sync` merges the ones added by all of them since
    the last sync: the statistics are then the same in every process. The ones at the last sync and the ones added
    since are buffers that are not saved: loading a state makes it the synced one.
    """

    # The saved buffers, registered in `__init__`
    mean: Tensor
    var: Tensor
    count: Tensor

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        super().__init__()
        f64 = torch.float64
        self.register_buffer("mean", torch.zeros(shape, dtype=f64))
        self.register_buffer("var", torch.ones(shape, dtype=f64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=f64))
        self.register_buffer("synced_mean", torch.zeros(shape, dtype=f64), persistent=False)
        self.register_buffer("synced_var", torch.ones(shape, dtype=f64), persistent=False)
        self.register_buffer("synced_count", torch.tensor(epsilon, dtype=f64), persistent=False)
        self.register_buffer("added_mean", torch.zeros(shape, dtype=f64), persistent=False)
        self.register_buffer("added_var", torch.zeros(shape, dtype=f64), persistent=False)
        self.register_buffer("added_count", torch.tensor(0.0, dtype=f64), persistent=False)

    def _get(self, prefix: str) -> SampleMoments:
        return SampleMoments(
            getattr(self, f"{prefix}mean").numpy(),
            getattr(self, f"{prefix}var").numpy(),
            float(getattr(self, f"{prefix}count")),
        )

    def _set(self, prefix: str, moments: SampleMoments) -> None:
        getattr(self, f"{prefix}mean").numpy()[...] = moments.mean
        getattr(self, f"{prefix}var").numpy()[...] = moments.var
        getattr(self, f"{prefix}count").numpy()[...] = moments.count

    @property
    def moments(self) -> SampleMoments:
        return self._get("")

    def update(self, x: np.ndarray) -> None:
        """Add the batch `x`, one row per sample."""
        x = np.asarray(x, dtype=np.float64)
        batch = SampleMoments(x.mean(axis=0), x.var(axis=0), x.shape[0])
        # Both merged before writing: `merge` reads the buffers through their views
        moments, added = self._get("").merge(batch), self._get("added_").merge(batch)
        self._set("", moments)
        self._set("added_", added)

    def sync(self, fabric: Fabric) -> None:
        """Merge the batches added by every process since the last sync, in the order of the processes (the same in
        every process, so the statistics are the same, bit for bit). With one process there is nothing to merge."""
        if fabric.world_size == 1:
            return
        added = self._get("added_")
        shape, size = added.mean.shape, added.mean.size
        flat = np.concatenate([added.mean.ravel(), added.var.ravel(), [added.count]])
        # One row per process
        gathered = cast(Tensor, fabric.all_gather(torch.as_tensor(flat, device=fabric.device))).cpu().numpy()
        moments = self._get("synced_")
        for row in gathered:
            mean, var, count = row[:size].reshape(shape), row[size:-1].reshape(shape), float(row[-1])
            moments = moments.merge(SampleMoments(mean, var, count))
        self._set("", moments)
        self._set("synced_", moments)
        self._set("added_", SampleMoments(np.zeros(shape), np.zeros(shape), 0))

    def _load_from_state_dict(self, state_dict: Any, prefix: str, *args: Any, **kwargs: Any) -> None:
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # The loaded statistics are the synced ones, with nothing added since
        self._set("synced_", self._get(""))
        self._set("added_", SampleMoments(np.zeros(self.mean.shape), np.zeros(self.mean.shape), 0))
