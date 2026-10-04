"""Where the players write the collected data and the algorithms read their training data from.

Every algorithm writes its steps in a `ReplayStore`: a `ReplayBuffer` read by a sampler of `sheeprl.data.samplers`. The
on-policy algorithms in a `Rollout`, whose `EpochSampler` draws the minibatches of an update; the off-policy algorithms
in a store that the training loop saves in the checkpoints when `buffer.checkpoint` is set.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, TypeVar

import numpy as np
import torch
from lightning import Fabric
from torch import Tensor

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import EpochSampler
from sheeprl.data.store import ReplayStore

Store = TypeVar("Store")


class Rollout(ReplayStore):
    """The store of the on-policy algorithms: the steps of the current rollout, in a `ReplayBuffer` of
    `algo.rollout_steps` steps of every environment, the observations that follow its last step, whose value bootstraps
    the returns, and the `EpochSampler` of the minibatches of its update.

    The training reads the whole rollout (`read`), computes from it what it trains on (e.g. the advantages), and draws
    the minibatches of its epochs from that (`minibatches`)."""

    def __init__(
        self, storage: ReplayBuffer, sampler: EpochSampler, device: str | torch.device = "cpu", from_numpy: bool = False
    ):
        super().__init__(storage, sampler, device, from_numpy)
        self.next_obs: Dict[str, np.ndarray] = {}

    @classmethod
    def build(cls, fabric: Fabric, cfg: Dict[str, Any], log_dir: str, size: int) -> "Rollout":
        """The rollout of `size` steps of every environment of the process, whose minibatches have
        `algo.per_rank_batch_size` steps (from the rollouts of all the processes with `buffer.share_data`)."""
        storage = ReplayBuffer(
            size,
            cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
        )
        sampler = EpochSampler(
            cfg.algo.per_rank_batch_size,
            fabric.world_size,
            fabric.global_rank,
            seed=cfg.seed,
            distributed=cfg.buffer.get("share_data", False),
        )
        return cls(storage, sampler, fabric.device, cfg.buffer.from_numpy)

    @property
    def buffer(self) -> ReplayBuffer:
        """The storage of the steps of the rollout."""
        return self.storage

    def add(self, data: Dict[str, Any], next_obs: Dict[str, np.ndarray], validate_args: bool = False) -> None:
        """Write one step (`data`, with shape `[1, num_envs, ...]`) and remember the observations that follow it."""
        super().add(data, validate_args=validate_args)
        self.next_obs = next_obs

    def read(self) -> Dict[str, Tensor]:
        """The steps of the rollout, `[Rollout_Steps, Num_Envs, ...]`, on the device, in the dtypes of the storage."""
        return self.storage.to_tensor(dtype=None, device=self.device, from_numpy=self.from_numpy)

    def minibatches(self, data: Dict[str, Tensor], epochs: int, dim: int = 0) -> Iterator[Dict[str, Tensor]]:
        """The minibatches of `epochs` epochs over the elements of `data`, along the dimension `dim` (e.g. the steps
        of the flattened rollout, or its sequences), drawn by the `EpochSampler`."""
        n = next(iter(data.values())).shape[dim]
        for idxes, _ in self.sampler.epochs(n, epochs):
            index = torch.as_tensor(idxes, device=next(iter(data.values())).device)
            yield {k: v.index_select(dim, index) for k, v in data.items()}


def load_replay_buffer(fabric: Fabric, saved: Any, store: Store) -> Store:
    """The replay buffer of this process, from the one saved in a checkpoint (`saved`) in place of `store`.

    A checkpoint holds the list of the buffers of all the processes, or a single buffer, which every process then
    starts from. A `ReplayStore` takes the storage and the sampler of the saved one (`ReplayStore.load`), or the saved
    storage of a checkpoint of sheeprl up to 0.8.2.
    """
    if isinstance(saved, list):
        if len(saved) != fabric.world_size:
            raise RuntimeError(
                f"The checkpoint holds {len(saved)} replay buffer(s), "
                f"but {fabric.world_size} processes are instantiated"
            )
        saved = saved[fabric.global_rank]
    if isinstance(store, ReplayStore):
        return store.load(saved)
    if isinstance(saved, type(store)):
        return saved
    raise RuntimeError(
        f"The checkpoint holds a replay buffer of type {type(saved).__name__}, not {type(store).__name__}"
    )
