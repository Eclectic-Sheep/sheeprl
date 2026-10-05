"""Where the players write the collected data and the algorithms read their training data from.

Every algorithm writes its steps in a `ReplayStore`: a `ReplayBuffer` read by a sampler of `sheeprl.data.samplers`. The
on-policy algorithms in the one of their rollout (`rollout_store`), whose `EpochSampler` draws the minibatches of an
update; the off-policy algorithms in one that the training loop saves in the checkpoints when `buffer.checkpoint` is
set.
"""

from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import EpochSampler
from sheeprl.data.store import ReplayStore
from sheeprl.utils import fs


def rollout_store(fabric: Fabric, cfg: Dict[str, Any], log_dir: str, size: int) -> ReplayStore:
    """The store of an on-policy algorithm: a rollout of `size` steps of every environment of the process, whose
    minibatches have `algo.per_rank_batch_size` steps (from the rollouts of all the processes with
    `buffer.share_data`)."""
    storage = ReplayBuffer(
        size,
        cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=fs.memmap_dir(cfg, log_dir, fabric.global_rank),
        obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
    )
    sampler = EpochSampler(
        cfg.algo.per_rank_batch_size,
        fabric.world_size,
        fabric.global_rank,
        seed=cfg.seed,
        distributed=cfg.buffer.get("share_data", False),
    )
    return ReplayStore(storage, sampler, fabric.device, cfg.buffer.from_numpy)


def load_replay_buffer(fabric: Fabric, saved: Any, store: ReplayStore) -> ReplayStore:
    """The replay buffer of this process, from the one saved in a checkpoint (`saved`) in place of `store`
    (`ReplayStore.load`).

    A checkpoint holds the list of the stores of all the processes, or a single store, which every process then starts
    from.
    """
    if isinstance(saved, list):
        if len(saved) != fabric.world_size:
            raise RuntimeError(
                f"The checkpoint holds {len(saved)} replay buffer(s), "
                f"but {fabric.world_size} processes are instantiated"
            )
        saved = saved[fabric.global_rank]
    return store.load(saved)
