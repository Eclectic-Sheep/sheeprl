"""Where the writers write the collected data and the algorithms read their training data from.

Every algorithm writes its steps in a `ReplayStore`: a `ReplayBuffer` read by a sampler of `sheeprl.data.samplers`,
built here for the three ways the algorithms train on their data:

- `rollout_store`: the rollout of an on-policy algorithm (PPO, A2C, PPO recurrent), whose `EpochSampler` draws the
  minibatches of an update;
- `transition_store`: the replay buffer of an off-policy algorithm trained on single steps (SAC, DroQ, SAC-AE), read by
  a `TransitionSampler`;
- `sequence_store`: the replay buffer of an algorithm trained on sequences of one environment (the Dreamers), read by a
  `SequenceSampler` or an `EpisodeSampler`, with `env_buffer_size` steps per environment.

The training loop saves the replay buffers of the off-policy algorithms in the checkpoints when `buffer.checkpoint` is
set (`load_replay_buffer` restores them).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from lightning import Fabric

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import EpisodeSampler, EpochSampler, SequenceSampler, TransitionSampler
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


def transition_store(
    fabric: Fabric, cfg: Dict[str, Any], log_dir: str, obs_keys: Tuple[str, ...] = ("observations",)
) -> ReplayStore:
    """The store of an off-policy algorithm trained on single steps (SAC, DroQ, SAC-AE): a replay buffer of
    `buffer.size` steps split among the environments of all the processes (1 in a dry run), sampled one step at a time
    (`TransitionSampler`, with the next observations with `buffer.sample_next_obs` and the online queue with
    `buffer.online`). Every process trains on its own data, as the Dreamers do, and `update` averages the gradients over
    the processes."""
    storage = ReplayBuffer(
        cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 1,
        cfg.env.num_envs,
        obs_keys=obs_keys,
        memmap=cfg.buffer.memmap and not cfg.buffer.on_device,
        memmap_dir=fs.memmap_dir(cfg, log_dir, fabric.global_rank),
        device=fabric.device if cfg.buffer.on_device else None,
    )
    sampler = TransitionSampler(cfg.buffer.sample_next_obs, cfg.buffer.online, seed=cfg.seed + fabric.global_rank)
    return ReplayStore(storage, sampler, fabric.device, cfg.buffer.from_numpy, cfg.buffer.prefetch)


def env_buffer_size(fabric: Fabric, cfg: Dict[str, Any], dry_run_size: int) -> int:
    """The capacity of the replay buffer of every environment: `buffer.size` split among the environments of all the
    processes, or `dry_run_size` in a dry run. It must hold a sequence of `algo.per_rank_sequence_length` steps (a dry
    run makes it large enough)."""
    sequence_length = cfg.algo.per_rank_sequence_length
    if cfg.dry_run:
        return max(dry_run_size, sequence_length)
    size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size)
    if size < sequence_length:
        raise ValueError(
            f"The replay buffer of every environment holds `buffer.size // (env.num_envs * world_size)` = {size} "
            f"steps, fewer than a sequence (`algo.per_rank_sequence_length={sequence_length}`): increase `buffer.size`"
        )
    return size


def sequence_store(
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    buffer_size: int,
    sequence_length: int,
    buffer_type: str = "sequential",
) -> ReplayStore:
    """The store of an algorithm trained on sequences (the Dreamers): a buffer of `buffer_size` steps per environment
    (`env_buffer_size`), every environment written at its own row (its first steps after the end of an episode),
    sampled in sequences of `sequence_length` steps of a single environment: anywhere (`buffer_type="sequential"`,
    `SequenceSampler`) or inside the episodes that ended (`"episode"`, `EpisodeSampler`, with their ends prioritized
    with `buffer.prioritize_ends`), with the online queue with `buffer.online`. Every process samples its buffer with a
    generator of its own."""
    buffer_type = buffer_type.lower()
    if buffer_type not in ("sequential", "episode"):
        raise ValueError(f"Unrecognized buffer type: must be one of `sequential` or `episode`, received: {buffer_type}")
    storage = ReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
        memmap=cfg.buffer.memmap and not cfg.buffer.on_device,
        memmap_dir=fs.memmap_dir(cfg, log_dir, fabric.global_rank),
        device=fabric.device if cfg.buffer.on_device else None,
    )
    if buffer_type == "episode":
        sampler = EpisodeSampler(
            sequence_length,
            prioritize_ends=cfg.buffer.prioritize_ends,
            online=cfg.buffer.online,
            seed=cfg.seed + fabric.global_rank,
        )
    else:
        sampler = SequenceSampler(sequence_length, online=cfg.buffer.online, seed=cfg.seed + fabric.global_rank)
    return ReplayStore(storage, sampler, fabric.device, cfg.buffer.from_numpy, cfg.buffer.prefetch)


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
