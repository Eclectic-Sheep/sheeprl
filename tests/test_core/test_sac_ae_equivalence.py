"""SAC-AE must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/sac_ae.json` was recorded with the SAC-AE training loop before the port (`main` at `118d6fb`), on CPU in
fp32: for each configuration below it holds every step written in the replay buffer, the losses of every gradient step
(critic, actor and entropy coefficient when updated, every term of the reconstruction when updated) and checksums of
the final checkpoint (weights, optimizer states, replay ratio, counters). The test runs the same configurations and
compares.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct), from the root of the repository:
    python -m tests.test_core.test_sac_ae_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_sac_ae_equivalence
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import types
from typing import Any, Dict, List
from unittest import mock

import numpy as np
import pytest
import torch

from sheeprl import ROOT_DIR
from tests.test_core.test_sac_equivalence import checksum, max_difference

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "sac_ae.json")

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "exp=sac_ae",
    "env.id=Pendulum-v1",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.accelerator=cpu",
    "fabric.devices=1",
    "fabric.precision=32-true",
    "metric.log_level=0",
    "metric.disable_timer=True",
    "checkpoint.every=0",
    "checkpoint.save_last=True",
    "algo.run_test=False",
    "algo.hidden_size=16",
    "algo.dense_units=16",
    "algo.cnn_channels_multiplier=1",
    "algo.encoder.features_dim=16",
    "seed=7",
]

CONFIGS: Dict[str, List[str]] = {
    # Pendulum rendered in 3 stacked frames and its state, both encoded and decoded; the default update periods:
    # targets and actor every 2 gradient steps, decoder at every one
    "pixels_and_state": [
        "env.num_envs=2",
        "env.max_episode_steps=10",
        "env.frame_stack=3",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.total_steps=24",
        "algo.learning_starts=8",
        "algo.per_rank_batch_size=4",
        "buffer.size=100",
    ],
    # Pixels only, one env: other update periods, a fractional replay ratio and a burst of pretraining steps
    "pixels_only_update_periods": [
        "env.num_envs=1",
        "env.max_episode_steps=7",
        "env.frame_stack=1",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.mlp_keys.encoder=[]",
        "algo.mlp_keys.decoder=[]",
        "algo.total_steps=20",
        "algo.learning_starts=6",
        "algo.replay_ratio=0.5",
        "algo.per_rank_pretrain_steps=3",
        "algo.actor.per_rank_update_freq=1",
        "algo.decoder.per_rank_update_freq=3",
        "algo.critic.per_rank_target_network_update_freq=3",
        "algo.per_rank_batch_size=4",
        "buffer.size=100",
        "buffer.memmap=False",
    ],
    # Next observations sampled from the buffer, the state only decoded, 3 critics, replay ratio 2
    "next_obs_sampled": [
        "env.num_envs=2",
        "env.max_episode_steps=9",
        "env.frame_stack=1",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[]",
        "algo.mlp_keys.encoder=[state]",
        "algo.total_steps=16",
        "algo.learning_starts=6",
        "algo.replay_ratio=2",
        "algo.critic.n=3",
        "algo.per_rank_batch_size=4",
        "buffer.size=100",
        "buffer.sample_next_obs=True",
    ],
    # One iteration, a buffer of one row, no warm-up
    "dry_run": [
        "env.num_envs=2",
        "env.frame_stack=1",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "dry_run=True",
        "algo.per_rank_batch_size=4",
        "buffer.size=100",
    ],
}


def record_sac_ae(name: str) -> Dict[str, Any]:
    """Train SAC-AE with the configuration `name` and record what it writes in the buffer, its losses and its final
    checkpoint."""
    from sheeprl.algos.sac_ae import sac_ae
    from sheeprl.cli import run
    from sheeprl.data.buffers import ReplayBuffer

    record: Dict[str, Any] = {"steps": [], "critic_losses": [], "actor_losses": [], "reconstruction_losses": []}
    buffer_add = ReplayBuffer.add
    critic_loss, policy_loss, entropy_loss = sac_ae.critic_loss, sac_ae.policy_loss, sac_ae.entropy_loss
    losses: Dict[str, float] = {}

    def recording_add(self, data, validate_args=False):
        record["steps"].append({k: np.asarray(v, dtype=np.float64).tolist() for k, v in sorted(data.items())})
        return buffer_add(self, data, validate_args=validate_args)

    def recording_critic_loss(*args, **kwargs):
        loss = critic_loss(*args, **kwargs)
        record["critic_losses"].append(loss.item())
        return loss

    def recording_policy_loss(*args, **kwargs):
        losses["policy"] = policy_loss(*args, **kwargs)
        return losses["policy"]

    def recording_entropy_loss(*args, **kwargs):
        # The entropy loss follows the actor loss: the update of the actor is complete
        loss = entropy_loss(*args, **kwargs)
        record["actor_losses"].append([losses["policy"].item(), loss.item()])
        return loss

    def recording_mse_loss(*args, **kwargs):
        # The reconstruction error of one decoded key (the only use of `F` in the module of SAC-AE)
        loss = torch.nn.functional.mse_loss(*args, **kwargs)
        record["reconstruction_losses"].append(loss.item())
        return loss

    root_dir = os.path.join("pytest_equivalence", f"sac_ae_{name}")
    args = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *CONFIGS[name], f"root_dir={root_dir}", "run_name=run"]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", args),
        mock.patch.object(ReplayBuffer, "add", recording_add),
        mock.patch.object(sac_ae, "critic_loss", recording_critic_loss),
        mock.patch.object(sac_ae, "policy_loss", recording_policy_loss),
        mock.patch.object(sac_ae, "entropy_loss", recording_entropy_loss),
        mock.patch.object(sac_ae, "F", types.SimpleNamespace(mse_loss=recording_mse_loss)),
    ):
        try:
            run()
        finally:
            # The old training loop ran with a `DDPStrategy` even on one device: its process group must not leak into
            # the next runs
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            for key in ("LOCAL_RANK", "NODE_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                os.environ.pop(key, None)

    log_dir = os.path.join("logs", "runs", root_dir)
    (ckpt_path,) = glob.glob(os.path.join(log_dir, "run", "version_*", "checkpoint", "*.ckpt"))
    state = torch.load(ckpt_path, weights_only=False)
    record["checkpoint"] = {
        "agent": checksum(state["agent"]),
        "encoder": checksum(state["encoder"]),
        "decoder": checksum(state["decoder"]),
        "qf_optimizer": checksum(state["qf_optimizer"]["state"]),
        "actor_optimizer": checksum(state["actor_optimizer"]["state"]),
        "alpha_optimizer": checksum(state["alpha_optimizer"]["state"]),
        "encoder_optimizer": checksum(state["encoder_optimizer"]["state"]),
        "decoder_optimizer": checksum(state["decoder_optimizer"]["state"]),
        "ratio": state["ratio"],
        "iter_num": state["iter_num"],
        "batch_size": state["batch_size"],
        "buffer_saved": "rb" in state,
    }
    del state
    shutil.rmtree(log_dir, ignore_errors=True)
    if len(os.listdir(os.path.dirname(log_dir))) == 0:
        os.rmdir(os.path.dirname(log_dir))
    return record


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_sac_ae_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_sac_ae(name)
    # Same structure (keys, number of steps and of updates), counters and flags
    max_difference(record["steps"], reference["steps"])
    for key in ("critic_losses", "actor_losses", "reconstruction_losses"):
        assert len(record[key]) == len(reference[key]), key
    for key in ("ratio", "iter_num", "batch_size", "buffer_saved"):
        assert record["checkpoint"][key] == reference["checkpoint"][key], key
    for key in ("critic_losses", "actor_losses", "reconstruction_losses"):
        torch.testing.assert_close(record[key], reference[key], rtol=1e-4, atol=1e-5)
    for step, reference_step in zip(record["steps"], reference["steps"]):
        torch.testing.assert_close(step, reference_step, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["checkpoint"], reference["checkpoint"], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record the reference instead of comparing with it")
    parser.add_argument("--reference", default=REFERENCE_PATH, help="the reference file to write or compare with")
    cli_args = parser.parse_args()
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    records = {name: record_sac_ae(name) for name in CONFIGS}
    if cli_args.record:
        os.makedirs(os.path.dirname(os.path.abspath(cli_args.reference)), exist_ok=True)
        with open(cli_args.reference, "w") as f:
            json.dump(records, f)
        print(f"Reference written to {cli_args.reference}")
    else:
        with open(cli_args.reference) as f:
            references = json.load(f)
        for name, record in records.items():
            print(f"{name}: largest difference from the reference = {max_difference(record, references[name])}")
