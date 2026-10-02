"""DroQ must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/droq.json` was recorded with the DroQ training loop before the port (`main` at `118d6fb`), on CPU in fp32:
for each configuration below it holds every step written in the replay buffer, the loss of every critic update, the
actor and entropy losses of every iteration and checksums of the final checkpoint (weights, optimizer states, replay
ratio, counters). The test runs the same configurations and compares.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct), from the root of the repository:
    python -m tests.test_core.test_droq_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_droq_equivalence
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Any, Dict, List
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sheeprl import ROOT_DIR
from tests.test_core.test_sac_equivalence import checksum, max_difference

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "droq.json")

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "exp=droq",
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
    "algo.hidden_size=32",
    "seed=7",
]

CONFIGS: Dict[str, List[str]] = {
    # The DroQ recipe: random actions for 8 iterations, then 40 critic updates per iteration (2 envs, ratio 20) and
    # one actor update; episodes truncated by the time limit every 10 steps
    "default": [
        "env.num_envs=2",
        "env.max_episode_steps=10",
        "algo.total_steps=32",
        "algo.learning_starts=16",
        "algo.per_rank_batch_size=8",
        "buffer.size=1000",
    ],
    # One env: a fractional replay ratio (0 or 1 critic updates per iteration) and a burst of pretraining steps
    "fractional_ratio_pretrain": [
        "env.num_envs=1",
        "env.max_episode_steps=7",
        "algo.total_steps=40",
        "algo.learning_starts=10",
        "algo.replay_ratio=0.5",
        "algo.per_rank_pretrain_steps=4",
        "algo.per_rank_batch_size=4",
        "buffer.size=1000",
        "buffer.memmap=False",
    ],
    # Next observations sampled from the buffer, 3 critics without dropout, 2 observation keys
    "next_obs_sampled_three_critics": [
        "env.num_envs=2",
        "env.max_episode_steps=9",
        "env.reward_as_observation=True",
        "algo.mlp_keys.encoder=[state,reward]",
        "algo.total_steps=32",
        "algo.learning_starts=8",
        "algo.replay_ratio=2",
        "algo.critic.n=3",
        "algo.critic.dropout=0",
        "algo.per_rank_batch_size=4",
        "buffer.size=1000",
        "buffer.sample_next_obs=True",
    ],
    # One iteration, a buffer of one row, no warm-up
    "dry_run": [
        "env.num_envs=2",
        "dry_run=True",
        "algo.per_rank_batch_size=4",
        "buffer.size=1000",
    ],
}


def record_droq(name: str) -> Dict[str, Any]:
    """Train DroQ with the configuration `name` and record what it writes in the buffer, its losses and its final
    checkpoint."""
    from sheeprl.algos.droq import droq
    from sheeprl.cli import run
    from sheeprl.data.buffers import ReplayBuffer

    record: Dict[str, Any] = {"steps": [], "critic_losses": [], "actor_losses": []}
    buffer_add = ReplayBuffer.add
    mse_loss, policy_loss, entropy_loss = F.mse_loss, droq.policy_loss, droq.entropy_loss
    losses: Dict[str, float] = {}

    def recording_add(self, data, validate_args=False):
        record["steps"].append({k: np.asarray(v, dtype=np.float64).tolist() for k, v in sorted(data.items())})
        return buffer_add(self, data, validate_args=validate_args)

    def recording_mse_loss(*args, **kwargs):
        # The loss of one critic update
        loss = mse_loss(*args, **kwargs)
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

    root_dir = os.path.join("pytest_equivalence", f"droq_{name}")
    args = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *CONFIGS[name], f"root_dir={root_dir}", "run_name=run"]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", args),
        mock.patch.object(ReplayBuffer, "add", recording_add),
        mock.patch.object(F, "mse_loss", recording_mse_loss),
        mock.patch.object(droq, "policy_loss", recording_policy_loss),
        mock.patch.object(droq, "entropy_loss", recording_entropy_loss),
    ):
        run()

    log_dir = os.path.join("logs", "runs", root_dir)
    (ckpt_path,) = glob.glob(os.path.join(log_dir, "run", "version_*", "checkpoint", "*.ckpt"))
    state = torch.load(ckpt_path, weights_only=False)
    record["checkpoint"] = {
        "agent": checksum(state["agent"]),
        "qf_optimizer": checksum(state["qf_optimizer"]["state"]),
        "actor_optimizer": checksum(state["actor_optimizer"]["state"]),
        "alpha_optimizer": checksum(state["alpha_optimizer"]["state"]),
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
def test_droq_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_droq(name)
    # Same structure (keys, number of steps and of updates), counters and flags
    max_difference(record["steps"], reference["steps"])
    assert len(record["critic_losses"]) == len(reference["critic_losses"])
    assert len(record["actor_losses"]) == len(reference["actor_losses"])
    for key in ("ratio", "iter_num", "batch_size", "buffer_saved"):
        assert record["checkpoint"][key] == reference["checkpoint"][key], key
    torch.testing.assert_close(record["critic_losses"], reference["critic_losses"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["actor_losses"], reference["actor_losses"], rtol=1e-4, atol=1e-5)
    for step, reference_step in zip(record["steps"], reference["steps"]):
        torch.testing.assert_close(step, reference_step, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["checkpoint"], reference["checkpoint"], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record the reference instead of comparing with it")
    parser.add_argument("--reference", default=REFERENCE_PATH, help="the reference file to write or compare with")
    cli_args = parser.parse_args()
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    records = {name: record_droq(name) for name in CONFIGS}
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
