"""SAC must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/sac.json` was recorded with the SAC training loop before the port (`main` at `118d6fb`), on CPU in fp32:
for each configuration below it holds every step written in the replay buffer (observations, actions, rewards, ...),
the three losses of every gradient step and checksums of the final checkpoint (weights, optimizer states, replay
ratio, counters). The test runs the same configurations and compares.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct), from the root of the repository:
    python -m tests.test_core.test_sac_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_sac_equivalence
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

from sheeprl import ROOT_DIR

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "sac.json")

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "exp=sac",
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
    # The default recipe: random actions for 8 iterations, then 2 gradient steps per iteration (2 envs, ratio 1);
    # episodes truncated by the time limit every 10 steps (the final observations are stored as next observations)
    "default": [
        "env.num_envs=2",
        "env.max_episode_steps=10",
        "algo.total_steps=64",
        "algo.learning_starts=16",
        "algo.per_rank_batch_size=8",
        "buffer.size=1000",
    ],
    # One env: the target networks are updated every other iteration (`target_network_frequency // 1 + 1`);
    # a fractional replay ratio (0 or 1 gradient steps per iteration) and a burst of pretraining steps
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
    # Targets updated every 3 iterations (4 // 2 + 1), next observations sampled from the buffer, 3 critics,
    # 2 observation keys, replay ratio 2
    "next_obs_sampled_three_critics": [
        "env.num_envs=2",
        "env.max_episode_steps=9",
        "env.reward_as_observation=True",
        "algo.mlp_keys.encoder=[state,reward]",
        "algo.total_steps=48",
        "algo.learning_starts=8",
        "algo.replay_ratio=2",
        "algo.critic.n=3",
        "algo.critic.target_network_frequency=4",
        "algo.per_rank_batch_size=4",
        "buffer.size=1000",
        "buffer.sample_next_obs=True",
    ],
    # One gradient step per iteration whatever the replay ratio (`exp=sac_benchmarks`)
    "run_benchmarks": [
        "env.num_envs=2",
        "algo.total_steps=32",
        "algo.learning_starts=8",
        "algo.replay_ratio=3",
        "algo.per_rank_batch_size=4",
        "buffer.size=1000",
        "+run_benchmarks=True",
    ],
    # One iteration, a buffer of one row, no warm-up
    "dry_run": [
        "env.num_envs=2",
        "dry_run=True",
        "algo.per_rank_batch_size=4",
        "buffer.size=1000",
    ],
}


def checksum(value: Any) -> Any:
    """Sum and sum of squares of every tensor in `value` (nested dicts and lists), the other values as they are."""
    if isinstance(value, torch.Tensor):
        value = value.detach().double()
        return [value.sum().item(), value.square().sum().item()]
    if isinstance(value, dict):
        return {str(k): checksum(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [checksum(v) for v in value]
    return value


def record_sac(name: str) -> Dict[str, Any]:
    """Train SAC with the configuration `name` and record what it writes in the buffer, its losses and its final
    checkpoint."""
    from sheeprl.algos.sac import sac
    from sheeprl.cli import run
    from sheeprl.data.buffers import ReplayBuffer

    record: Dict[str, Any] = {"steps": [], "losses": []}
    buffer_add = ReplayBuffer.add
    critic_loss, policy_loss, entropy_loss = sac.critic_loss, sac.policy_loss, sac.entropy_loss
    losses: Dict[str, float] = {}

    def recording_add(self, data, validate_args=False):
        record["steps"].append({k: np.asarray(v, dtype=np.float64).tolist() for k, v in sorted(data.items())})
        return buffer_add(self, data, validate_args=validate_args)

    def recording_critic_loss(*args, **kwargs):
        losses["critic"] = critic_loss(*args, **kwargs)
        return losses["critic"]

    def recording_policy_loss(*args, **kwargs):
        losses["policy"] = policy_loss(*args, **kwargs)
        return losses["policy"]

    def recording_entropy_loss(*args, **kwargs):
        # The entropy loss is the last of the three: the gradient step is complete
        loss = entropy_loss(*args, **kwargs)
        record["losses"].append([losses["critic"].item(), losses["policy"].item(), loss.item()])
        return loss

    root_dir = os.path.join("pytest_equivalence", f"sac_{name}")
    args = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *CONFIGS[name], f"root_dir={root_dir}", "run_name=run"]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", args),
        mock.patch.object(ReplayBuffer, "add", recording_add),
        mock.patch.object(sac, "critic_loss", recording_critic_loss),
        mock.patch.object(sac, "policy_loss", recording_policy_loss),
        mock.patch.object(sac, "entropy_loss", recording_entropy_loss),
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


def max_difference(record: Any, reference: Any) -> float:
    """Return the largest absolute difference between two records; raise if their structure differs."""
    if isinstance(reference, dict):
        assert isinstance(record, dict) and record.keys() == reference.keys(), f"{record.keys()} != {reference.keys()}"
        return max([max_difference(record[k], reference[k]) for k in reference] + [0.0])
    if isinstance(reference, list):
        assert isinstance(record, list) and len(record) == len(reference), f"{len(record)} != {len(reference)}"
        return max([max_difference(a, b) for a, b in zip(record, reference)] + [0.0])
    if isinstance(reference, (bool, str)) or reference is None:
        assert record == reference, f"{record} != {reference}"
        return 0.0
    return abs(float(record) - float(reference))


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_sac_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_sac(name)
    # Same structure (keys, number of steps and of gradient steps), counters and flags
    max_difference(record["steps"], reference["steps"])
    assert len(record["losses"]) == len(reference["losses"])
    for key in ("ratio", "iter_num", "batch_size", "buffer_saved"):
        assert record["checkpoint"][key] == reference["checkpoint"][key], key
    torch.testing.assert_close(record["losses"], reference["losses"], rtol=1e-4, atol=1e-5)
    for step, reference_step in zip(record["steps"], reference["steps"]):
        torch.testing.assert_close(step, reference_step, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["checkpoint"], reference["checkpoint"], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record the reference instead of comparing with it")
    parser.add_argument("--reference", default=REFERENCE_PATH, help="the reference file to write or compare with")
    cli_args = parser.parse_args()
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    records = {name: record_sac(name) for name in CONFIGS}
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
