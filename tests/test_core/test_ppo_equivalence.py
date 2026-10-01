"""PPO must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/ppo.json` was recorded with the PPO training loop before the port (`main` at `118d6fb`), on CPU in fp32:
for each configuration below it holds the actions, log-probs and values played at every environment step, the three
losses of every minibatch and a checksum of the final weights. The test runs the same configurations and compares.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct):
    python -m tests.test_core.test_ppo_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_ppo_equivalence
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

import pytest
import torch

from sheeprl import ROOT_DIR

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "ppo.json")

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "fabric.accelerator=cpu",
    "fabric.devices=1",
    "fabric.precision=32-true",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "metric.log_level=0",
    "metric.disable_timer=True",
    "checkpoint.every=0",
    "checkpoint.save_last=True",
    "algo.run_test=False",
    "seed=7",
]

CONFIGS: Dict[str, List[str]] = {
    # CartPole with the default PPO recipe: 3 iterations of 2 envs x 16 steps, last minibatch smaller (32 = 5 x 6 + 2)
    "cartpole_default": [
        "exp=ppo",
        "algo.rollout_steps=16",
        "algo.per_rank_batch_size=6",
        "algo.update_epochs=2",
        "algo.total_steps=96",
    ],
    # Every option of the PPO loss and of the schedules switched on
    "cartpole_all_options": [
        "exp=ppo",
        "algo.rollout_steps=16",
        "algo.per_rank_batch_size=6",
        "algo.update_epochs=2",
        "algo.total_steps=96",
        "algo.anneal_lr=True",
        "algo.anneal_clip_coef=True",
        "algo.anneal_ent_coef=True",
        "algo.ent_coef=0.01",
        "algo.normalize_advantages=True",
        "algo.clip_vloss=True",
        "algo.max_grad_norm=0.5",
        "env.clip_rewards=True",
    ],
    # Pixels and vectors, frame stacking, episodes truncated by the time limit (bootstrapped values)
    "discrete_pixels_truncated": [
        "exp=ppo",
        "env=dummy",
        "env.id=discrete_dummy",
        "env.max_episode_steps=3",
        "env.frame_stack=2",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.encoder.cnn_features_dim=16",
        "algo.rollout_steps=8",
        "algo.per_rank_batch_size=4",
        "algo.update_epochs=2",
        "algo.total_steps=32",
    ],
    "multidiscrete": [
        "exp=ppo",
        "env=dummy",
        "env.id=multidiscrete_dummy",
        "algo.cnn_keys.encoder=[]",
        "algo.mlp_keys.encoder=[state]",
        "algo.rollout_steps=8",
        "algo.per_rank_batch_size=4",
        "algo.update_epochs=2",
        "algo.total_steps=32",
    ],
    "continuous_normal": [
        "exp=ppo",
        "env=dummy",
        "env.id=continuous_dummy",
        "algo.cnn_keys.encoder=[]",
        "algo.mlp_keys.encoder=[state]",
        "algo.rollout_steps=8",
        "algo.per_rank_batch_size=4",
        "algo.update_epochs=2",
        "algo.total_steps=32",
    ],
    "continuous_tanh_normal": [
        "exp=ppo",
        "env=dummy",
        "env.id=continuous_dummy",
        "distribution.type=tanh_normal",
        "algo.cnn_keys.encoder=[]",
        "algo.mlp_keys.encoder=[state]",
        "algo.rollout_steps=8",
        "algo.per_rank_batch_size=4",
        "algo.update_epochs=2",
        "algo.total_steps=32",
    ],
}


def record_ppo(name: str) -> Dict[str, Any]:
    """Train PPO with the configuration `name` and record what it plays, its losses and its final weights."""
    from sheeprl.algos.ppo import agent, ppo
    from sheeprl.cli import run

    record: Dict[str, Any] = {"actions": [], "logprobs": [], "values": [], "losses": []}
    player_forward = agent.PPOPlayer.forward
    policy_loss, value_loss, entropy_loss = ppo.policy_loss, ppo.value_loss, ppo.entropy_loss
    losses: Dict[str, float] = {}

    def recording_player_forward(self, obs):
        actions, logprobs, values = player_forward(self, obs)
        record["actions"].append(torch.cat(actions, dim=-1).tolist())
        record["logprobs"].append(logprobs.tolist())
        record["values"].append(values.tolist())
        return actions, logprobs, values

    def recording_policy_loss(*args, **kwargs):
        losses["policy"] = policy_loss(*args, **kwargs)
        return losses["policy"]

    def recording_value_loss(*args, **kwargs):
        losses["value"] = value_loss(*args, **kwargs)
        return losses["value"]

    def recording_entropy_loss(*args, **kwargs):
        # The entropy loss is the last of the three: the minibatch is complete
        loss = entropy_loss(*args, **kwargs)
        record["losses"].append([losses["policy"].item(), losses["value"].item(), loss.item()])
        return loss

    root_dir = os.path.join("pytest_equivalence", name)
    args = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *CONFIGS[name], f"root_dir={root_dir}", "run_name=run"]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", args),
        mock.patch.object(agent.PPOPlayer, "forward", recording_player_forward),
        mock.patch.object(ppo, "policy_loss", recording_policy_loss),
        mock.patch.object(ppo, "value_loss", recording_value_loss),
        mock.patch.object(ppo, "entropy_loss", recording_entropy_loss),
    ):
        run()

    log_dir = os.path.join("logs", "runs", root_dir)
    (ckpt_path,) = glob.glob(os.path.join(log_dir, "run", "version_*", "checkpoint", "*.ckpt"))
    state = torch.load(ckpt_path, weights_only=False)
    shutil.rmtree(log_dir, ignore_errors=True)
    if len(os.listdir(os.path.dirname(log_dir))) == 0:
        os.rmdir(os.path.dirname(log_dir))
    record["weights"] = {
        k: [v.double().sum().item(), v.double().square().sum().item()] for k, v in sorted(state["agent"].items())
    }
    record["iter_num"] = state["iter_num"]
    return record


def max_difference(record: Dict[str, Any], reference: Dict[str, Any]) -> float:
    """Return the largest absolute difference between two records; raise if their structure differs."""
    assert record.keys() == reference.keys()
    assert record["iter_num"] == reference["iter_num"]
    assert record["weights"].keys() == reference["weights"].keys()
    largest = 0.0
    for key in ("actions", "logprobs", "values", "losses"):
        assert len(record[key]) == len(reference[key]), f"{key}: {len(record[key])} != {len(reference[key])}"
        if len(record[key]) > 0:
            diff = torch.tensor(record[key], dtype=torch.float64) - torch.tensor(reference[key], dtype=torch.float64)
            largest = max(largest, diff.abs().max().item())
    for k, v in record["weights"].items():
        diff = torch.tensor(v, dtype=torch.float64) - torch.tensor(reference["weights"][k], dtype=torch.float64)
        largest = max(largest, diff.abs().max().item())
    return largest


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_ppo_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_ppo(name)
    torch.testing.assert_close(record["losses"], reference["losses"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["actions"], reference["actions"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["logprobs"], reference["logprobs"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["values"], reference["values"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["weights"], reference["weights"], rtol=1e-4, atol=1e-5)
    assert record["iter_num"] == reference["iter_num"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record the reference instead of comparing with it")
    cli_args = parser.parse_args()
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    records = {name: record_ppo(name) for name in CONFIGS}
    if cli_args.record:
        os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
        with open(REFERENCE_PATH, "w") as f:
            json.dump(records, f)
        print(f"Reference written to {REFERENCE_PATH}")
    else:
        with open(REFERENCE_PATH) as f:
            references = json.load(f)
        for name, record in records.items():
            print(f"{name}: largest difference from the reference = {max_difference(record, references[name])}")
