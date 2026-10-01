"""A2C must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/a2c.json` was recorded with the A2C training loop before the port (`main` at `118d6fb`), on CPU in fp32:
for each configuration below it holds the actions, log-probs and values played at every environment step, the three
losses of every minibatch, and checksums of the final checkpoint (weights, optimizer and scheduler states). The test
runs the same configurations and compares.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct), from the root of the repository:
    python -m tests.test_core.test_a2c_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_a2c_equivalence
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
from tests.test_core.test_sac_equivalence import checksum, max_difference

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "a2c.json")

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "exp=a2c",
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
    # CartPole with the default recipe: 4 iterations of 2 envs x 5 steps, one minibatch per iteration
    "cartpole_default": ["algo.total_steps=40"],
    # Every option on: 3 minibatches per iteration (10 = 4 + 4 + 2) whose gradients are accumulated, advantage
    # normalization (not a key of the A2C config: added with `+`), gradient clipping, entropy, mean reduction, the
    # learning-rate scheduler (never stepped, #31)
    "cartpole_all_options": [
        "algo.total_steps=40",
        "algo.per_rank_batch_size=4",
        "+algo.normalize_advantages=True",
        "algo.max_grad_norm=0.5",
        "algo.ent_coef=0.01",
        "algo.loss_reduction=mean",
        "algo.anneal_lr=True",
    ],
    # Pixels and vectors, frame stacking, episodes truncated by the time limit (bootstrapped values)
    "discrete_pixels_truncated": [
        "env=dummy",
        "env.id=discrete_dummy",
        "env.max_episode_steps=3",
        "env.frame_stack=2",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.encoder.cnn_features_dim=16",
        "algo.rollout_steps=4",
        "algo.total_steps=32",
    ],
    "multidiscrete": [
        "env=dummy",
        "env.id=multidiscrete_dummy",
        "algo.cnn_keys.encoder=[]",
        "algo.mlp_keys.encoder=[state]",
        "algo.rollout_steps=4",
        "algo.per_rank_batch_size=3",
        "algo.total_steps=32",
    ],
    "continuous": [
        "env=dummy",
        "env.id=continuous_dummy",
        "algo.cnn_keys.encoder=[]",
        "algo.mlp_keys.encoder=[state]",
        "algo.rollout_steps=4",
        "algo.total_steps=32",
    ],
}


def overrides(name: str) -> List[str]:
    """The overrides of the configuration `name`: the common ones, replaced by its own when they set the same key."""
    keys = {arg.split("=")[0] for arg in CONFIGS[name]}
    return [arg for arg in COMMON_ARGS if arg.split("=")[0] not in keys] + CONFIGS[name]


def record_a2c(name: str) -> Dict[str, Any]:
    """Train A2C with the configuration `name` and record what it plays, its losses and its final checkpoint."""
    from sheeprl.algos.a2c import a2c
    from sheeprl.algos.ppo import agent
    from sheeprl.cli import run

    record: Dict[str, Any] = {"actions": [], "logprobs": [], "values": [], "losses": []}
    player_forward = agent.PPOPlayer.forward
    policy_loss, value_loss, entropy_loss = a2c.policy_loss, a2c.value_loss, a2c.entropy_loss
    losses: Dict[str, torch.Tensor] = {}

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

    root_dir = os.path.join("pytest_equivalence", f"a2c_{name}")
    args = [os.path.join(ROOT_DIR, "__main__.py"), *overrides(name), f"root_dir={root_dir}", "run_name=run"]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", args),
        mock.patch.object(agent.PPOPlayer, "forward", recording_player_forward),
        mock.patch.object(a2c, "policy_loss", recording_policy_loss),
        mock.patch.object(a2c, "value_loss", recording_value_loss),
        mock.patch.object(a2c, "entropy_loss", recording_entropy_loss),
    ):
        run()

    log_dir = os.path.join("logs", "runs", root_dir)
    (ckpt_path,) = glob.glob(os.path.join(log_dir, "run", "version_*", "checkpoint", "*.ckpt"))
    state = torch.load(ckpt_path, weights_only=False)
    record["checkpoint"] = {
        "agent": checksum(state["agent"]),
        "optimizer": checksum(state["optimizer"]["state"]),
        "scheduler": checksum(state["scheduler"]),
        "iter_num": state["iter_num"],
        "batch_size": state["batch_size"],
    }
    del state
    shutil.rmtree(log_dir, ignore_errors=True)
    if len(os.listdir(os.path.dirname(log_dir))) == 0:
        os.rmdir(os.path.dirname(log_dir))
    return record


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_a2c_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_a2c(name)
    # Same structure: steps played, minibatches, checkpoint entries
    max_difference(record, reference)
    for key in ("iter_num", "batch_size"):
        assert record["checkpoint"][key] == reference["checkpoint"][key], key
    for key in ("actions", "logprobs", "values", "losses", "checkpoint"):
        torch.testing.assert_close(record[key], reference[key], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record the reference instead of comparing with it")
    parser.add_argument("--reference", default=REFERENCE_PATH, help="the reference file to write or compare with")
    cli_args = parser.parse_args()
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    records = {name: record_a2c(name) for name in CONFIGS}
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
