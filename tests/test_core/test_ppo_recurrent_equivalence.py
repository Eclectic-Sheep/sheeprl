"""PPO-recurrent must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/ppo_recurrent.json` was recorded with the PPO-recurrent training loop before the port (`main` at
`118d6fb`), on CPU in fp32: for each configuration below it holds the actions, log-probs and values played at every
environment step, the three losses of every minibatch and checksums of the final checkpoint (weights, optimizer
state). The test runs the same configurations and compares. The continuous actions crash on `main` (known issue
#29): `continuous` was recorded after their fix, as a regression reference. `algo.clip_vloss` is not covered: its value
loss lost its factor ½ (#33) after `118d6fb`.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct), from the root of the repository:
    python -m tests.test_core.test_ppo_recurrent_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_ppo_recurrent_equivalence
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

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "ppo_recurrent.json")

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "exp=ppo_recurrent",
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
    "algo.update_epochs=2",
]

CONFIGS: Dict[str, List[str]] = {
    # CartPole with the default recipe: 3 iterations of 2 envs x 16 steps, episodes split in sequences of 4, in 2
    # minibatches of sequences
    "cartpole_default": [
        "algo.rollout_steps=16",
        "algo.per_rank_sequence_length=4",
        "algo.per_rank_num_batches=2",
        "algo.total_steps=96",
    ],
    # Every option of the loss and of the schedules, the recurrent state kept across the episodes, the MLPs before and
    # after the LSTM, 3 minibatches (the last one smaller), sequences of 5 (the rollout isn't a multiple of it)
    "cartpole_all_options": [
        "algo.rollout_steps=16",
        "algo.per_rank_sequence_length=5",
        "algo.per_rank_num_batches=3",
        "algo.total_steps=96",
        "algo.anneal_lr=True",
        "algo.anneal_clip_coef=True",
        "algo.anneal_ent_coef=True",
        "algo.ent_coef=0.01",
        "algo.normalize_advantages=True",
        "algo.max_grad_norm=0.5",
        "algo.loss_reduction=sum",
        "algo.reset_recurrent_state_on_done=False",
        "algo.rnn.pre_rnn_mlp.apply=True",
        "algo.rnn.post_rnn_mlp.apply=True",
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
        "algo.rollout_steps=8",
        "algo.per_rank_sequence_length=2",
        "algo.per_rank_num_batches=2",
        "algo.total_steps=32",
    ],
    # Recorded after the fix of the continuous actions (#29), which crashed on `main`
    "continuous": [
        "env=dummy",
        "env.id=continuous_dummy",
        "algo.cnn_keys.encoder=[]",
        "algo.mlp_keys.encoder=[state]",
        "algo.rollout_steps=8",
        "algo.per_rank_sequence_length=3",
        "algo.per_rank_num_batches=2",
        "algo.total_steps=32",
    ],
    "multidiscrete": [
        "env=dummy",
        "env.id=multidiscrete_dummy",
        "algo.cnn_keys.encoder=[]",
        "algo.mlp_keys.encoder=[state]",
        "algo.rollout_steps=8",
        "algo.per_rank_sequence_length=3",
        "algo.per_rank_num_batches=2",
        "algo.total_steps=32",
    ],
}


def record_ppo_recurrent(name: str) -> Dict[str, Any]:
    """Train PPO-recurrent with the configuration `name` and record what it plays, its losses and its final
    checkpoint."""
    from sheeprl.algos.ppo_recurrent import agent, ppo_recurrent
    from sheeprl.cli import run

    record: Dict[str, Any] = {"actions": [], "logprobs": [], "values": [], "losses": []}
    player_forward = agent.RecurrentPPOPlayer.forward
    policy_loss = ppo_recurrent.policy_loss
    value_loss = ppo_recurrent.value_loss
    entropy_loss = ppo_recurrent.entropy_loss
    losses: Dict[str, float] = {}

    def recording_player_forward(self, *args, **kwargs):
        actions, logprobs, values, states = player_forward(self, *args, **kwargs)
        record["actions"].append(torch.cat(actions, dim=-1).tolist())
        record["logprobs"].append(logprobs.tolist())
        record["values"].append(values.tolist())
        return actions, logprobs, values, states

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

    root_dir = os.path.join("pytest_equivalence", f"ppo_recurrent_{name}")
    args = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *CONFIGS[name], f"root_dir={root_dir}", "run_name=run"]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", args),
        mock.patch.object(agent.RecurrentPPOPlayer, "forward", recording_player_forward),
        mock.patch.object(ppo_recurrent, "policy_loss", recording_policy_loss),
        mock.patch.object(ppo_recurrent, "value_loss", recording_value_loss),
        mock.patch.object(ppo_recurrent, "entropy_loss", recording_entropy_loss),
    ):
        run()

    log_dir = os.path.join("logs", "runs", root_dir)
    (ckpt_path,) = glob.glob(os.path.join(log_dir, "run", "version_*", "checkpoint", "*.ckpt"))
    state = torch.load(ckpt_path, weights_only=False)
    shutil.rmtree(log_dir, ignore_errors=True)
    if len(os.listdir(os.path.dirname(log_dir))) == 0:
        os.rmdir(os.path.dirname(log_dir))
    record["checkpoint"] = {
        "agent": checksum(state["agent"]),
        "optimizer": checksum(state["optimizer"]["state"]),
        "iter_num": state["iter_num"],
    }
    return record


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_ppo_recurrent_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_ppo_recurrent(name)
    # Same structure: steps played, minibatches, checkpoint entries
    max_difference(record, reference)
    assert record["checkpoint"]["iter_num"] == reference["checkpoint"]["iter_num"]
    for key in ("actions", "logprobs", "values", "losses"):
        torch.testing.assert_close(record[key], reference[key], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["checkpoint"], reference["checkpoint"], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record the reference instead of comparing with it")
    cli_args = parser.parse_args()
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    records = {name: record_ppo_recurrent(name) for name in CONFIGS}
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
