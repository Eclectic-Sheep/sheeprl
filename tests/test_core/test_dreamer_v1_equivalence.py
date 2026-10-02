"""DreamerV1 must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/dreamer_v1.json` was recorded with the DreamerV1 training loop before the port (`main` at `118d6fb`), on CPU
in fp32: for each configuration below it holds every row written in the replay buffer,
every value given to the metric aggregator (the losses, the KL, the entropies and the gradient norms of every gradient
step, the episode statistics) and checksums of the final checkpoint (weights, optimizer states, replay ratio). The
images in the buffer are recorded as checksums. The test runs the same configurations and compares.

The continue model (`algo.world_model.use_continues`) is not covered: it crashes on `main` (known issue #11).

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct), from the root of the repository:
    python -m tests.test_core.test_dreamer_v1_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_dreamer_v1_equivalence
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List

import pytest
import torch

from tests.test_core.test_dreamer_v2_equivalence import overrides, record_run
from tests.test_core.test_sac_equivalence import max_difference

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "dreamer_v1.json")

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "env.sync_env=True",
    "env.capture_video=False",
    "env.num_envs=2",
    "fabric.accelerator=cpu",
    "fabric.devices=1",
    "fabric.precision=32-true",
    # The aggregator records the metrics of every gradient step; they are logged once, at the end
    "metric.log_level=1",
    "metric.log_every=1000000",
    "metric.disable_timer=True",
    "checkpoint.every=0",
    "checkpoint.save_last=True",
    "algo.run_test=False",
    "seed=7",
    "buffer.size=1000",
    # A small agent
    "algo.dense_units=16",
    "algo.mlp_layers=1",
    "algo.world_model.encoder.cnn_channels_multiplier=2",
    "algo.world_model.recurrent_model.recurrent_state_size=16",
    "algo.world_model.representation_model.hidden_size=16",
    "algo.world_model.transition_model.hidden_size=16",
    "algo.world_model.stochastic_size=4",
    "algo.horizon=4",
    "algo.per_rank_batch_size=2",
    "algo.per_rank_sequence_length=4",
    "algo.learning_starts=8",
]

VECTORS = [
    "algo.cnn_keys.encoder=[]",
    "algo.cnn_keys.decoder=[]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
]
PIXELS_AND_VECTORS = [
    "algo.cnn_keys.encoder=[rgb]",
    "algo.cnn_keys.decoder=[rgb]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
]

CONFIGS: Dict[str, List[str]] = {
    # Discrete actions, real rewards and terminations: 4 iterations of random actions, then one gradient step every
    # other iteration (2 envs, replay ratio 0.25), with pretraining steps at the first one
    "cartpole": [
        "env=gym",
        "env.id=CartPole-v1",
        *VECTORS,
        "algo.total_steps=48",
        "algo.replay_ratio=0.25",
        "algo.per_rank_pretrain_steps=16",
    ],
    # Continuous actions with a decay of the exploration noise (with known issue #17 the noise is `expl_min` from the
    # first step), episodes truncated by the time limit, 2 gradient steps per iteration, rewards squashed with tanh,
    # buffer in memory
    "pendulum_truncated": [
        "env=gym",
        "env.id=Pendulum-v1",
        "env.max_episode_steps=6",
        "env.clip_rewards=True",
        *VECTORS,
        "algo.total_steps=32",
        "algo.replay_ratio=1",
        "algo.actor.expl_decay=1000",
        "algo.actor.expl_min=0.1",
        "buffer.memmap=False",
    ],
    # Pixels and vectors, discrete actions (the exploration noise replaces some), terminations every 5 steps, no
    # gradient clipping of the world model
    "dummy_pixels": [
        "env=dummy",
        "env.id=discrete_dummy",
        *PIXELS_AND_VECTORS,
        "algo.total_steps=40",
        "algo.replay_ratio=0.5",
        "algo.world_model.clip_gradients=0",
    ],
    # Multi-discrete actions of 2 envs, episodes truncated every 5 steps (the actions played by the envs are the ones
    # of `main`, #10)
    "dummy_multidiscrete": [
        "env=dummy",
        "env.id=multidiscrete_dummy",
        "env.max_episode_steps=5",
        *VECTORS,
        "algo.total_steps=32",
        "algo.replay_ratio=0.5",
    ],
    # The dry run: one iteration
    "dry_run": [
        "env=dummy",
        "env.id=continuous_dummy",
        *VECTORS,
        "dry_run=True",
        "algo.per_rank_sequence_length=1",
    ],
}


def record_dreamer_v1(name: str) -> Dict[str, Any]:
    root_dir = os.path.join("pytest_equivalence", f"dreamer_v1_{name}")
    record, _ = record_run(
        ["exp=dreamer_v1", *overrides(COMMON_ARGS, CONFIGS[name]), f"root_dir={root_dir}", "run_name=run"]
    )
    log_dir = os.path.join("logs", "runs", root_dir)
    shutil.rmtree(log_dir, ignore_errors=True)
    if len(os.listdir(os.path.dirname(log_dir))) == 0:
        os.rmdir(os.path.dirname(log_dir))
    return record


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_dreamer_v1_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_dreamer_v1(name)
    # Same structure: rows written, metrics and the number of their values, checkpoint entries
    max_difference(record, reference)
    for key in ("ratio", "iter_num", "batch_size", "buffer_saved"):
        assert record["checkpoint"][key] == reference["checkpoint"][key], key
    torch.testing.assert_close(record["metrics"], reference["metrics"], rtol=1e-4, atol=1e-5)
    for step, reference_step in zip(record["steps"], reference["steps"]):
        torch.testing.assert_close(step, reference_step, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(record["checkpoint"], reference["checkpoint"], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record the reference instead of comparing with it")
    parser.add_argument("--reference", default=REFERENCE_PATH, help="the reference file to write or compare with")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), help="the configurations to run")
    cli_args = parser.parse_args()
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    records = {name: record_dreamer_v1(name) for name in cli_args.configs}
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
