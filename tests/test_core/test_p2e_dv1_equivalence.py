"""P2E-DV1 must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/p2e_dv1.json` was recorded with the P2E-DV1 training loops before the port (`main` at `118d6fb`), on CPU in
fp32. For each configuration below, an exploration run is followed by a finetuning run from its last checkpoint; for
both it holds every row written in the replay buffer, every value given to the metric aggregator and checksums of the
final checkpoint. The images in the buffer are recorded as checksums. The test runs the same configurations and
compares.

All the configurations were recorded again after the fixes of DreamerV1 and P2E-DV1 (#10, #11, #17, #24, #54, #59, #60),
checked against the reference of `main` first: with every fix disabled the code gave it exactly; with one fix enabled
only the configurations it concerns changed: the multi-discrete ones (#10: the actions stored as played), the one with
an exploration decay (#17), the Pendulum ones (#54: the actions normalized to [-1, 1]), the discrete ones with
exploration noise (#59: a random draw per environment), all those that train on sequences (#24: the episode starts in
the sequences); #60 changes none (they use the default minimum std). The continue model
(`algo.world_model.use_continues`), which crashed on `main` (#11), is now used by the Pendulum configurations.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with training loops known to be correct), from the root of the repository:
    python -m tests.test_core.test_p2e_dv1_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_p2e_dv1_equivalence
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Tuple

import pytest
import torch

from tests.test_core.test_dreamer_v1_equivalence import COMMON_ARGS as DREAMER_V1_ARGS
from tests.test_core.test_dreamer_v2_equivalence import PIXELS_AND_VECTORS, VECTORS, overrides, record_run
from tests.test_core.test_sac_equivalence import max_difference

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "p2e_dv1.json")

COMMON_ARGS = [*DREAMER_V1_ARGS, "algo.ensembles.n=3"]

# The overrides of the exploration and of the finetuning of each configuration
CONFIGS: Dict[str, Tuple[List[str], List[str]]] = {
    # Discrete actions, real rewards and terminations; the finetuning starts from the buffer of the exploration
    "cartpole": (
        ["env=gym", "env.id=CartPole-v1", *VECTORS, "algo.total_steps=48", "algo.replay_ratio=0.25"],
        ["env=gym", "env.id=CartPole-v1", "algo.total_steps=32", "buffer.load_from_exploration=True"],
    ),
    # Continuous actions (the intrinsic reward is backpropagated through the dynamics), truncated episodes, the
    # continue model, 2 gradient steps per iteration; the finetuning starts from an empty buffer
    "pendulum": (
        [
            "env=gym",
            "env.id=Pendulum-v1",
            "env.max_episode_steps=6",
            *VECTORS,
            "algo.world_model.use_continues=True",
            "algo.total_steps=32",
            "algo.replay_ratio=1",
        ],
        ["env=gym", "env.id=Pendulum-v1", "algo.total_steps=32", "algo.replay_ratio=1"],
    ),
    # Pixels and vectors, discrete actions, terminations every 5 steps; the finetuning starts from the buffer of the
    # exploration
    "dummy_pixels": (
        ["env=dummy", "env.id=discrete_dummy", *PIXELS_AND_VECTORS, "algo.total_steps=40", "algo.replay_ratio=0.5"],
        [
            "env=dummy",
            "env.id=discrete_dummy",
            "algo.total_steps=24",
            "algo.replay_ratio=0.5",
            "buffer.load_from_exploration=True",
        ],
    ),
    # Multi-discrete actions of 2 envs, truncated every 5 steps (the actions the envs play are the ones of `main`, #10)
    "dummy_multidiscrete": (
        [
            "env=dummy",
            "env.id=multidiscrete_dummy",
            "env.max_episode_steps=5",
            *VECTORS,
            "algo.total_steps=32",
            "algo.replay_ratio=0.5",
        ],
        ["env=dummy", "env.id=multidiscrete_dummy", "algo.total_steps=24", "algo.replay_ratio=0.5"],
    ),
}


def record_p2e_dv1(name: str) -> Dict[str, Any]:
    """Run the exploration and then the finetuning of the configuration `name`."""
    exploration_args, finetuning_args = CONFIGS[name]
    root_dir = os.path.join("pytest_equivalence", f"p2e_dv1_{name}")
    exploration, ckpt_path = record_run(
        [
            "exp=p2e_dv1_exploration",
            *overrides(COMMON_ARGS, exploration_args),
            f"root_dir={root_dir}/exploration",
            "run_name=run",
        ]
    )
    finetuning, _ = record_run(
        [
            "exp=p2e_dv1_finetuning",
            *overrides(COMMON_ARGS, finetuning_args),
            f"checkpoint.exploration_ckpt_path={ckpt_path}",
            f"root_dir={root_dir}/finetuning",
            "run_name=run",
        ]
    )
    log_dir = os.path.join("logs", "runs", root_dir)
    shutil.rmtree(log_dir, ignore_errors=True)
    if len(os.listdir(os.path.dirname(log_dir))) == 0:
        os.rmdir(os.path.dirname(log_dir))
    return {"exploration": exploration, "finetuning": finetuning}


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_p2e_dv1_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_p2e_dv1(name)
    # Same structure: rows written, metrics and the number of their values, checkpoint entries
    max_difference(record, reference)
    for phase in ("exploration", "finetuning"):
        for key in ("ratio", "iter_num", "batch_size", "buffer_saved"):
            assert record[phase]["checkpoint"][key] == reference[phase]["checkpoint"][key], (phase, key)
        torch.testing.assert_close(record[phase]["metrics"], reference[phase]["metrics"], rtol=1e-4, atol=1e-5)
        for step, reference_step in zip(record[phase]["steps"], reference[phase]["steps"]):
            torch.testing.assert_close(step, reference_step, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(record[phase]["checkpoint"], reference[phase]["checkpoint"], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record the reference instead of comparing with it")
    parser.add_argument("--reference", default=REFERENCE_PATH, help="the reference file to write or compare with")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), help="the configurations to run")
    cli_args = parser.parse_args()
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    records = {name: record_p2e_dv1(name) for name in cli_args.configs}
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
