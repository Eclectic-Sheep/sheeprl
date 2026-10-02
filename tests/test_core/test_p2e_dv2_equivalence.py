"""P2E-DV2 must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/p2e_dv2.json` was recorded with the P2E-DV2 training loops before the port (`main` at `118d6fb`), on CPU in
fp32. For each configuration below, an exploration run is followed by a finetuning run from its last checkpoint; for
both it holds every row written in the replay buffer, every value given to the metric aggregator and checksums of the
final checkpoint. The images in the buffer are recorded as checksums. The test runs the same configurations and
compares.

On `main` the finetuning never writes `truncated` in the buffer (known issue #12, fixed by the port), so `pendulum`,
whose finetuning plays episodes truncated by the time limit, was recorded with the port: checked against the reference
of `main` first, the only differences are the 4 `truncated` of the 2 rows ending those episodes (1 instead of 0). It
was recorded again when the continuous actions were normalized to [-1, 1] (#54), checked first: identical with
`algo.normalize_actions=False` (see `test_dreamer_v2_equivalence.py` for the actions).

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with training loops known to be correct), from the root of the repository:
    python -m tests.test_core.test_p2e_dv2_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_p2e_dv2_equivalence
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Tuple

import pytest
import torch

from tests.test_core.test_dreamer_v2_equivalence import COMMON_ARGS as DREAMER_V2_ARGS
from tests.test_core.test_dreamer_v2_equivalence import PIXELS_AND_VECTORS, VECTORS, overrides, record_run
from tests.test_core.test_sac_equivalence import max_difference

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "p2e_dv2.json")

COMMON_ARGS = [*DREAMER_V2_ARGS, "algo.ensembles.n=3", "algo.per_rank_pretrain_steps=0"]

# The overrides of the exploration and of the finetuning of each configuration
CONFIGS: Dict[str, Tuple[List[str], List[str]]] = {
    # Discrete actions, real rewards and terminations; the finetuning starts from the buffer of the exploration
    "cartpole": (
        ["env=gym", "env.id=CartPole-v1", *VECTORS, "algo.total_steps=48", "algo.replay_ratio=0.25"],
        ["env=gym", "env.id=CartPole-v1", "algo.total_steps=32", "buffer.load_from_exploration=True"],
    ),
    # Continuous actions (the intrinsic reward is backpropagated through the dynamics), truncated episodes, the
    # continue model, 2 gradient steps per iteration, the target critics copied every 2 gradient steps; the
    # finetuning starts from an empty buffer
    "pendulum": (
        [
            "env=gym",
            "env.id=Pendulum-v1",
            "env.max_episode_steps=6",
            *VECTORS,
            "algo.total_steps=32",
            "algo.replay_ratio=1",
            "algo.world_model.use_continues=True",
            "algo.critic.per_rank_target_network_update_freq=2",
        ],
        ["env=gym", "env.id=Pendulum-v1", "algo.total_steps=32", "algo.replay_ratio=1"],
    ),
    # Pixels and vectors, discrete actions, terminations every 5 steps, the episode buffer (the training starts once
    # some episodes are complete); the finetuning starts from the buffer of the exploration
    "dummy_episodes": (
        [
            "env=dummy",
            "env.id=discrete_dummy",
            *PIXELS_AND_VECTORS,
            "algo.learning_starts=16",
            "algo.total_steps=40",
            "algo.replay_ratio=0.5",
            "buffer.type=episode",
        ],
        [
            "env=dummy",
            "env.id=discrete_dummy",
            "algo.total_steps=24",
            "algo.replay_ratio=0.5",
            "buffer.type=episode",
            "buffer.load_from_exploration=True",
        ],
    ),
}


def record_p2e_dv2(name: str) -> Dict[str, Any]:
    """Run the exploration and then the finetuning of the configuration `name`."""
    exploration_args, finetuning_args = CONFIGS[name]
    root_dir = os.path.join("pytest_equivalence", f"p2e_dv2_{name}")
    exploration, ckpt_path = record_run(
        [
            "exp=p2e_dv2_exploration",
            *overrides(COMMON_ARGS, exploration_args),
            f"root_dir={root_dir}/exploration",
            "run_name=run",
        ]
    )
    finetuning, _ = record_run(
        [
            "exp=p2e_dv2_finetuning",
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
def test_p2e_dv2_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_p2e_dv2(name)
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
    records = {name: record_p2e_dv2(name) for name in cli_args.configs}
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
