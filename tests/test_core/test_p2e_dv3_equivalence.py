"""P2E-DV3 must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/p2e_dv3.json` was recorded with the P2E-DV3 training loops before the port (`main` at `118d6fb`), on CPU in
fp32. For each configuration below, an exploration run is followed by a finetuning run from its last checkpoint; for
both it holds every row written in the replay buffer, every value given to the metric aggregator and checksums of the
final checkpoint. The images in the buffer are recorded as checksums. The test runs the same configurations and
compares.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with training loops known to be correct), from the root of the repository:
    python -m tests.test_core.test_p2e_dv3_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_p2e_dv3_equivalence
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Tuple
from unittest import mock

import numpy as np
import pytest
import torch

from sheeprl import ROOT_DIR
from tests.test_core.test_sac_equivalence import checksum, max_difference

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "p2e_dv3.json")

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
    "buffer.checkpoint=True",
    "algo.total_steps=32",
    "algo.learning_starts=8",
    "algo.replay_ratio=0.5",
    # A small agent
    "algo.dense_units=16",
    "algo.mlp_layers=1",
    "algo.world_model.encoder.cnn_channels_multiplier=2",
    "algo.world_model.recurrent_model.recurrent_state_size=16",
    "algo.world_model.representation_model.hidden_size=16",
    "algo.world_model.transition_model.hidden_size=16",
    "algo.world_model.stochastic_size=4",
    "algo.world_model.discrete_size=4",
    "algo.ensembles.n=3",
    "algo.horizon=4",
    "algo.per_rank_batch_size=2",
    "algo.per_rank_sequence_length=4",
]

VECTORS = [
    "algo.cnn_keys.encoder=[]",
    "algo.cnn_keys.decoder=[]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
]

# The overrides of the exploration and of the finetuning of each configuration
CONFIGS: Dict[str, Tuple[List[str], List[str]]] = {
    # Discrete actions, real rewards and terminations; the finetuning starts from the buffer of the exploration
    "cartpole": (
        ["env=gym", "env.id=CartPole-v1", *VECTORS],
        ["env=gym", "env.id=CartPole-v1", "buffer.load_from_exploration=True"],
    ),
    # Continuous actions (the intrinsic reward is backpropagated through the dynamics), truncated episodes, 2 gradient
    # steps per iteration, the target critics updated every 2 gradient steps; the finetuning starts from an empty buffer
    "pendulum": (
        [
            "env=gym",
            "env.id=Pendulum-v1",
            "env.max_episode_steps=6",
            *VECTORS,
            "algo.replay_ratio=1",
            "algo.critic.per_rank_target_network_update_freq=2",
        ],
        ["env=gym", "env.id=Pendulum-v1", "algo.replay_ratio=1"],
    ),
    # Pixels and vectors, multi-discrete actions, terminations every 129 steps and truncations every 6
    "dummy_pixels": (
        [
            "env=dummy",
            "env.id=multidiscrete_dummy",
            "env.max_episode_steps=6",
            "algo.cnn_keys.encoder=[rgb]",
            "algo.cnn_keys.decoder=[rgb]",
            "algo.mlp_keys.encoder=[state]",
            "algo.mlp_keys.decoder=[state]",
        ],
        ["env=dummy", "env.id=multidiscrete_dummy", "buffer.load_from_exploration=True"],
    ),
}


def overrides(common: List[str], own: List[str]) -> List[str]:
    """The common overrides, replaced by the own ones when they set the same key."""
    keys = {arg.split("=")[0] for arg in own}
    return [arg for arg in common if arg.split("=")[0] not in keys] + own


def summary(value: Any) -> Any:
    """The values of a small array; the sum, the sum of squares and the shape of a large one (e.g. an image)."""
    array = np.asarray(value, dtype=np.float64)
    if array.size <= 64:
        return array.tolist()
    return [array.sum().item(), np.square(array).sum().item(), list(array.shape)]


def record_run(args: List[str]) -> Tuple[Dict[str, Any], str]:
    """Run the CLI with `args`, recording what is written in the buffer, the metrics and the final checkpoint.

    Returns the record and the path of the final checkpoint.
    """
    from sheeprl.cli import run
    from sheeprl.data.buffers import EnvIndependentReplayBuffer
    from sheeprl.utils.metric import MetricAggregator

    record: Dict[str, Any] = {"steps": [], "metrics": {}}
    buffer_add = EnvIndependentReplayBuffer.add
    aggregator_update = MetricAggregator.update

    def recording_add(self, data, indices=None, validate_args=False):
        step = {k: summary(v) for k, v in sorted(data.items())}
        step["indices"] = None if indices is None else list(indices)
        record["steps"].append(step)
        return buffer_add(self, data, indices, validate_args=validate_args)

    def recording_update(self, name, value):
        if not self.disabled and name in self.metrics:
            value = value.detach().cpu() if isinstance(value, torch.Tensor) else value
            record["metrics"].setdefault(name, []).extend(summary(value) if np.ndim(value) else [float(value)])
        return aggregator_update(self, name, value)

    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", [os.path.join(ROOT_DIR, "__main__.py"), *args]),
        mock.patch.object(EnvIndependentReplayBuffer, "add", recording_add),
        mock.patch.object(MetricAggregator, "update", recording_update),
    ):
        run()

    root_dir = next(arg.split("=", 1)[1] for arg in args if arg.startswith("root_dir="))
    (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, "run", "version_*", "checkpoint", "*.ckpt"))
    state = torch.load(ckpt_path, weights_only=False)
    # Everything but the buffer and the counter added by the core
    record["checkpoint"] = {
        k: checksum(v) for k, v in sorted(state.items()) if k not in ("rb", "per_rank_gradient_steps")
    }
    record["checkpoint"]["buffer_saved"] = "rb" in state
    return record, ckpt_path


def record_p2e_dv3(name: str) -> Dict[str, Any]:
    """Run the exploration and then the finetuning of the configuration `name`."""
    exploration_args, finetuning_args = CONFIGS[name]
    root_dir = os.path.join("pytest_equivalence", f"p2e_dv3_{name}")
    exploration, ckpt_path = record_run(
        [
            "exp=p2e_dv3_exploration",
            *overrides(COMMON_ARGS, exploration_args),
            f"root_dir={root_dir}/exploration",
            "run_name=run",
        ]
    )
    finetuning, _ = record_run(
        [
            "exp=p2e_dv3_finetuning",
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
def test_p2e_dv3_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_p2e_dv3(name)
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
    records = {name: record_p2e_dv3(name) for name in cli_args.configs}
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
