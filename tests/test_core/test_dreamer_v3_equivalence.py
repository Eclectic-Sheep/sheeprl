"""DreamerV3 must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/dreamer_v3.json` was recorded with the DreamerV3 training loop before the port (`refactor/core` at
`ec5d039`, where DreamerV3 is still the one of `main` `118d6fb`), on CPU in fp32: for each configuration below it
holds every row written in the replay buffer (the reset rows of the ended episodes included), every value given to
the metric aggregator (the losses, the KL, the entropies and the gradient norms of every gradient step, the episode
statistics) and checksums of the final checkpoint (weights, optimizer states, return normalization, replay ratio).
The images in the buffer are recorded as checksums. The test runs the same configurations and compares.

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct), from the root of the repository:
    python -m tests.test_core.test_dreamer_v3_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_dreamer_v3_equivalence
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
from tests.test_core.test_sac_equivalence import checksum, max_difference

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "dreamer_v3.json")

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "exp=dreamer_v3",
    "env.sync_env=True",
    "env.capture_video=False",
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
    "algo.world_model.discrete_size=4",
    "algo.horizon=4",
    "algo.per_rank_batch_size=2",
    "algo.per_rank_sequence_length=4",
]

VECTORS = ["algo.cnn_keys.encoder=[]", "algo.cnn_keys.decoder=[]"]
PIXELS_AND_VECTORS = [
    "algo.cnn_keys.encoder=[rgb]",
    "algo.cnn_keys.decoder=[rgb]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
]

CONFIGS: Dict[str, List[str]] = {
    # Discrete actions, real rewards and terminations: 4 iterations of random actions, then one gradient step every
    # other iteration (2 envs, replay ratio 0.25)
    "cartpole": [
        "env=gym",
        "env.id=CartPole-v1",
        "env.num_envs=2",
        *VECTORS,
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
        "algo.total_steps=48",
        "algo.learning_starts=8",
        "algo.replay_ratio=0.25",
    ],
    # Continuous actions, episodes truncated by the time limit, 2 gradient steps per iteration, the target critic
    # updated every 2 gradient steps, rewards squashed with tanh, buffer in memory
    "pendulum_truncated": [
        "env=gym",
        "env.id=Pendulum-v1",
        "env.num_envs=2",
        "env.max_episode_steps=6",
        "env.clip_rewards=True",
        *VECTORS,
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
        "algo.total_steps=32",
        "algo.learning_starts=8",
        "algo.replay_ratio=1",
        "algo.critic.per_rank_target_network_update_freq=2",
        "buffer.memmap=False",
    ],
    # Pixels and vectors, discrete actions, terminations every 5 steps; decoupled RSSM, no gradient clipping of the
    # actor, a fixed initial recurrent state, a burst of pretraining steps
    "dummy_pixels_decoupled_rssm": [
        "env=dummy",
        "env.id=discrete_dummy",
        "env.num_envs=2",
        *PIXELS_AND_VECTORS,
        "algo.total_steps=32",
        "algo.learning_starts=8",
        "algo.replay_ratio=0.5",
        "algo.per_rank_pretrain_steps=3",
        "algo.world_model.decoupled_rssm=True",
        "algo.world_model.learnable_initial_recurrent_state=False",
        "algo.actor.clip_gradients=0",
    ],
    # Pixels and vectors, multi-discrete actions, truncations every 6 steps, no Hafner initialization
    "dummy_multidiscrete": [
        "env=dummy",
        "env.id=multidiscrete_dummy",
        "env.num_envs=2",
        "env.max_episode_steps=6",
        *PIXELS_AND_VECTORS,
        "algo.total_steps=32",
        "algo.learning_starts=8",
        "algo.replay_ratio=0.5",
        "algo.hafner_initialization=False",
    ],
    # One iteration, a buffer of two rows, no random actions
    "dry_run": [
        "env=dummy",
        "env.id=continuous_dummy",
        "env.num_envs=2",
        *PIXELS_AND_VECTORS,
        "dry_run=True",
        "algo.per_rank_sequence_length=1",
    ],
}


def overrides(name: str) -> List[str]:
    """The overrides of the configuration `name`: the common ones, replaced by its own when they set the same key."""
    keys = {arg.split("=")[0] for arg in CONFIGS[name]}
    return [arg for arg in COMMON_ARGS if arg.split("=")[0] not in keys] + CONFIGS[name]


def summary(value: Any) -> Any:
    """The values of a small array; the sum, the sum of squares and the shape of a large one (e.g. an image)."""
    array = np.asarray(value, dtype=np.float64)
    if array.size <= 64:
        return array.tolist()
    return [array.sum().item(), np.square(array).sum().item(), list(array.shape)]


def record_dreamer_v3(name: str) -> Dict[str, Any]:
    """Train DreamerV3 with the configuration `name` and record what it writes in the buffer, its metrics and its
    final checkpoint."""
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

    root_dir = os.path.join("pytest_equivalence", f"dreamer_v3_{name}")
    args = [os.path.join(ROOT_DIR, "__main__.py"), *overrides(name), f"root_dir={root_dir}", "run_name=run"]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", args),
        mock.patch.object(EnvIndependentReplayBuffer, "add", recording_add),
        mock.patch.object(MetricAggregator, "update", recording_update),
    ):
        run()

    log_dir = os.path.join("logs", "runs", root_dir)
    (ckpt_path,) = glob.glob(os.path.join(log_dir, "run", "version_*", "checkpoint", "*.ckpt"))
    state = torch.load(ckpt_path, weights_only=False)
    record["checkpoint"] = {
        **{k: checksum(state[k]) for k in ("world_model", "actor", "critic", "target_critic", "moments")},
        **{k: checksum(state[k]["state"]) for k in ("world_optimizer", "actor_optimizer", "critic_optimizer")},
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
def test_dreamer_v3_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_dreamer_v3(name)
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
    records = {name: record_dreamer_v3(name) for name in cli_args.configs}
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
