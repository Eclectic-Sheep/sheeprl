"""DreamerV2 must train exactly as before it was ported to the shared core (`sheeprl/core`).

`references/dreamer_v2.json` was recorded with the DreamerV2 training loop before the port (`main` at `118d6fb`), on CPU
in fp32: for each configuration below it holds every row written in the replay buffer (sequential or episode buffer),
every value given to the metric aggregator (the losses, the KL, the entropies and the gradient norms of every gradient
step, the episode statistics) and checksums of the final checkpoint (weights, optimizer states, replay ratio). The
images in the buffer are recorded as checksums. The test runs the same configurations and compares.

Two configurations were recorded again after fixes that change them on purpose, checked against the old reference
first: `pendulum_truncated` (the continuous actions normalized to [-1, 1], #54: with `algo.normalize_actions=False`
identical; with it the stored random actions are exactly half the old ones and Pendulum receives twice the stored
actions) and `dummy_multidiscrete` (the random multi-discrete actions of several envs stored as played, #10: every
stored action is now the played one, 3 of the 16 steps were not, and the rows before the first wrong one are
otherwise identical).

On the machine that recorded the reference the values match exactly. Other platforms can use a different BLAS, so
the test allows a small tolerance.

To record the reference again (only with a training loop known to be correct), from the root of the repository:
    python -m tests.test_core.test_dreamer_v2_equivalence --record
To print the largest difference from the reference for every configuration:
    python -m tests.test_core.test_dreamer_v2_equivalence
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

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "references", "dreamer_v2.json")

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
    "algo.world_model.discrete_size=4",
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
    # Continuous actions, episodes truncated by the time limit, the continue model, an actor objective mixing
    # REINFORCE and the dynamics, 2 gradient steps per iteration, the target critic copied every 2 gradient steps,
    # rewards squashed with tanh, buffer in memory
    "pendulum_truncated": [
        "env=gym",
        "env.id=Pendulum-v1",
        "env.max_episode_steps=6",
        "env.clip_rewards=True",
        *VECTORS,
        "algo.total_steps=32",
        "algo.replay_ratio=1",
        "algo.per_rank_pretrain_steps=0",
        "algo.world_model.use_continues=True",
        "algo.actor.objective_mix=0.5",
        "algo.critic.per_rank_target_network_update_freq=2",
        "buffer.memmap=False",
    ],
    # Pixels and vectors, discrete actions, the episode buffer (prioritizing the ends) filled by episodes truncated
    # every 5 steps (the training starts once some are complete), layer norms, no gradient clipping of the world model
    "dummy_pixels_episodes": [
        "env=dummy",
        "env.id=discrete_dummy",
        "env.max_episode_steps=5",
        *PIXELS_AND_VECTORS,
        "algo.learning_starts=16",
        "algo.total_steps=40",
        "algo.replay_ratio=0.5",
        "algo.per_rank_pretrain_steps=0",
        "algo.layer_norm=True",
        "algo.world_model.clip_gradients=0",
        "buffer.type=episode",
        "buffer.prioritize_ends=True",
    ],
    # Multi-discrete actions, episodes truncated every 5 steps
    "dummy_multidiscrete": [
        "env=dummy",
        "env.id=multidiscrete_dummy",
        "env.max_episode_steps=5",
        *VECTORS,
        "algo.total_steps=32",
        "algo.replay_ratio=0.5",
        "algo.per_rank_pretrain_steps=0",
        "algo.world_model.use_continues=True",
    ],
    # The dry run: one iteration, the episode buffer (every env ends its episode)
    "dry_run": [
        "env=dummy",
        "env.id=continuous_dummy",
        *VECTORS,
        "dry_run=True",
        "buffer.type=episode",
        "algo.per_rank_sequence_length=1",
    ],
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
    """Run the CLI with `args`, recording what is written in the buffer (sequential or episode buffer), the metrics
    and the final checkpoint.

    Returns the record and the path of the final checkpoint.
    """
    from sheeprl.cli import run
    from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer
    from sheeprl.utils.metric import MetricAggregator

    record: Dict[str, Any] = {"steps": [], "metrics": {}}
    aggregator_update = MetricAggregator.update

    def recording(add):
        def recording_add(self, data, indices=None, validate_args=False):
            step = {k: summary(v) for k, v in sorted(data.items())}
            step["indices"] = None if indices is None else list(indices)
            record["steps"].append(step)
            return add(self, data, indices, validate_args=validate_args)

        return recording_add

    def recording_update(self, name, value):
        if not self.disabled and name in self.metrics:
            value = value.detach().cpu() if isinstance(value, torch.Tensor) else value
            record["metrics"].setdefault(name, []).extend(summary(value) if np.ndim(value) else [float(value)])
        return aggregator_update(self, name, value)

    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", [os.path.join(ROOT_DIR, "__main__.py"), *args]),
        mock.patch.object(EnvIndependentReplayBuffer, "add", recording(EnvIndependentReplayBuffer.add)),
        mock.patch.object(EpisodeBuffer, "add", recording(EpisodeBuffer.add)),
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


def record_dreamer_v2(name: str) -> Dict[str, Any]:
    root_dir = os.path.join("pytest_equivalence", f"dreamer_v2_{name}")
    record, _ = record_run(
        ["exp=dreamer_v2", *overrides(COMMON_ARGS, CONFIGS[name]), f"root_dir={root_dir}", "run_name=run"]
    )
    log_dir = os.path.join("logs", "runs", root_dir)
    shutil.rmtree(log_dir, ignore_errors=True)
    if len(os.listdir(os.path.dirname(log_dir))) == 0:
        os.rmdir(os.path.dirname(log_dir))
    return record


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_dreamer_v2_matches_the_reference(name):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)[name]
    record = record_dreamer_v2(name)
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
    records = {name: record_dreamer_v2(name) for name in cli_args.configs}
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
