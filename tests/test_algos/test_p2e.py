"""Plan2Explore (P2E-DV1, P2E-DV2, P2E-DV3): the update of the ensembles, their optimizer, the rows of a finetuning."""

import copy
import glob
import importlib
import inspect
import os
import shutil
import sys
from unittest import mock

import numpy as np
import pytest

from sheeprl import ROOT_DIR
from sheeprl.utils.imports import _IS_WINDOWS

P2E_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "dry_run=True",
    "env=dummy",
    "env.id=discrete_dummy",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=0",
    "checkpoint.save_last=True",
    "algo.run_test=False",
    "algo.cnn_keys.encoder=[]",
    "algo.cnn_keys.decoder=[]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
    "algo.dense_units=8",
    "algo.world_model.recurrent_model.recurrent_state_size=8",
    "algo.world_model.representation_model.hidden_size=8",
    "algo.world_model.transition_model.hidden_size=8",
    "algo.horizon=4",
    "algo.per_rank_batch_size=1",
    "algo.learning_starts=0",
    "algo.replay_ratio=1",
    "algo.ensembles.n=2",
    "buffer.size=10",
]


def run_p2e(args, root_dir):
    from sheeprl.cli import run

    argv = [os.path.join(ROOT_DIR, "__main__.py"), *P2E_ARGS, *args, f"root_dir={root_dir}"]
    with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(sys, "argv", argv):
        run()


@pytest.mark.skipif(_IS_WINDOWS, reason="The CPU bf16 matmul crashes on part of the Windows runners")
@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_the_ensembles_are_trained_in_mixed_precision(version):
    # Their loss was back-propagated with `loss.backward()`: Fabric's modules in mixed precision require
    # `fabric.backward`, and the training crashed
    root_dir = f"pytest_p2e_dv{version}_mixed"
    sequence_length = "1" if version == "3" else "2"
    try:
        run_p2e(
            [
                f"exp=p2e_dv{version}_exploration",
                "fabric.precision=bf16-mixed",
                f"algo.per_rank_sequence_length={sequence_length}",
            ],
            root_dir,
        )
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)


@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_the_ensembles_are_trained_with_their_optimizer(version):
    # Their optimizer was the one of the critic (P2E-DV2, P2E-DV3) or of the world model (P2E-DV1), not
    # `algo.ensembles.optimizer`
    module = importlib.import_module(f"sheeprl.algos.p2e_dv{version}.p2e_dv{version}_exploration")
    optimizers = []

    def recording_train(*args, **kwargs):
        optimizers.append(inspect.signature(module_train).bind(*args, **kwargs).arguments["ensemble_optimizer"])
        return module_train(*args, **kwargs)

    module_train = module.train
    root_dir = f"pytest_p2e_dv{version}_ensemble_optimizer"
    try:
        with mock.patch.object(module, "train", recording_train):
            run_p2e(
                [
                    f"exp=p2e_dv{version}_exploration",
                    f"algo.per_rank_sequence_length={'1' if version == '3' else '2'}",
                    "algo.ensembles.optimizer.lr=0.0123",
                    "algo.ensembles.optimizer.weight_decay=0.0456",
                ],
                root_dir,
            )
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert len(optimizers) > 0
    for optimizer in optimizers:
        (group,) = optimizer.param_groups
        assert group["lr"] == 0.0123 and group["weight_decay"] == 0.0456


def test_the_p2e_dv2_finetuning_stores_the_truncated_episodes():
    # It wrote `terminated` twice and never `truncated`: the episode buffer never closed the truncated episodes
    from sheeprl.data.buffers import EnvIndependentReplayBuffer

    root_dir = "pytest_p2e_dv2_truncated"
    # Episodes of 3 steps in 4 iterations, played and not trained
    args = ["dry_run=False", "algo.total_steps=8", "env.max_episode_steps=3", "algo.replay_ratio=0"]
    rows = []
    buffer_add = EnvIndependentReplayBuffer.add

    def recording_add(self, data, *args, **kwargs):
        rows.append(copy.deepcopy({k: np.asarray(v) for k, v in data.items()}))
        return buffer_add(self, data, *args, **kwargs)

    try:
        run_p2e(["exp=p2e_dv2_exploration", "algo.per_rank_sequence_length=2", *args, "run_name=exploration"], root_dir)
        (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, "exploration", "version_*", "checkpoint", "*"))
        with mock.patch.object(EnvIndependentReplayBuffer, "add", recording_add):
            run_p2e(
                [
                    "exp=p2e_dv2_finetuning",
                    "algo.per_rank_sequence_length=2",
                    *args,
                    f"checkpoint.exploration_ckpt_path={ckpt_path}",
                    "run_name=finetuning",
                ],
                root_dir,
            )
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    # The step that reaches the time limit, in both environments
    assert [row["truncated"].sum() for row in rows if row["truncated"].any()] == [2]
    assert not any(row["terminated"].any() for row in rows)
