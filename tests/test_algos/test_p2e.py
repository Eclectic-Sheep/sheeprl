"""Plan2Explore (P2E-DV1, P2E-DV2, P2E-DV3): the update of the ensembles, their optimizer, the finetunings."""

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
import torch

from sheeprl import ROOT_DIR
from sheeprl.utils import compile as compile_utils
from sheeprl.utils.imports import _IS_WINDOWS

from .compiled import assert_same_step, no_host_reads, recording, same_random_numbers

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
        assert group["lr"] == 0.0123
        if version == "2":
            # The weight decay of DreamerV2, which multiplies the weights by `1 - weight_decay` before every step: AdamW
            # with the weight decay divided by the learning rate
            assert isinstance(optimizer.optimizer, torch.optim.AdamW)
            assert group["lr"] * group["weight_decay"] == pytest.approx(0.0456)
        else:
            assert group["weight_decay"] == 0.0456


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


def test_a_p2e_dv3_finetuning_keeps_the_slow_critic_of_the_exploration():
    # The first gradient step of a finetuning copied its critic into its target critic, as a new DreamerV3 does: the
    # slow critic learned by the exploration was discarded
    from sheeprl.algos.p2e_dv3 import p2e_dv3_finetuning

    root_dir = "pytest_p2e_dv3_slow_critic"
    args = ["algo.per_rank_sequence_length=1"]
    critics = []

    def first_train(*args, **kwargs):
        # The task critic and target critic at the first gradient step
        if len(critics) == 0:
            critics.append({k: v.clone() for k, v in args[3].module.state_dict().items()})
            critics.append({k: v.clone() for k, v in args[4].state_dict().items()})
        return finetuning_train(*args, **kwargs)

    finetuning_train = p2e_dv3_finetuning.train
    try:
        run_p2e(["exp=p2e_dv3_exploration", *args, "run_name=exploration"], root_dir)
        (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, "exploration", "version_*", "checkpoint", "*"))
        saved = torch.load(ckpt_path, weights_only=False)
        with mock.patch.object(p2e_dv3_finetuning, "train", first_train):
            run_p2e(
                [
                    "exp=p2e_dv3_finetuning",
                    *args,
                    f"checkpoint.exploration_ckpt_path={ckpt_path}",
                    "run_name=finetuning",
                ],
                root_dir,
            )
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    critic, target_critic = critics
    tau = 0.02
    for k, v in saved["critic_task"].items():
        torch.testing.assert_close(critic[k], v)
        torch.testing.assert_close(target_critic[k], tau * v + (1 - tau) * saved["target_critic_task"][k])


def test_the_p2e_dv3_exploration_trains_the_decoupled_rssm():
    # Its world model unrolled the RSSM with the arguments of the coupled one: with the decoupled RSSM it crashed
    root_dir = "pytest_p2e_dv3_decoupled"
    try:
        run_p2e(
            ["exp=p2e_dv3_exploration", "algo.per_rank_sequence_length=1", "algo.world_model.decoupled_rssm=True"],
            root_dir,
        )
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_the_exploration_losses_compile_into_single_graphs(monkeypatch, version, continuous):
    # With `algo.compile.enabled`, every loss of the exploration and of the task behaviour is traced into a single
    # graph, without reading a tensor on the host
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    compile = torch.compile

    def single_graph(fn, mode=None):
        graph = compile(fn, backend="eager", fullgraph=True)

        def traced(*args, **kwargs):
            with monkeypatch.context() as patch:
                no_host_reads(patch)
                return graph(*args, **kwargs)

        return traced

    monkeypatch.setattr(torch, "compile", single_graph)
    root_dir = f"pytest_p2e_dv{version}_single_graphs_{continuous}"
    # Compiled as in a new process: Dynamo makes dynamic the arguments that changed between the compilations of the
    # earlier tests (e.g. the clipping of the actions, 0 for the discrete ones), which a run keeps constant
    torch._dynamo.reset()
    try:
        run_p2e(
            [
                f"exp=p2e_dv{version}_exploration",
                f"env.id={'continuous' if continuous else 'discrete'}_dummy",
                f"algo.per_rank_sequence_length={'1' if version == '3' else '2'}",
                "algo.compile.enabled=True",
            ],
            root_dir,
        )
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
        torch._dynamo.reset()
    assert len(compile_utils._COMPILED) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_the_compiled_exploration_losses_are_the_eager_ones(monkeypatch, version, continuous):
    # The same weights, the same batch and the same random numbers: the same losses and gradients of three gradient
    # steps of the exploration (the CUDA graphs are recorded at the second and replayed at the third), with and
    # without `torch.compile`. The exploration and the task behaviours, and the exploration critics of P2E-DV3, run
    # the same compiled losses with modules of their own
    module = importlib.import_module(f"sheeprl.algos.p2e_dv{version}.p2e_dv{version}_exploration")
    called = [importlib.import_module("sheeprl.algos.dreamer_v3.dreamer_v3")] if version == "3" else []
    module_train = module.train
    same_random_numbers(monkeypatch)
    args = [
        f"exp=p2e_dv{version}_exploration",
        f"env.id={'continuous' if continuous else 'discrete'}_dummy",
        "fabric.accelerator=cuda",
        "float32_matmul_precision=highest",
        "torch_backends_cudnn_benchmark=False",
        "dry_run=False",
        "checkpoint.save_last=False",
        "algo.total_steps=12",
        "algo.learning_starts=8",
        "algo.replay_ratio=0.5",
        "algo.per_rank_batch_size=2",
        "algo.per_rank_sequence_length=4",
        "algo.horizon=3",
    ]
    if version != "1":
        # The objective of the actor mixes the dynamics and REINFORCE
        args.append("algo.actor.objective_mix=0.5")
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(compile_utils, "_COMPILED", {})
        root_dir = f"pytest_p2e_dv{version}_compiled_{continuous}_{enabled}"
        with monkeypatch.context() as patch:
            aggregator, losses, grads = recording(module, patch, *called)

            recorded = []

            def three_steps(*args, **kwargs):
                # The first gradient step of the run is taken three times on its batch: the later ones aren't
                # recorded
                if recorded:
                    metrics = module_train(*args, **kwargs)
                    del grads[recorded[0] :]
                    return metrics
                for _ in range(3):
                    metrics = module_train(*args, **kwargs)
                    for name, value in metrics.items():
                        aggregator.update(name, value)
                recorded.append(len(grads))
                return metrics

            patch.setattr(module, "train", three_steps)
            try:
                run_p2e([*args, f"algo.compile.enabled={enabled}"], root_dir)
            finally:
                shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
        assert (len(compile_utils._COMPILED) > 0) == enabled
        results.append((losses, grads))
    assert_same_step(*results)
