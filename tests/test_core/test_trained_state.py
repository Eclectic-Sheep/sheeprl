"""The algorithms on the shared core are evaluated (`sheeprl-eval`) and registered (`sheeprl-registration`) with the
models of their training state, restored from a checkpoint by `sheeprl.core.load_trained_state`."""

from __future__ import annotations

import contextlib
import glob
import importlib
import os
import shutil
import sys
import types
from typing import Any, Dict, List
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf

from sheeprl import ROOT_DIR
from sheeprl.utils.utils import dotdict

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "dry_run=True",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=0",
    "metric.disable_timer=True",
    "checkpoint.save_last=True",
    "algo.run_test=False",
]

# For every algorithm: the overrides of a short training, and the models it registers
ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "args": ["exp=ppo", "algo.rollout_steps=4", "algo.per_rank_batch_size=4"],
        "models": ["agent"],
    },
}


def train(name: str, root_dir: str) -> str:
    """Train the algorithm `name` for one iteration and return the path of its checkpoint."""
    from sheeprl.cli import run

    args = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *ALGORITHMS[name]["args"], f"root_dir={root_dir}"]
    with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(sys, "argv", [*args, "run_name=run"]):
        run()
    (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, "run", "version_*", "checkpoint", "*.ckpt"))
    return ckpt_path


def assert_same_weights(module: torch.nn.Module, saved: Dict[str, torch.Tensor]) -> None:
    """The weights of `module` are the saved ones (the keys of `module` are a subset of the saved ones)."""
    weights = module.state_dict()
    assert len(weights) > 0 and set(weights) <= set(saved), set(weights) - set(saved)
    for k, v in weights.items():
        assert torch.equal(v.cpu(), saved[k].cpu()), k


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_evaluation_plays_the_trained_models(name):
    from sheeprl.cli import evaluation

    root_dir = f"pytest_trained_state_{name}"
    ckpt_path = train(name, root_dir)
    saved = torch.load(ckpt_path, weights_only=False)

    evaluate = importlib.import_module(f"sheeprl.algos.{name}.evaluate")
    played: List[torch.nn.Module] = []

    def test(policy, *args, **kwargs):
        played.append(policy)

    try:
        with (
            mock.patch.object(evaluate, "test", test),
            mock.patch.object(
                sys, "argv", ["sheeprl_eval.py", f"checkpoint_path={ckpt_path}", "env.capture_video=False"]
            ),
        ):
            evaluation()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    # The policy plays with the trained weights
    (policy,) = played
    assert_same_weights(policy, saved["agent"])


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_registration_logs_the_trained_models(name):
    from lightning import Fabric

    from sheeprl.utils.env import make_env

    root_dir = f"pytest_registered_state_{name}"
    ckpt_path = train(name, root_dir)
    # The configuration of the training, saved next to the checkpoints
    train_cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml")
    train_cfg = dotdict(OmegaConf.to_container(OmegaConf.load(train_cfg_path), resolve=True))
    saved = torch.load(ckpt_path, weights_only=False)

    # A stand-in for MLflow, which records the logged models
    logged: Dict[str, torch.nn.Module] = {}
    fake_mlflow = types.SimpleNamespace(
        start_run=lambda **kwargs: contextlib.nullcontext(),
        pytorch=types.SimpleNamespace(log_model=lambda model, artifact_path: logged.setdefault(artifact_path, model)),
        log_dict=lambda *args, **kwargs: None,
    )
    utils = importlib.import_module(f"sheeprl.algos.{name}.utils")
    cfg = dotdict({"run": {"id": None, "name": None}, "experiment": {"id": None}, "to_log": train_cfg, **train_cfg})
    fabric = Fabric(devices=1, accelerator="cpu")
    env = make_env(cfg, cfg.seed, 0, None, "test", vector_env_idx=0)()
    try:
        with (
            mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}),
            mock.patch.object(utils, "_IS_MLFLOW_AVAILABLE", True),
        ):
            utils.log_models_from_checkpoint(fabric, env, cfg, saved)
    finally:
        env.close()
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert sorted(logged) == sorted(ALGORITHMS[name]["models"])
    for k, model in logged.items():
        assert_same_weights(model, saved[k])
