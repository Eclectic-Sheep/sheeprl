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


def entries(saved: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """The saved weights whose key starts with `prefix`, without it."""
    return {k[len(prefix) :]: v for k, v in saved.items() if k.startswith(prefix)}


# The overrides of a short training of the Dreamer-like algorithms, on the dummy environment
DREAMER_ARGS = [
    "env=dummy",
    "env.id=discrete_dummy",
    "algo.cnn_keys.encoder=[rgb]",
    "algo.cnn_keys.decoder=[rgb]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
    "algo.dense_units=8",
    "algo.world_model.encoder.cnn_channels_multiplier=2",
    "algo.world_model.recurrent_model.recurrent_state_size=8",
    "algo.world_model.representation_model.hidden_size=8",
    "algo.world_model.transition_model.hidden_size=8",
    "algo.horizon=4",
    "algo.per_rank_batch_size=1",
    "algo.per_rank_sequence_length=1",
    "algo.learning_starts=0",
    "buffer.size=10",
]


def dreamer_policy(saved: Dict[str, torch.Tensor], actor: str) -> Dict[str, torch.Tensor]:
    """The weights of a Dreamer player: the encoder and the RSSM of the world model, and the actor `actor`."""
    return {
        **{k: v for k, v in saved["world_model"].items() if k.startswith(("encoder.", "rssm."))},
        **{"actor." + k: v for k, v in saved[actor].items()},
    }


def dreamer_v2_policy(saved: Dict[str, torch.Tensor], actor: str) -> Dict[str, torch.Tensor]:
    """The weights of a DreamerV2 (or DreamerV1) player: the encoder, the recurrent and representation models of the
    world model, and the actor `actor`."""
    world_model = saved["world_model"]
    return {
        **{k: v for k, v in world_model.items() if k.startswith("encoder.")},
        **{k[len("rssm.") :]: v for k, v in world_model.items() if k.startswith("rssm.recurrent_model.")},
        **{k[len("rssm.") :]: v for k, v in world_model.items() if k.startswith("rssm.representation_model.")},
        **{"actor." + k: v for k, v in saved[actor].items()},
    }


def p2e_saved_model(saved: Dict[str, Any], name: str) -> Dict[str, torch.Tensor]:
    """The saved weights of the P2E model `name`: the exploration critics are saved together, by critic."""
    for prefix, key in (("critic_exploration_", "module"), ("target_critic_exploration_", "target_module")):
        if name.startswith(prefix):
            return saved["critics_exploration"][name[len(prefix) :]][key]
    return saved[name]


P2E_TASK_MODELS = ["world_model", "actor_task", "critic_task", "target_critic_task", "moments_task"]

# For every algorithm: the overrides of a short training, the weights its policy plays with (from the checkpoint) and
# the models it registers. Optionally: its module (`module`, default the name), the saved weights of a registered model
# (`saved_model`, default the entry of the checkpoint named after it) and the exploration it starts from
# (`exploration`, an algorithm trained before it)
ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "args": ["exp=ppo", "algo.rollout_steps=4", "algo.per_rank_batch_size=4"],
        "policy": lambda saved: saved["agent"],
        "models": ["agent"],
    },
    "a2c": {
        "args": ["exp=a2c", "algo.rollout_steps=4"],
        "policy": lambda saved: saved["agent"],
        "models": ["agent"],
    },
    "ppo_recurrent": {
        "args": ["exp=ppo_recurrent", "algo.rollout_steps=4", "algo.per_rank_sequence_length=2"],
        "policy": lambda saved: saved["agent"],
        "models": ["agent"],
    },
    "sac": {
        "args": ["exp=sac", "env.id=Pendulum-v1", "algo.per_rank_batch_size=4", "algo.learning_starts=0"],
        # The policy is made of the modules of the actor
        "policy": lambda saved: entries(saved["agent"], "_actor."),
        "models": ["agent"],
    },
    "droq": {
        "args": ["exp=droq", "env.id=Pendulum-v1", "algo.per_rank_batch_size=4", "algo.learning_starts=0"],
        # The policy is made of the modules of the actor, as for SAC
        "policy": lambda saved: entries(saved["agent"], "_actor."),
        "models": ["agent"],
    },
    "sac_ae": {
        "args": [
            "exp=sac_ae",
            # A rendered environment with bounded continuous actions (those of the dummy environments are unbounded)
            # whose renderer draws only shapes: Pendulum loads an image, which needs the PNG support of pygame
            "env.id=MountainCarContinuous-v0",
            "env.frame_stack=1",
            "algo.cnn_keys.encoder=[rgb]",
            "algo.mlp_keys.encoder=[state]",
            "algo.hidden_size=8",
            "algo.dense_units=8",
            "algo.cnn_channels_multiplier=1",
            "algo.encoder.features_dim=8",
            "algo.per_rank_batch_size=4",
            "algo.learning_starts=0",
        ],
        # The policy is made of the modules of the actor (its encoder shares the layers of the critics' one)
        "policy": lambda saved: entries(saved["agent"], "_actor."),
        "models": ["agent", "encoder", "decoder"],
    },
    "dreamer_v3": {
        "args": ["exp=dreamer_v3", *DREAMER_ARGS],
        "policy": lambda saved: dreamer_policy(saved, "actor"),
        "models": ["world_model", "actor", "critic", "target_critic", "moments"],
    },
    "dreamer_v1": {
        "args": ["exp=dreamer_v1", *DREAMER_ARGS],
        "policy": lambda saved: dreamer_v2_policy(saved, "actor"),
        "models": ["world_model", "actor", "critic"],
    },
    "p2e_dv1_exploration": {
        "module": "p2e_dv1",
        "args": ["exp=p2e_dv1_exploration", *DREAMER_ARGS, "algo.ensembles.n=2"],
        # The evaluation plays the task actor
        "policy": lambda saved: dreamer_v2_policy(saved, "actor_task"),
        "models": ["world_model", "actor_task", "critic_task", "ensembles", "actor_exploration", "critic_exploration"],
    },
    "p2e_dv1_finetuning": {
        "module": "p2e_dv1",
        "exploration": "p2e_dv1_exploration",
        "args": ["exp=p2e_dv1_finetuning", *DREAMER_ARGS],
        "policy": lambda saved: dreamer_v2_policy(saved, "actor_task"),
        "models": ["world_model", "actor_task", "critic_task"],
    },
    "dreamer_v2": {
        "args": ["exp=dreamer_v2", *DREAMER_ARGS],
        "policy": lambda saved: dreamer_v2_policy(saved, "actor"),
        "models": ["world_model", "actor", "critic", "target_critic"],
    },
    "p2e_dv2_exploration": {
        "module": "p2e_dv2",
        "args": ["exp=p2e_dv2_exploration", *DREAMER_ARGS, "algo.ensembles.n=2"],
        # The evaluation plays the task actor
        "policy": lambda saved: dreamer_v2_policy(saved, "actor_task"),
        "models": [
            "world_model",
            "actor_task",
            "critic_task",
            "target_critic_task",
            "ensembles",
            "actor_exploration",
            "critic_exploration",
            "target_critic_exploration",
        ],
    },
    "p2e_dv2_finetuning": {
        "module": "p2e_dv2",
        "exploration": "p2e_dv2_exploration",
        "args": ["exp=p2e_dv2_finetuning", *DREAMER_ARGS],
        "policy": lambda saved: dreamer_v2_policy(saved, "actor_task"),
        "models": ["world_model", "actor_task", "critic_task", "target_critic_task"],
    },
    "p2e_dv3_exploration": {
        "module": "p2e_dv3",
        "args": ["exp=p2e_dv3_exploration", *DREAMER_ARGS],
        # The evaluation plays the task actor
        "policy": lambda saved: dreamer_policy(saved, "actor_task"),
        "models": [
            *P2E_TASK_MODELS,
            "ensembles",
            "actor_exploration",
            *[
                f"{m}_exploration_{k}"
                for m in ("critic", "target_critic", "moments")
                for k in ("intrinsic", "extrinsic")
            ],
        ],
        "saved_model": p2e_saved_model,
    },
    "p2e_dv3_finetuning": {
        "module": "p2e_dv3",
        "exploration": "p2e_dv3_exploration",
        "args": ["exp=p2e_dv3_finetuning", *DREAMER_ARGS],
        "policy": lambda saved: dreamer_policy(saved, "actor_task"),
        "models": P2E_TASK_MODELS,
    },
}


def train(name: str, root_dir: str, run_name: str = "run") -> str:
    """Train the algorithm `name` for one iteration (after its exploration, if any) and return the path of its
    checkpoint."""
    from sheeprl.cli import run

    args = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *ALGORITHMS[name]["args"], f"root_dir={root_dir}"]
    if "exploration" in ALGORITHMS[name]:
        exploration_ckpt_path = train(ALGORITHMS[name]["exploration"], root_dir, run_name="exploration")
        args.append(f"checkpoint.exploration_ckpt_path={exploration_ckpt_path}")
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", [*args, f"run_name={run_name}"]),
    ):
        run()
    (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, run_name, "version_*", "checkpoint", "*.ckpt"))
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

    evaluate = importlib.import_module(f"sheeprl.algos.{ALGORITHMS[name].get('module', name)}.evaluate")
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
    assert_same_weights(policy, ALGORITHMS[name]["policy"](saved))


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

    def log_model(model, name, serialization_format):
        # MLflow 3 traces the models by default (`pt2`): the agents are saved as they are
        assert serialization_format == "pickle"
        logged.setdefault(name, model)

    fake_mlflow = types.SimpleNamespace(
        start_run=lambda **kwargs: contextlib.nullcontext(),
        pytorch=types.SimpleNamespace(log_model=log_model),
        log_dict=lambda *args, **kwargs: None,
    )
    utils = importlib.import_module(f"sheeprl.algos.{ALGORITHMS[name].get('module', name)}.utils")
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
    saved_model = ALGORITHMS[name].get("saved_model", lambda saved, k: saved[k])
    for k, model in logged.items():
        assert_same_weights(model, saved_model(saved, k))
