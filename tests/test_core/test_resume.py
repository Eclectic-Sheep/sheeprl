"""A resumed run continues as the run it resumes. Off-policy: no random actions after `algo.learning_starts`, and the
training goes on at once when the replay buffer is in the checkpoint. Without it, the policy fills a new buffer first.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from typing import Dict, List, Tuple
from unittest import mock

import pytest
import torch

from sheeprl import ROOT_DIR
from sheeprl.algos.dreamer_v3.dreamer_v3 import DreamerV3
from sheeprl.algos.sac.sac import SAC
from sheeprl.core.runner import GymEnvironment

# 2 envs, 1 process: 2 policy steps per iteration; `learning_starts=4` gives 2 iterations of random actions, and the
# replay ratio 1 gives 2 gradient steps per iteration from the second one. The first run lasts 4 iterations, the
# resumed one 4 more
COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=0",
    "algo.run_test=False",
    "algo.learning_starts=4",
    "algo.replay_ratio=1",
    "checkpoint.save_last=True",
]

ALGORITHMS: Dict[str, Tuple[type, List[str]]] = {
    "sac": (SAC, ["exp=sac", "env.id=Pendulum-v1", "algo.per_rank_batch_size=4"]),
    "dreamer_v3": (
        DreamerV3,
        [
            "exp=dreamer_v3",
            "env=dummy",
            "env.id=discrete_dummy",
            "algo.cnn_keys.encoder=[rgb]",
            "algo.mlp_keys.encoder=[state]",
            "algo.dense_units=8",
            "algo.world_model.encoder.cnn_channels_multiplier=2",
            "algo.world_model.recurrent_model.recurrent_state_size=8",
            "algo.world_model.representation_model.hidden_size=8",
            "algo.world_model.transition_model.hidden_size=8",
            "algo.horizon=4",
            "algo.per_rank_batch_size=1",
            "algo.per_rank_sequence_length=1",
            "buffer.size=16",
        ],
    ),
}


def train(name: str, args: List[str]) -> Tuple[int, int]:
    """Run a training of `name` with `args` and return its random actions (calls, one per iteration) and its gradient
    steps."""
    from sheeprl.cli import run

    algo_cls, algo_args = ALGORITHMS[name]
    argv = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *algo_args, *args]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(
            GymEnvironment, "random_actions", autospec=True, side_effect=GymEnvironment.random_actions
        ) as random_actions,
        mock.patch.object(algo_cls, "train_step", autospec=True, side_effect=algo_cls.train_step) as train_step,
    ):
        run()
    return random_actions.call_count, train_step.call_count


@pytest.mark.parametrize("name", list(ALGORITHMS))
@pytest.mark.parametrize("buffer_checkpoint", [True, False])
def test_resumed_run_plays_no_random_actions_after_learning_starts(name, buffer_checkpoint):
    root_dir = f"pytest_resume_{name}_{buffer_checkpoint}"
    buffer_args = [f"buffer.checkpoint={buffer_checkpoint}", f"root_dir={root_dir}"]
    try:
        # Random actions in iterations 1 and 2; training from iteration 2
        assert train(name, ["algo.total_steps=8", "run_name=first", *buffer_args]) == (2, 3 * 2)
        (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, "first", "version_*", "checkpoint", "*.ckpt"))
        resumed = train(
            name, ["algo.total_steps=16", "run_name=resumed", f"checkpoint.resume_from={ckpt_path}", *buffer_args]
        )
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    if buffer_checkpoint:
        # The training goes on in iterations 5 to 8, as without the interruption
        assert resumed == (0, 4 * 2)
    else:
        # The policy fills a new buffer in iterations 5 and 6; training from iteration 6
        assert resumed == (0, 3 * 2)


@pytest.mark.parametrize("exp", ["ppo", "ppo_recurrent"])
def test_on_policy_run_resumes(exp):
    # The on-policy configs have no `algo.learning_starts`: the resume warning about it used to raise a `TypeError`.
    # The annealed values follow the `algo.total_steps` of the resumed run: the first run has 2 iterations, the
    # resumed one 4, so its last iteration uses 1/4 of the initial values (the restored scheduler kept the learning
    # rate at 0 after the 2 iterations of the first run)
    from sheeprl.cli import run

    root_dir = f"pytest_resume_{exp}"
    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *[a for a in COMMON_ARGS if "learning_starts" not in a and "replay_ratio" not in a],
        f"exp={exp}",
        *(["algo.per_rank_sequence_length=2"] if exp == "ppo_recurrent" else []),
        "algo.rollout_steps=4",
        "algo.per_rank_batch_size=4",
        "algo.anneal_lr=True",
        "algo.optimizer.lr=0.001",
        "algo.anneal_clip_coef=True",
        "algo.clip_coef=0.2",
        f"root_dir={root_dir}",
    ]
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}):
            with mock.patch.object(sys, "argv", [*argv, "algo.total_steps=16", "run_name=first"]):
                run()
            (ckpt_path,) = glob.glob(
                os.path.join("logs", "runs", root_dir, "first", "version_*", "checkpoint", "*.ckpt")
            )
            with mock.patch.object(
                sys, "argv", [*argv, "algo.total_steps=32", "run_name=resumed", f"checkpoint.resume_from={ckpt_path}"]
            ):
                run()
        resumed = glob.glob(os.path.join("logs", "runs", root_dir, "resumed", "version_*", "checkpoint", "*.ckpt"))
        assert [os.path.basename(p) for p in resumed] == ["ckpt_32_0.ckpt"]
        state = torch.load(resumed[0], weights_only=False)
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.001 / 4)
