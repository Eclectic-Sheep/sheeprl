"""A resumed run continues as the run it resumes. Off-policy: no random actions after `algo.learning_starts`, and the
training goes on at once when the replay buffer is in the checkpoint. Without it, the policy fills a new buffer first.
"""

from __future__ import annotations

import glob
import importlib
import os
import shutil
import sys
from typing import Any, Dict, List, Tuple
from unittest import mock

import gymnasium as gym
import pytest
import torch

from sheeprl import ROOT_DIR

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

DREAMER_ARGS = [
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
    "algo.per_rank_pretrain_steps=0",
    "buffer.size=16",
]

# For every algorithm: its module, the overrides of its training and the exploration it starts from, if any. The
# buffers hold the 8 steps of every environment: on Windows a memory-mapped buffer takes its whole size on the disk
# (SAC-AE's default one, of images, 12 GB)
ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "sac": {
        "module": "sheeprl.algos.sac.sac",
        "args": ["exp=sac", "env.id=Pendulum-v1", "algo.per_rank_batch_size=4", "buffer.size=16"],
    },
    "droq": {
        "module": "sheeprl.algos.droq.droq",
        "args": ["exp=droq", "env.id=Pendulum-v1", "algo.per_rank_batch_size=4", "buffer.size=16"],
    },
    "sac_ae": {
        "module": "sheeprl.algos.sac_ae.sac_ae",
        "args": [
            "exp=sac_ae",
            # A rendered environment with bounded continuous actions, whose renderer draws only shapes
            "env.id=MountainCarContinuous-v0",
            "env.frame_stack=1",
            "algo.cnn_keys.encoder=[rgb]",
            "algo.mlp_keys.encoder=[state]",
            "algo.hidden_size=8",
            "algo.dense_units=8",
            "algo.cnn_channels_multiplier=1",
            "algo.encoder.features_dim=8",
            "algo.per_rank_batch_size=4",
            "buffer.size=16",
        ],
    },
    "dreamer_v1": {"module": "sheeprl.algos.dreamer_v1.dreamer_v1", "args": ["exp=dreamer_v1", *DREAMER_ARGS]},
    "dreamer_v2": {"module": "sheeprl.algos.dreamer_v2.dreamer_v2", "args": ["exp=dreamer_v2", *DREAMER_ARGS]},
    "dreamer_v3": {"module": "sheeprl.algos.dreamer_v3.dreamer_v3", "args": ["exp=dreamer_v3", *DREAMER_ARGS]},
    "p2e_dv3_exploration": {
        "module": "sheeprl.algos.p2e_dv3.p2e_dv3_exploration",
        "args": ["exp=p2e_dv3_exploration", *DREAMER_ARGS],
    },
    "p2e_dv3_finetuning": {
        "module": "sheeprl.algos.p2e_dv3.p2e_dv3_finetuning",
        "args": ["exp=p2e_dv3_finetuning", *DREAMER_ARGS],
        "exploration": "p2e_dv3_exploration",
    },
}


def train(name: str, args: List[str]) -> Tuple[int, int]:
    """Run a training of `name` with `args` and return its random actions (calls, one per iteration) and its gradient
    steps."""
    from sheeprl.cli import run

    module = importlib.import_module(ALGORITHMS[name]["module"])
    gradient_steps = []

    def counting_train(*args, **kwargs):
        # DroQ's `train` does all the gradient steps of an iteration, the others one each
        gradient_steps.append(args[-1] if name == "droq" else 1)
        return module_train(*args, **kwargs)

    module_train = module.train
    argv = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *ALGORITHMS[name]["args"], *args]
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(module, "train", counting_train),
        # The random actions of the vectorized environments: Pendulum's are a `Box`, the dummy environment's a
        # `MultiDiscrete`
        mock.patch.object(gym.spaces.Box, "sample", autospec=True, side_effect=gym.spaces.Box.sample) as box_sample,
        mock.patch.object(
            gym.spaces.MultiDiscrete, "sample", autospec=True, side_effect=gym.spaces.MultiDiscrete.sample
        ) as multidiscrete_sample,
    ):
        run()
    return box_sample.call_count + multidiscrete_sample.call_count, sum(gradient_steps)


def checkpoint_of(root_dir: str, run_name: str) -> str:
    (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, run_name, "version_*", "checkpoint", "*.ckpt"))
    return ckpt_path


@pytest.mark.parametrize("name", list(ALGORITHMS))
@pytest.mark.parametrize("buffer_checkpoint", [True, False])
def test_resumed_run_plays_no_random_actions_after_learning_starts(name, buffer_checkpoint):
    root_dir = f"pytest_resume_{name}_{buffer_checkpoint}"
    buffer_args = [f"buffer.checkpoint={buffer_checkpoint}", f"root_dir={root_dir}"]
    try:
        if "exploration" in ALGORITHMS[name]:
            # The finetuning plays no random actions: it waits `learning_starts` steps before training, in the buffer
            # of its exploration
            exploration = ALGORITHMS[name]["exploration"]
            train(exploration, ["algo.total_steps=8", "run_name=exploration", "buffer.checkpoint=True", *buffer_args])
            buffer_args.append(f"checkpoint.exploration_ckpt_path={checkpoint_of(root_dir, 'exploration')}")
            assert train(name, ["algo.total_steps=8", "run_name=first", *buffer_args]) == (0, 3 * 2)
        else:
            # Random actions in iterations 1 and 2; training from iteration 2
            assert train(name, ["algo.total_steps=8", "run_name=first", *buffer_args]) == (2, 3 * 2)
        resumed = train(
            name,
            [
                "algo.total_steps=16",
                "run_name=resumed",
                f"checkpoint.resume_from={checkpoint_of(root_dir, 'first')}",
                *buffer_args,
            ],
        )
        resumed_checkpoint = torch.load(checkpoint_of(root_dir, "resumed"), weights_only=False)
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    if buffer_checkpoint:
        # The training goes on in iterations 5 to 8, as without the interruption
        assert resumed == (0, 4 * 2)
    else:
        # The policy fills a new buffer in iterations 5 and 6; training from iteration 6
        assert resumed == (0, 3 * 2)
    # The gradient steps of the run count the ones of the run it resumes
    assert resumed_checkpoint["per_rank_gradient_steps"] == 3 * 2 + resumed[1]


def test_a_resumed_dreamer_v3_updates_its_target_critic_with_tau():
    # The first gradient step of a run copies the critic into the target critic: a resumed run started counting its
    # gradient steps from 0 again, so it copied it again, discarding the slow critic of the checkpoint
    from sheeprl.algos.dreamer_v3 import dreamer_v3

    root_dir = "pytest_resume_dreamer_v3_target_critic"
    args = ["buffer.checkpoint=True", f"root_dir={root_dir}"]
    critics = []

    def first_train(*args, **kwargs):
        # The critic and the target critic of the first gradient step
        if len(critics) == 0:
            critics.append({k: v.clone() for k, v in args[3].module.state_dict().items()})
            critics.append({k: v.clone() for k, v in args[4].state_dict().items()})
        return dreamer_v3_train(*args, **kwargs)

    dreamer_v3_train = dreamer_v3.train
    try:
        train("dreamer_v3", ["algo.total_steps=8", "run_name=first", *args])
        ckpt_path = checkpoint_of(root_dir, "first")
        saved = torch.load(ckpt_path, weights_only=False)
        with mock.patch.object(dreamer_v3, "train", first_train):
            train(
                "dreamer_v3", ["algo.total_steps=16", "run_name=resumed", f"checkpoint.resume_from={ckpt_path}", *args]
            )
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    critic, target_critic = critics
    tau = 0.02
    for k, v in saved["critic"].items():
        assert torch.equal(critic[k], v), k
        assert torch.allclose(target_critic[k], tau * v + (1 - tau) * saved["target_critic"][k]), k
    assert any(not torch.equal(target_critic[k], critic[k]) for k in critic)


@pytest.mark.parametrize("exp", ["ppo", "a2c", "ppo_recurrent"])
def test_on_policy_run_resumes(exp):
    # The on-policy configs have no `algo.learning_starts`: the resume warning about it used to raise a `TypeError`
    from sheeprl.cli import run

    root_dir = f"pytest_resume_{exp}"
    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *[a for a in COMMON_ARGS if "learning_starts" not in a and "replay_ratio" not in a],
        f"exp={exp}",
        "algo.rollout_steps=4",
        "algo.per_rank_batch_size=4",
        f"root_dir={root_dir}",
    ]
    if exp == "ppo_recurrent":
        argv.append("algo.per_rank_sequence_length=2")
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}):
            with mock.patch.object(sys, "argv", [*argv, "algo.total_steps=16", "run_name=first"]):
                run()
            with mock.patch.object(
                sys,
                "argv",
                [
                    *argv,
                    "algo.total_steps=32",
                    "run_name=resumed",
                    f"checkpoint.resume_from={checkpoint_of(root_dir, 'first')}",
                ],
            ):
                run()
        resumed = glob.glob(os.path.join("logs", "runs", root_dir, "resumed", "version_*", "checkpoint", "*.ckpt"))
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert [os.path.basename(p) for p in resumed] == ["ckpt_32_0.ckpt"]
