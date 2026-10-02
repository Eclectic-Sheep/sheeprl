"""The timers of the phases of the iterations, and the speeds of a run."""

import os
import shutil
import sys
from unittest import mock

import pytest

from sheeprl import ROOT_DIR
from sheeprl.utils.timer import timer

RUN_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "dry_run=True",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=1",
    "metric.disable_timer=False",
    "algo.run_test=False",
]


@pytest.mark.parametrize("exp", ["ppo", "sac"])
def test_the_phases_are_timed_by_every_process_on_its_own(exp, monkeypatch):
    # The training was timed with `metric.sync_on_compute`: with it, the times of all the processes were summed and
    # the training speed divided by their number
    from sheeprl.cli import run

    monkeypatch.setattr(timer, "timers", {})
    root_dir = f"pytest_timer_{exp}"
    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *RUN_ARGS,
        f"exp={exp}",
        "metric.sync_on_compute=True",
        f"root_dir={root_dir}",
    ]
    if exp == "sac":
        argv += ["env.id=Pendulum-v1", "algo.per_rank_batch_size=4", "algo.learning_starts=0"]
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(sys, "argv", argv):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert set(timer.timers) >= {"Time/train_time", "Time/env_interaction_time"}
    assert not any(t.sync_on_compute for t in timer.timers.values())


@pytest.mark.parametrize(
    "device,disabled,synchronized", [("cuda:0", False, True), ("cpu", False, False), ("cuda:0", True, False)]
)
def test_the_training_timer_waits_for_the_gpu(device, disabled, synchronized, monkeypatch):
    # The GPU runs the training after the CPU has launched it: the training timer stopped before it finished, and the
    # first copy to the CPU of the interaction that followed waited for it, timed with the interaction
    from sheeprl.utils.timer import training_timer

    monkeypatch.setattr(timer, "timers", {})
    monkeypatch.setattr(timer, "disabled", disabled)
    with mock.patch("torch.cuda.synchronize") as synchronize:
        with training_timer(device):
            pass
    assert synchronize.call_count == int(synchronized)


def logged_speeds(exp, args, root_dir, run_name="run", keep=False):
    """The speeds logged by a run of `exp` (a dry run, unless `args` says otherwise), with every phase timed 1
    second."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from sheeprl.cli import run

    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *RUN_ARGS,
        f"exp={exp}",
        *args,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]
    try:
        with (
            mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                timer, "compute", return_value={"Time/train_time": 1.0, "Time/env_interaction_time": 1.0}
            ),
        ):
            run()
        events = EventAccumulator(os.path.join("logs", "runs", root_dir, run_name, "version_0"))
        events.Reload()
        return {tag: [e.value for e in events.Scalars(tag)] for tag in ("Time/sps_train", "Time/sps_env_interaction")}
    finally:
        if not keep:
            shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)


@pytest.mark.parametrize(
    "exp,args,gradient_steps",
    [
        # 2 environments, 4 steps: 8 samples in 2 minibatches of 4, for 3 epochs
        (
            "ppo",
            ["algo.rollout_steps=4", "algo.per_rank_batch_size=4", "algo.update_epochs=3"],
            3 * 2,
        ),
        # 2 policy steps, replay ratio 2
        (
            "sac",
            ["env.id=Pendulum-v1", "algo.per_rank_batch_size=4", "algo.learning_starts=0", "algo.replay_ratio=2"],
            2 * 2,
        ),
    ],
)
def test_the_training_speed_counts_the_gradient_steps(exp, args, gradient_steps):
    # It counted the training iterations, whatever their gradient steps: its meaning changed with the replay ratio, the
    # number of environments, the epochs and the minibatches
    speeds = logged_speeds(exp, args, f"pytest_speed_{exp}")
    assert speeds["Time/sps_train"] == [gradient_steps]


def test_a_resumed_run_times_the_interaction_of_its_own_steps():
    # Its first interaction speed divided the policy steps since the last log of the run it resumes (also the ones
    # played before its checkpoint) by the time of its own steps. 2 environments, 4 steps: 8 policy steps an iteration,
    # a checkpoint every iteration, a log only at the end (the 3rd iteration)
    root_dir = "pytest_speed_resumed"
    args = [
        "dry_run=False",
        "algo.rollout_steps=4",
        "algo.per_rank_batch_size=4",
        "algo.total_steps=24",
        "metric.log_every=1000",
        "checkpoint.every=8",
    ]
    try:
        logged_speeds("ppo", args, root_dir, run_name="first", keep=True)
        ckpt_path = os.path.join("logs", "runs", root_dir, "first", "version_0", "checkpoint", "ckpt_8_0.ckpt")
        # Resumed after the first iteration: it plays 16 policy steps, timed 1 second
        speeds = logged_speeds("ppo", [*args, f"checkpoint.resume_from={ckpt_path}"], root_dir, run_name="resumed")
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert speeds["Time/sps_env_interaction"] == [16]
