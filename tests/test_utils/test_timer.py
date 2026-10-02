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
