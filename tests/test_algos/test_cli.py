import os
import shutil
import subprocess
import sys
import warnings

import pytest


def test_fsdp_strategy_fail():
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            "lightning run model --strategy=fsdp --devices=1 sheeprl.py ppo",
            shell=True,
            check=True,
        )


def test_run_decoupled_algo():
    subprocess.run(
        "lightning run model --strategy=ddp --devices=2 sheeprl.py ppo_decoupled "
        "exp=ppo dry_run=True rollout_steps=1 cnn_keys.encoder=[rgb] mlp_keys.encoder=[state] "
        "env.capture_video=False",
        shell=True,
        check=True,
    )


def test_run_algo():
    subprocess.run(
        sys.executable
        + " sheeprl.py ppo exp=ppo dry_run=True rollout_steps=1 cnn_keys.encoder=[rgb] mlp_keys.encoder=[state] "
        "env.capture_video=False",
        shell=True,
        check=True,
    )


def test_resume_from_checkpoint():
    root_dir = "pytest_test_ckpt"
    run_name = "test_ckpt"
    subprocess.run(
        sys.executable
        + " sheeprl.py dreamer_v3 exp=dreamer_v3 env=dummy dry_run=True "
        + "env.capture_video=False algo.dense_units=8 algo.horizon=8 "
        + "algo.world_model.encoder.cnn_channels_multiplier=2 gradient_steps=1 "
        + "algo.world_model.recurrent_model.recurrent_state_size=8 "
        + "algo.world_model.representation_model.hidden_size=8 learning_starts=0 "
        + "algo.world_model.transition_model.hidden_size=8 buffer.size=10 "
        + "algo.layer_norm=True per_rank_batch_size=1 per_rank_sequence_length=1 "
        + f"train_every=1 root_dir={root_dir} run_name={run_name}",
        shell=True,
        check=True,
    )

    ckpt_path = os.path.join("logs", "runs", root_dir, run_name, "version_0", "checkpoint")
    ckpt_file_name = os.listdir(ckpt_path)[-1]
    ckpt_path = os.path.join(ckpt_path, ckpt_file_name)
    subprocess.run(
        sys.executable
        + f" sheeprl.py dreamer_v3 exp=dreamer_v3 checkpoint.resume_from={ckpt_path} "
        + "root_dir=pytest_resume_ckpt run_name=test_resume",
        shell=True,
        check=True,
    )

    try:
        path = os.path.join("logs", "runs", "pytest_test_ckpt")
        shutil.rmtree(path)
    except (OSError, WindowsError):
        warnings.warn("Unable to delete folder {}.".format(path))
    try:
        path = os.path.join("logs", "runs", "pytest_resume_ckpt")
        shutil.rmtree(path)
    except (OSError, WindowsError):
        warnings.warn("Unable to delete folder {}.".format(path))
