import os
import shutil
import subprocess
import sys
import warnings
from unittest import mock

import pytest

from sheeprl import ROOT_DIR
from sheeprl.cli import run


def test_dp_strategy_str_warning():
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        "exp=ppo",
        "fabric.strategy=dp",
        "fabric.devices=1",
        "dry_run=True",
        "algo.rollout_steps=1",
        "metric.log_level=0",
    ]
    with mock.patch.object(sys, "argv", args):
        with pytest.warns(UserWarning) as record:
            run()
        assert len(record) >= 1
        assert (
            record[0].message.args[0] == "Running an algorithm with a strategy (dp) different "
            "than 'auto' or 'dpp' can cause unexpected problems. "
            "Please launch the script with a 'DDP' strategy with 'python sheeprl.py fabric.strategy=ddp' "
            "or the 'auto' one with 'python sheeprl.py fabric.strategy=auto' if you run into any problems."
        )


def test_module_not_found():
    args = [os.path.join(ROOT_DIR, "__main__.py"), "exp=ppo", "algo.name=not_found", "metric.log_level=0"]
    with mock.patch.object(sys, "argv", args):
        with pytest.raises(
            RuntimeError, match="Given the algorithm named 'not_found', no module has been found to be imported."
        ):
            run()


def test_dp_strategy_instance_warning():
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        "exp=test_decoupled_strategy_instance",
        "algo=ppo",
        "algo.rollout_steps=1",
        "metric.log_level=0",
    ]
    with mock.patch.object(sys, "argv", args):
        with pytest.warns(UserWarning) as record:
            run()
        assert len(record) >= 1
        assert (
            record[0].message.args[0] == "Running an algorithm with a strategy (DataParallelStrategy) "
            "different than 'SingleDeviceStrategy' or 'DDPStrategy' can cause unexpected problems. "
            "Please launch the script with a 'DDP' strategy with 'python sheeprl.py fabric.strategy=ddp' "
            "or with a single device with 'python sheeprl.py fabric.strategy=auto fabric.devices=1' "
            "if you run into any problems."
        )


def test_decoupled_strategy_instance_fail():
    args = [os.path.join(ROOT_DIR, "__main__.py"), "exp=test_decoupled_strategy_instance", "metric.log_level=0"]
    with pytest.raises(
        ValueError,
        match=r"\w+ is currently not supported for decoupled algorithms. "
        "Please launch the script with a 'DDP' strategy with 'python sheeprl.py fabric.strategy=ddp'",
    ):
        with mock.patch.object(sys, "argv", args):
            run()


def test_strategy_warning():
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        "exp=ppo",
        "fabric.strategy=dp",
        "fabric.devices=1",
        "dry_run=True",
        "algo.rollout_steps=1",
        "metric.log_level=0",
    ]
    with mock.patch.object(sys, "argv", args):
        with pytest.warns(UserWarning) as record:
            run()
        assert len(record) >= 1
        assert (
            record[0].message.args[0] == "Running an algorithm with a strategy (dp) "
            "different than 'auto' or 'dpp' can cause unexpected problems. "
            "Please launch the script with a 'DDP' strategy with 'python sheeprl.py fabric.strategy=ddp' "
            "or the 'auto' one with 'python sheeprl.py fabric.strategy=auto' if you run into any problems."
        )


def test_run_decoupled_algo():
    subprocess.run(
        sys.executable + " sheeprl.py exp=ppo_decoupled fabric.strategy=ddp fabric.devices=2 "
        "dry_run=True algo.rollout_steps=1 algo.cnn_keys.encoder=[rgb] algo.mlp_keys.encoder=[state] "
        "env.capture_video=False checkpoint.save_last=False metric.log_level=0 "
        "metric.disable_timer=True",
        shell=True,
        check=True,
    )


def test_run_algo():
    subprocess.run(
        sys.executable + " sheeprl.py exp=ppo dry_run=True algo.rollout_steps=1 "
        "algo.cnn_keys.encoder=[rgb] algo.mlp_keys.encoder=[state] "
        "env.capture_video=False checkpoint.save_last=False metric.log_level=0 "
        "metric.disable_timer=True",
        shell=True,
        check=True,
    )


def test_resume_from_checkpoint():
    root_dir = "pytest_test_ckpt"
    run_name = "test_ckpt"
    subprocess.run(
        sys.executable + " sheeprl.py exp=dreamer_v3 env=dummy dry_run=True "
        "env.capture_video=False algo.dense_units=8 algo.horizon=8 "
        "algo.cnn_keys.encoder=[rgb] algo.cnn_keys.decoder=[rgb] "
        "algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] "
        "algo.world_model.encoder.cnn_channels_multiplier=2 algo.replay_ratio=1 "
        "algo.world_model.recurrent_model.recurrent_state_size=8 "
        "algo.world_model.representation_model.hidden_size=8 algo.learning_starts=0 "
        "algo.world_model.transition_model.hidden_size=8 buffer.size=10 "
        "algo.per_rank_batch_size=1 algo.per_rank_sequence_length=1 "
        f"root_dir={root_dir} run_name={run_name} "
        "checkpoint.save_last=True metric.log_level=0 metric.disable_timer=True",
        shell=True,
        check=True,
    )

    ckpt_root = os.path.join("logs", "runs", root_dir, run_name)
    ckpt_dir = sorted([d for d in os.listdir(ckpt_root) if "version" in d])[-1]
    ckpt_path = os.path.join(ckpt_root, ckpt_dir, "checkpoint")
    ckpt_file_name = os.listdir(ckpt_path)[-1]
    ckpt_path = os.path.join(ckpt_path, ckpt_file_name)
    subprocess.run(
        sys.executable
        + f" sheeprl.py exp=dreamer_v3 env=dummy checkpoint.resume_from={ckpt_path} "
        + "root_dir=pytest_resume_ckpt run_name=test_resume metric.log_level=0 "
        + "algo.cnn_keys.encoder=[rgb] algo.cnn_keys.decoder=[rgb] "
        + "algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state]",
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


def test_resume_from_checkpoint_env_error():
    root_dir = "pytest_test_ckpt"
    run_name = "test_ckpt"
    subprocess.run(
        sys.executable + " sheeprl.py exp=dreamer_v3 env=dummy dry_run=True "
        "env.capture_video=False algo.dense_units=8 algo.horizon=8 "
        "algo.cnn_keys.encoder=[rgb] algo.cnn_keys.decoder=[rgb] "
        "algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] "
        "algo.world_model.encoder.cnn_channels_multiplier=2 algo.replay_ratio=1 "
        "algo.world_model.recurrent_model.recurrent_state_size=8 "
        "algo.world_model.representation_model.hidden_size=8 algo.learning_starts=0 "
        "algo.world_model.transition_model.hidden_size=8 buffer.size=10 "
        "algo.per_rank_batch_size=1 algo.per_rank_sequence_length=1 "
        f"root_dir={root_dir} run_name={run_name} "
        "checkpoint.save_last=True metric.log_level=0 metric.disable_timer=True",
        shell=True,
        check=True,
    )

    ckpt_root = os.path.join("logs", "runs", root_dir, run_name)
    ckpt_dir = sorted([d for d in os.listdir(ckpt_root) if "version" in d])[-1]
    ckpt_path = os.path.join(ckpt_root, ckpt_dir, "checkpoint")
    ckpt_file_name = os.listdir(ckpt_path)[-1]
    ckpt_path = os.path.join(ckpt_path, ckpt_file_name)
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        "exp=dreamer_v3",
        f"checkpoint.resume_from={ckpt_path}",
        "root_dir=pytest_resume_ckpt",
        "run_name=test_resume",
        "metric.log_level=0",
    ]
    with mock.patch.object(sys, "argv", args):
        with pytest.raises(
            ValueError,
            match=(
                "This experiment is run with a different environment "
                "from the one of the experiment you want to restart"
            ),
        ):
            run()

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


def test_resume_from_checkpoint_algo_error():
    root_dir = "pytest_test_ckpt"
    run_name = "test_ckpt"
    subprocess.run(
        sys.executable + " sheeprl.py exp=dreamer_v3 env=dummy dry_run=True "
        "env.capture_video=False algo.dense_units=8 algo.horizon=8 "
        "algo.cnn_keys.encoder=[rgb] algo.cnn_keys.decoder=[rgb] "
        "algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] "
        "algo.world_model.encoder.cnn_channels_multiplier=2 algo.replay_ratio=1 "
        "algo.world_model.recurrent_model.recurrent_state_size=8 "
        "algo.world_model.representation_model.hidden_size=8 algo.learning_starts=0 "
        "algo.world_model.transition_model.hidden_size=8 buffer.size=10 "
        "algo.per_rank_batch_size=1 algo.per_rank_sequence_length=1 "
        f"root_dir={root_dir} run_name={run_name} "
        "checkpoint.save_last=True metric.log_level=0 metric.disable_timer=True",
        shell=True,
        check=True,
    )

    ckpt_root = os.path.join("logs", "runs", root_dir, run_name)
    ckpt_dir = sorted([d for d in os.listdir(ckpt_root) if "version" in d])[-1]
    ckpt_path = os.path.join(ckpt_root, ckpt_dir, "checkpoint")
    ckpt_file_name = os.listdir(ckpt_path)[-1]
    ckpt_path = os.path.join(ckpt_path, ckpt_file_name)
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        "exp=ppo",
        "env=dummy",
        f"checkpoint.resume_from={ckpt_path}",
        "root_dir=pytest_resume_ckpt",
        "run_name=test_resume",
        "metric.log_level=0",
    ]
    print(args)
    with mock.patch.object(sys, "argv", args):
        with pytest.raises(
            ValueError,
            match=(
                "This experiment is run with a different algorithm "
                "from the one of the experiment you want to restart"
            ),
        ):
            run()

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


def test_evaluate():
    root_dir = "pytest_test_evaluate"
    run_name = "test_evaluate"
    subprocess.run(
        sys.executable + " sheeprl.py exp=dreamer_v3 env=dummy dry_run=True "
        "env.capture_video=False algo.dense_units=8 algo.horizon=8 "
        "algo.cnn_keys.encoder=[rgb] algo.cnn_keys.decoder=[rgb] "
        "algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] "
        "algo.world_model.encoder.cnn_channels_multiplier=2 algo.replay_ratio=1 "
        "algo.world_model.recurrent_model.recurrent_state_size=8 "
        "algo.world_model.representation_model.hidden_size=8 algo.learning_starts=0 "
        "algo.world_model.transition_model.hidden_size=8 buffer.size=10 "
        "algo.per_rank_batch_size=1 algo.per_rank_sequence_length=1 "
        f"root_dir={root_dir} run_name={run_name} "
        "checkpoint.save_last=True metric.log_level=0 metric.disable_timer=True",
        shell=True,
        check=True,
    )

    ckpt_root = os.path.join("logs", "runs", root_dir, run_name)
    ckpt_dir = sorted([d for d in os.listdir(ckpt_root) if "version" in d])[-1]
    ckpt_path = os.path.join(ckpt_root, ckpt_dir, "checkpoint")
    ckpt_file_name = os.listdir(ckpt_path)[-1]
    ckpt_path = os.path.join(ckpt_path, ckpt_file_name)
    try:
        subprocess.run(
            sys.executable + f" sheeprl_eval.py checkpoint_path={ckpt_path} env.capture_video=False",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.output)

    try:
        path = os.path.join("logs", "runs", "pytest_test_evaluate")
        shutil.rmtree(path)
    except (OSError, WindowsError):
        warnings.warn("Unable to delete folder {}.".format(path))
