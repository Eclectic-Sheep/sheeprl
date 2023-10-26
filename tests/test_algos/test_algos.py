import os
import shutil
import sys
import time
import warnings
from contextlib import nullcontext
from unittest import mock

import pytest
import torch.distributed as dist

from sheeprl import ROOT_DIR
from sheeprl.cli import run
from sheeprl.utils.imports import _IS_WINDOWS


@pytest.fixture(params=["1", "2"])
def devices(request):
    return request.param


@pytest.fixture()
def standard_args():
    return [
        os.path.join(ROOT_DIR, "__main__.py"),
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
        "dry_run=True",
        "checkpoint.save_last=False",
        "metric.log_level=0",
        "metric.disable_timer=True",
        "env.num_envs=1",
        "fabric.devices=auto",
        f"env.sync_env={_IS_WINDOWS}",
        "env.capture_video=False",
    ]


@pytest.fixture()
def start_time():
    return str(int(time.time()))


@pytest.fixture(autouse=True)
def mock_env_and_destroy(devices):
    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(devices)}, clear=False) as _fixture:
        if _IS_WINDOWS and devices != "1":
            pytest.skip()
        yield _fixture
    if dist.is_initialized():
        dist.destroy_process_group()


def remove_test_dir(path: str) -> None:
    """Utility function to cleanup a temporary folder if it still exists."""
    try:
        shutil.rmtree(path, False, None)
    except (OSError, WindowsError):
        warnings.warn("Unable to delete folder {}.".format(path))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_droq(standard_args, checkpoint_buffer, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "droq", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=droq",
        "per_rank_batch_size=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        f"buffer.checkpoint={checkpoint_buffer}",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_sac(standard_args, checkpoint_buffer, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "sac", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=sac",
        "per_rank_batch_size=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        f"buffer.checkpoint={checkpoint_buffer}",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_sac_ae(standard_args, checkpoint_buffer, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "sac_ae", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=sac_ae",
        "per_rank_batch_size=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "mlp_keys.encoder=[state]",
        "cnn_keys.encoder=[rgb]",
        "env.screen_size=64",
        "algo.hidden_size=4",
        "algo.dense_units=4",
        "algo.cnn_channels_multiplier=2",
        "algo.actor.network_frequency=1",
        "algo.decoder.update_freq=1",
        f"buffer.checkpoint={checkpoint_buffer}",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_sac_decoupled(standard_args, checkpoint_buffer, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "sac_decoupled", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=sac_decoupled",
        "per_rank_batch_size=1",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        f"fabric.devices={os.environ['LT_DEVICES']}",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        f"buffer.checkpoint={checkpoint_buffer}",
    ]

    with mock.patch.object(sys, "argv", args):
        with pytest.raises(RuntimeError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
            run()

    if os.environ["LT_DEVICES"] != "1":
        remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_ppo(standard_args, start_time, env_id):
    root_dir = os.path.join(f"pytest_{start_time}", "ppo", os.environ["LT_DEVICES"])
    run_name = "test_ppo"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=ppo",
        "env=dummy",
        f"algo.rollout_steps={os.environ['LT_DEVICES']}",
        "per_rank_batch_size=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        f"env.id={env_id}",
        "cnn_keys.encoder=[rgb]",
        "mlp_keys.encoder=[]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_ppo_decoupled(standard_args, start_time, env_id):
    root_dir = os.path.join(f"pytest_{start_time}", "ppo_decoupled", os.environ["LT_DEVICES"])
    run_name = "test_ppo_decoupled"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=ppo_decoupled",
        "env=dummy",
        f"fabric.devices={os.environ['LT_DEVICES']}",
        f"algo.rollout_steps={os.environ['LT_DEVICES']}",
        "per_rank_batch_size=1",
        "algo.update_epochs=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        f"env.id={env_id}",
        "cnn_keys.encoder=[rgb]",
        "mlp_keys.encoder=[]",
    ]

    with mock.patch.object(sys, "argv", args):
        with pytest.raises(RuntimeError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
            run()

    if os.environ["LT_DEVICES"] != "1":
        remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
def test_ppo_recurrent(standard_args, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "ppo_recurrent", os.environ["LT_DEVICES"])
    run_name = "test_ppo_recurrent"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=ppo_recurrent",
        "algo.rollout_steps=2",
        "per_rank_batch_size=1",
        "per_rank_sequence_length=2",
        "algo.update_epochs=2",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_dreamer_v1(standard_args, env_id, checkpoint_buffer, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "dreamer_v1", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=dreamer_v1",
        "env=dummy",
        "per_rank_batch_size=1",
        "per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        "algo.horizon=2",
        f"env.id={env_id}",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        f"buffer.checkpoint={checkpoint_buffer}",
        "cnn_keys.encoder=[rgb]",
        "cnn_keys.decoder=[rgb]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_p2e_dv1(standard_args, env_id, checkpoint_buffer, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "p2e_dv1", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=p2e_dv1",
        "env=dummy",
        "per_rank_batch_size=2",
        "per_rank_sequence_length=2",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        "algo.horizon=8",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        f"buffer.checkpoint={checkpoint_buffer}",
        "cnn_keys.encoder=[rgb]",
        "cnn_keys.decoder=[rgb]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_p2e_dv2(standard_args, env_id, checkpoint_buffer, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "p2e_dv2", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=p2e_dv2",
        "env=dummy",
        "per_rank_batch_size=2",
        "per_rank_sequence_length=2",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        "algo.horizon=2",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.world_model.representation_model.hidden_size=8",
        "algo.world_model.transition_model.hidden_size=8",
        "cnn_keys.encoder=[rgb]",
        "algo.per_rank_pretrain_steps=1",
        f"buffer.checkpoint={checkpoint_buffer}",
        "cnn_keys.encoder=[rgb]",
        "cnn_keys.decoder=[rgb]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_dreamer_v2(standard_args, env_id, checkpoint_buffer, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "dreamer_v2", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=dreamer_v2",
        "env=dummy",
        "per_rank_batch_size=1",
        "per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        "algo.horizon=8",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.world_model.representation_model.hidden_size=8",
        "algo.world_model.transition_model.hidden_size=8",
        "cnn_keys.encoder=[rgb]",
        "algo.per_rank_pretrain_steps=1",
        "algo.layer_norm=True",
        f"buffer.checkpoint={checkpoint_buffer}",
        "cnn_keys.encoder=[rgb]",
        "cnn_keys.decoder=[rgb]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_dreamer_v3(standard_args, env_id, checkpoint_buffer, start_time):
    root_dir = os.path.join("pytest_" + start_time, "dreamer_v3", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=dreamer_v3",
        "env=dummy",
        "per_rank_batch_size=1",
        "per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        "algo.horizon=8",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.world_model.representation_model.hidden_size=8",
        "algo.world_model.transition_model.hidden_size=8",
        "cnn_keys.encoder=[rgb]",
        "algo.layer_norm=True",
        "algo.train_every=1",
        f"buffer.checkpoint={checkpoint_buffer}",
        "cnn_keys.encoder=[rgb]",
        "cnn_keys.decoder=[rgb]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_p2e_dv3(standard_args, env_id, checkpoint_buffer, start_time):
    root_dir = os.path.join("pytest_" + start_time, "p2e_dv3", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=p2e_dv3",
        "env=dummy",
        "per_rank_batch_size=1",
        "per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.per_rank_gradient_steps=1",
        "algo.horizon=8",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.world_model.representation_model.hidden_size=8",
        "algo.world_model.transition_model.hidden_size=8",
        "algo.layer_norm=True",
        "algo.train_every=1",
        f"buffer.checkpoint={checkpoint_buffer}",
        "cnn_keys.encoder=[rgb]",
        "cnn_keys.decoder=[rgb]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()

    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))
