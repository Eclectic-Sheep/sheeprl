import os
import shutil
import sys
import time
import warnings
from contextlib import nullcontext
from unittest import mock

import pytest

from sheeprl import ROOT_DIR
from sheeprl.cli import run
from sheeprl.utils.imports import _IS_WINDOWS


@pytest.fixture(params=["1", "2"])
def devices(request):
    return request.param


@pytest.fixture()
def standard_args():
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
        "dry_run=True",
        "checkpoint.save_last=False",
        "env.num_envs=1",
        f"env.sync_env={_IS_WINDOWS}",
        "env.capture_video=False",
        "fabric.devices=auto",
        "fabric.accelerator=cpu",
        "fabric.precision=bf16-true",
        "metric.log_level=0",
        "metric.disable_timer=True",
    ]
    if os.environ.get("MLFLOW_TRACKING_URI", None) is not None:
        args.extend(["logger@metric.logger=mlflow", "model_manager.disabled=False", "metric.log_level=1"])
    return args


@pytest.fixture()
def start_time():
    return str(int(time.time()))


@pytest.fixture(autouse=True)
def mock_env_and_destroy(devices):
    os.environ["LT_DEVICES"] = str(devices)
    if _IS_WINDOWS and devices != "1":
        pytest.skip()
    yield


def remove_test_dir(path: str) -> None:
    """Utility function to cleanup a temporary folder if it still exists."""
    try:
        shutil.rmtree(path, False, None)
    except (OSError, WindowsError):
        warnings.warn("Unable to delete folder {}.".format(path))


def test_droq(standard_args, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "droq", os.environ["LT_DEVICES"])
    run_name = "test_droq"
    args = standard_args + [
        "exp=droq",
        "algo.per_rank_batch_size=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


def test_sac(standard_args, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "sac", os.environ["LT_DEVICES"])
    run_name = "test_sac"
    args = standard_args + [
        "exp=sac",
        "algo.per_rank_batch_size=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


def test_sac_ae(standard_args, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "sac_ae", os.environ["LT_DEVICES"])
    run_name = "test_sac_ae"
    args = standard_args + [
        "exp=sac_ae",
        "algo.per_rank_batch_size=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.mlp_keys.encoder=[state]",
        "algo.cnn_keys.encoder=[rgb]",
        "env.screen_size=64",
        "algo.hidden_size=4",
        "algo.dense_units=4",
        "algo.cnn_channels_multiplier=2",
        "algo.actor.per_rank_update_freq=1",
        "algo.decoder.per_rank_update_freq=1",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


def test_sac_decoupled(standard_args, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "sac_decoupled", os.environ["LT_DEVICES"])
    run_name = "test_sac_decoupled"
    args = standard_args + [
        "exp=sac_decoupled",
        "algo.per_rank_batch_size=1",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        f"fabric.devices={os.environ['LT_DEVICES']}",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]

    with mock.patch.object(sys, "argv", args):
        with pytest.raises(RuntimeError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
            run()

    if os.environ["LT_DEVICES"] != "1":
        remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


def test_a2c(standard_args, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "ppo", os.environ["LT_DEVICES"])
    run_name = "test_ppo"
    args = standard_args + [
        "exp=a2c",
        f"algo.rollout_steps={os.environ['LT_DEVICES']}",
        "algo.per_rank_batch_size=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.cnn_keys.encoder=[]",
        "algo.mlp_keys.encoder=[state]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_ppo(standard_args, start_time, env_id):
    root_dir = os.path.join(f"pytest_{start_time}", "ppo", os.environ["LT_DEVICES"])
    run_name = "test_ppo"
    args = standard_args + [
        "exp=ppo",
        "env=dummy",
        f"algo.rollout_steps={os.environ['LT_DEVICES']}",
        "algo.per_rank_batch_size=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        f"env.id={env_id}",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_ppo_decoupled(standard_args, start_time, env_id):
    root_dir = os.path.join(f"pytest_{start_time}", "ppo_decoupled", os.environ["LT_DEVICES"])
    run_name = "test_ppo_decoupled"
    args = standard_args + [
        "exp=ppo_decoupled",
        "env=dummy",
        f"fabric.devices={os.environ['LT_DEVICES']}",
        f"algo.rollout_steps={os.environ['LT_DEVICES']}",
        "algo.per_rank_batch_size=1",
        "algo.update_epochs=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        f"env.id={env_id}",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
    ]

    with mock.patch.object(sys, "argv", args):
        with pytest.raises(RuntimeError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
            run()

    if os.environ["LT_DEVICES"] != "1":
        remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


def test_ppo_recurrent(standard_args, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "ppo_recurrent", os.environ["LT_DEVICES"])
    run_name = "test_ppo_recurrent"
    args = standard_args + [
        "exp=ppo_recurrent",
        "algo.rollout_steps=2",
        "algo.per_rank_batch_size=1",
        "algo.per_rank_sequence_length=2",
        "algo.update_epochs=2",
        "fabric.precision=32",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_dreamer_v1(standard_args, env_id, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "dreamer_v1", os.environ["LT_DEVICES"])
    run_name = "test_dreamer_v1"
    args = standard_args + [
        "exp=dreamer_v1",
        "env=dummy",
        "algo.per_rank_batch_size=1",
        "algo.per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=2",
        f"env.id={env_id}",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_p2e_dv1(standard_args, env_id, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "p2e_dv1", os.environ["LT_DEVICES"])
    run_name = "test_p2e_dv1"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len([d for d in os.listdir(ckpt_path) if "version" in d])
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=p2e_dv1_exploration",
        "env=dummy",
        "algo.per_rank_batch_size=2",
        "algo.per_rank_sequence_length=2",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=4",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=2",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=2",
        "algo.world_model.representation_model.hidden_size=2",
        "algo.world_model.transition_model.hidden_size=2",
        "buffer.checkpoint=True",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
        "checkpoint.save_last=True",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
        import torch.distributed

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            del os.environ["LOCAL_RANK"]
            del os.environ["NODE_RANK"]
            del os.environ["WORLD_SIZE"]
            del os.environ["MASTER_ADDR"]
            del os.environ["MASTER_PORT"]

    ckpt_path = os.path.join("logs", "runs", ckpt_path)
    checkpoints = os.listdir(ckpt_path)
    if len(checkpoints) > 0:
        ckpt_path = os.path.join(ckpt_path, checkpoints[-1])
    else:
        raise RuntimeError("No exploration checkpoints")
    args = standard_args + [
        "exp=p2e_dv1_finetuning",
        f"checkpoint.exploration_ckpt_path={ckpt_path}",
        "algo.per_rank_batch_size=2",
        "algo.per_rank_sequence_length=2",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=4",
        "env=dummy",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=2",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=2",
        "algo.world_model.representation_model.hidden_size=2",
        "algo.world_model.transition_model.hidden_size=2",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
    ]
    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_dreamer_v2(standard_args, env_id, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "dreamer_v2", os.environ["LT_DEVICES"])
    run_name = "test_dreamer_v2"
    args = standard_args + [
        "exp=dreamer_v2",
        "env=dummy",
        "algo.per_rank_batch_size=1",
        "algo.per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=8",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.world_model.representation_model.hidden_size=8",
        "algo.world_model.transition_model.hidden_size=8",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.per_rank_pretrain_steps=1",
        "algo.layer_norm=True",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_p2e_dv2(standard_args, env_id, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "p2e_dv2", os.environ["LT_DEVICES"])
    run_name = "test_p2e_dv2"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len([d for d in os.listdir(ckpt_path) if "version" in d])
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=p2e_dv2_exploration",
        "env=dummy",
        "algo.per_rank_batch_size=2",
        "algo.per_rank_sequence_length=2",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=4",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=2",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=2",
        "algo.world_model.representation_model.hidden_size=2",
        "algo.world_model.transition_model.hidden_size=2",
        "buffer.checkpoint=True",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
        "checkpoint.save_last=True",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
        import torch.distributed

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            del os.environ["LOCAL_RANK"]
            del os.environ["NODE_RANK"]
            del os.environ["WORLD_SIZE"]
            del os.environ["MASTER_ADDR"]
            del os.environ["MASTER_PORT"]

    ckpt_path = os.path.join("logs", "runs", ckpt_path)
    checkpoints = os.listdir(ckpt_path)
    if len(checkpoints) > 0:
        ckpt_path = os.path.join(ckpt_path, checkpoints[-1])
    else:
        raise RuntimeError("No exploration checkpoints")
    args = standard_args + [
        "exp=p2e_dv2_finetuning",
        f"checkpoint.exploration_ckpt_path={ckpt_path}",
        "algo.per_rank_batch_size=2",
        "algo.per_rank_sequence_length=2",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=4",
        "env=dummy",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=2",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=2",
        "algo.world_model.representation_model.hidden_size=2",
        "algo.world_model.transition_model.hidden_size=2",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
    ]
    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_dreamer_v3(standard_args, env_id, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "dreamer_v3", os.environ["LT_DEVICES"])
    run_name = "test_dreamer_v3"
    args = standard_args + [
        "exp=dreamer_v3",
        "env=dummy",
        "algo.per_rank_batch_size=1",
        "algo.per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=8",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.world_model.representation_model.hidden_size=8",
        "algo.world_model.transition_model.hidden_size=8",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
        "algo.mlp_layer_norm.cls=sheeprl.models.models.LayerNorm",
        "algo.cnn_layer_norm.cls=sheeprl.models.models.LayerNormChannelLast",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_p2e_dv3(standard_args, env_id, start_time):
    root_dir = os.path.join(f"pytest_{start_time}", "p2e_dv3", os.environ["LT_DEVICES"])
    run_name = "test_p2e_dv3"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len([d for d in os.listdir(ckpt_path) if "version" in d])
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "exp=p2e_dv3_exploration",
        "env=dummy",
        "algo.per_rank_batch_size=1",
        "algo.per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=8",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.world_model.representation_model.hidden_size=8",
        "algo.world_model.transition_model.hidden_size=8",
        "buffer.checkpoint=True",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
        "checkpoint.save_last=True",
        "algo.mlp_layer_norm.cls=sheeprl.models.models.LayerNorm",
        "algo.cnn_layer_norm.cls=sheeprl.models.models.LayerNormChannelLast",
    ]

    with mock.patch.object(sys, "argv", args):
        run()
        import torch.distributed

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            del os.environ["LOCAL_RANK"]
            del os.environ["NODE_RANK"]
            del os.environ["WORLD_SIZE"]
            del os.environ["MASTER_ADDR"]
            del os.environ["MASTER_PORT"]

    ckpt_path = os.path.join("logs", "runs", ckpt_path)
    checkpoints = os.listdir(ckpt_path)
    if len(checkpoints) > 0:
        ckpt_path = os.path.join(ckpt_path, checkpoints[-1])
    else:
        raise RuntimeError("No exploration checkpoints")
    args = standard_args + [
        "exp=p2e_dv3_finetuning",
        f"checkpoint.exploration_ckpt_path={ckpt_path}",
        "algo.per_rank_batch_size=1",
        "algo.per_rank_sequence_length=1",
        f"buffer.size={int(os.environ['LT_DEVICES'])}",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=8",
        "env=dummy",
        "env.id=" + env_id,
        f"root_dir={root_dir}",
        f"run_name={run_name}",
        "algo.dense_units=8",
        "algo.world_model.encoder.cnn_channels_multiplier=2",
        "algo.world_model.recurrent_model.recurrent_state_size=8",
        "algo.world_model.representation_model.hidden_size=8",
        "algo.world_model.transition_model.hidden_size=8",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",
        "algo.mlp_keys.encoder=[state]",
        "algo.mlp_keys.decoder=[state]",
        "algo.mlp_layer_norm.cls=sheeprl.models.models.LayerNorm",
        "algo.cnn_layer_norm.cls=sheeprl.models.models.LayerNormChannelLast",
    ]
    with mock.patch.object(sys, "argv", args):
        run()

    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))
