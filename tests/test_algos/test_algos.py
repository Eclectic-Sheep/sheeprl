import os
import shutil
import sys
import time
import warnings
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
        "env.num_envs=2",
        "env.sync_env=True",
        "env.capture_video=False",
        "fabric.devices=auto",
        "fabric.accelerator=cpu",
        # The CPU bf16 matmul of torch crashes with an illegal instruction (0xc000001d) on part of the Windows
        # runners of GitHub Actions, depending on the host CPU: Windows runs in fp32, Linux keeps testing bf16
        f"fabric.precision={'32-true' if _IS_WINDOWS else 'bf16-true'}",
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
    except OSError:
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


def test_droq_actor_batch_size(standard_args, start_time):
    # The actor and the entropy coefficient must be updated with a whole minibatch on every rank.
    # The policy loss is checked on the rank-0 process, which is the one running the test
    from sheeprl.algos.droq import droq

    batch_sizes = []

    def policy_loss(alpha, logprobs, qf_values):
        batch_sizes.append(logprobs.shape[0])
        return droq_policy_loss(alpha, logprobs, qf_values)

    droq_policy_loss = droq.policy_loss
    root_dir = os.path.join(f"pytest_{start_time}", "droq", os.environ["LT_DEVICES"])
    run_name = "test_droq_actor_batch_size"
    args = standard_args + [
        "exp=droq",
        "algo.per_rank_batch_size=4",
        "buffer.size=8",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]

    with mock.patch.object(droq, "policy_loss", policy_loss), mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))
    assert len(batch_sizes) > 0 and all(batch_size == 4 for batch_size in batch_sizes)


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


def test_sac_gradient_steps(standard_args, start_time):
    # Every iteration plays `num_envs * world_size` policy steps, so with a replay ratio of 1 every rank
    # must perform `num_envs` gradient steps. They are counted on the rank-0 process, which runs the test
    from sheeprl.algos.sac import sac

    gradient_steps = []
    sac_train_step = sac.SAC.train_step

    def train_step(*args, **kwargs):
        gradient_steps.append(1)
        return sac_train_step(*args, **kwargs)

    root_dir = os.path.join(f"pytest_{start_time}", "sac", os.environ["LT_DEVICES"])
    run_name = "test_sac_gradient_steps"
    args = standard_args + [
        "exp=sac",
        "algo.per_rank_batch_size=1",
        "buffer.size=8",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]

    with mock.patch.object(sac.SAC, "train_step", train_step), mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))
    # A single iteration is played with `dry_run=True` and `env.num_envs=2`
    assert len(gradient_steps) == 2


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


@pytest.mark.parametrize("algo", ["ppo", "a2c", "ppo_recurrent"])
def test_on_policy_truncated_episodes(standard_args, start_time, algo):
    # The time limit truncates the episodes during the rollout, so the value of the final observation
    # of every truncated episode is bootstrapped: the frame-stacked `rgb` is the only selected key,
    # while the `state` key returned by the environment is not used by the agent
    root_dir = os.path.join(f"pytest_{start_time}", algo, os.environ["LT_DEVICES"])
    run_name = f"test_{algo}_truncated_episodes"
    args = standard_args + [
        f"exp={algo}",
        "env=dummy",
        "env.id=discrete_dummy",
        "env.max_episode_steps=2",
        "env.frame_stack=2",
        "algo.rollout_steps=4",
        "algo.per_rank_batch_size=1",
        "algo.cnn_keys.encoder=[rgb]",
        "algo.mlp_keys.encoder=[]",
        f"root_dir={root_dir}",
        f"run_name={run_name}",
    ]
    if algo == "ppo_recurrent":
        args += ["algo.per_rank_sequence_length=2", "fabric.precision=32"]

    with mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))


@pytest.mark.parametrize("env_id", [None, "continuous_dummy"])
def test_ppo_recurrent(standard_args, start_time, env_id):
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
    if env_id is not None:
        # Continuous actions crashed (#29)
        args += ["env=dummy", f"env.id={env_id}", "algo.cnn_keys.encoder=[]", "algo.mlp_keys.encoder=[state]"]

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


def test_dreamer_v3_restart_on_exception(standard_args, start_time):
    # The second environment crashes during its first step and it is restarted by the `RestartOnException`
    # wrapper: the restart must be handled for that environment only (the environments run in the rank-0 process)
    if os.environ["LT_DEVICES"] != "1":
        pytest.skip("The crash is injected in the environments of the rank-0 process only")
    from sheeprl.algos.dreamer_v3.agent import PlayerDV3
    from sheeprl.envs.dummy import DiscreteDummyEnv

    envs, crashes, reset_envs = [], [], []
    dummy_init, dummy_step, player_init_states = DiscreteDummyEnv.__init__, DiscreteDummyEnv.step, PlayerDV3.init_states

    def init(self, *args, **kwargs):
        dummy_init(self, *args, **kwargs)
        envs.append(self)

    def step(self, action):
        if self is envs[1] and len(crashes) == 0:
            crashes.append(1)
            raise RuntimeError("Environment crashed")
        return dummy_step(self, action)

    def init_states(self, reset_envs_idxes=None):
        reset_envs.append(reset_envs_idxes)
        return player_init_states(self, reset_envs_idxes)

    root_dir = os.path.join(f"pytest_{start_time}", "dreamer_v3", os.environ["LT_DEVICES"])
    run_name = "test_dreamer_v3_restart_on_exception"
    args = standard_args + [
        "exp=dreamer_v3",
        "env=dummy",
        "env.id=discrete_dummy",
        "algo.per_rank_batch_size=1",
        "algo.per_rank_sequence_length=1",
        "buffer.size=4",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=8",
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
    ]

    with (
        mock.patch.object(DiscreteDummyEnv, "__init__", init),
        mock.patch.object(DiscreteDummyEnv, "step", step),
        mock.patch.object(PlayerDV3, "init_states", init_states),
        mock.patch("time.sleep"),
        mock.patch.object(sys, "argv", args),
    ):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))
    assert len(crashes) == 1
    assert [1] in reset_envs


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


@pytest.mark.parametrize("algo", ["p2e_dv1", "p2e_dv2", "p2e_dv3"])
def test_p2e_intrinsic_reward_is_differentiable(standard_args, start_time, algo):
    # The intrinsic reward must be differentiable w.r.t. the imagined trajectories, so that the exploration
    # actor is trained through it when the actions are continuous. The inputs of the ensembles are checked
    # on the rank-0 process, which is the one running the test
    if os.environ["LT_DEVICES"] != "1":
        pytest.skip("The ensembles are checked on the rank-0 process only")
    import importlib

    exploration = importlib.import_module(f"sheeprl.algos.{algo}.{algo}_exploration")
    inputs_require_grad = []

    # The algorithms ported to the shared core create their models with `build_models` (the ensembles last), the
    # others with `build_agent` (the ensembles second)
    builder_name, ensembles_idx = ("build_models", -1) if hasattr(exploration, "build_models") else ("build_agent", 1)
    exploration_builder = getattr(exploration, builder_name)

    def builder(*args, **kwargs):
        models = exploration_builder(*args, **kwargs)
        for ens in models[ensembles_idx]:
            ens.register_forward_pre_hook(lambda _, inputs: inputs_require_grad.append(inputs[0].requires_grad))
        return models

    root_dir = os.path.join(f"pytest_{start_time}", algo, os.environ["LT_DEVICES"])
    run_name = f"test_{algo}_intrinsic_reward_is_differentiable"
    args = standard_args + [
        f"exp={algo}_exploration",
        "env=dummy",
        "env.id=continuous_dummy",
        "algo.per_rank_batch_size=2",
        f"algo.per_rank_sequence_length={1 if algo == 'p2e_dv3' else 2}",
        "buffer.size=4",
        "algo.learning_starts=0",
        "algo.replay_ratio=1",
        "algo.horizon=4",
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
    ]

    with mock.patch.object(exploration, builder_name, builder), mock.patch.object(sys, "argv", args):
        run()
    remove_test_dir(os.path.join("logs", "runs", f"pytest_{start_time}"))
    assert any(inputs_require_grad)
