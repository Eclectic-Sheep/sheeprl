import importlib
import os
import shutil
import sys
import time
from contextlib import closing, nullcontext
from unittest import mock

import pytest
import torch.distributed as dist
from lightning import Fabric
from lightning.fabric.fabric import _is_using_cli

from sheeprl.utils.imports import _IS_ATARI_AVAILABLE, _IS_ATARI_ROMS_AVAILABLE, _IS_WINDOWS


@pytest.fixture(params=["1", "2", "3"])
def devices(request):
    return request.param


@pytest.fixture()
def standard_args():
    return ["--num_envs=1", "--dry_run"]


@pytest.fixture()
def start_time():
    return str(int(time.time()))


@pytest.fixture(autouse=True)
def mock_env_and_destroy(devices):
    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(devices)}) as _fixture:
        if _IS_WINDOWS and devices != "1":
            pytest.skip()
        yield _fixture
    if dist.is_initialized():
        dist.destroy_process_group()


def check_checkpoint(ckpt_path: str, target_keys: set, checkpoint_buffer: bool = True):
    fabric = Fabric(accelerator="cpu")
    if not _is_using_cli():
        fabric.launch()

    # check the presence of the checkpoint
    assert os.path.isdir(ckpt_path)
    state = fabric.load(ckpt_path + "/" + os.listdir(ckpt_path)[-1])

    # the keys in the checkpoint must match with the expected keys
    ckpt_keys = set(state.keys())
    assert len(ckpt_keys.intersection(target_keys)) == len(ckpt_keys) == len(target_keys)

    # if checkpoint_buffer is false, then "rb" cannot be in the checkpoint keys
    assert checkpoint_buffer or "rb" not in ckpt_keys


@pytest.mark.timeout(60)
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_droq(standard_args, checkpoint_buffer, start_time):
    task = importlib.import_module("sheeprl.algos.droq.droq")
    root_dir = os.path.join("pytest_" + start_time, "droq", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "--per_rank_batch_size=1",
        f"--buffer_size={int(os.environ['LT_DEVICES'])}",
        "--learning_starts=0",
        "--gradient_steps=1",
        "--root_dir=" + root_dir,
        "--run_name=" + run_name,
    ]
    if checkpoint_buffer:
        args.append("--checkpoint_buffer")

    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
        keys = {"agent", "qf_optimizer", "actor_optimizer", "alpha_optimizer", "args", "global_step"}
        if checkpoint_buffer:
            keys.add("rb")
        check_checkpoint(ckpt_path, keys, checkpoint_buffer)
        shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_sac(standard_args, checkpoint_buffer, start_time):
    task = importlib.import_module("sheeprl.algos.sac.sac")
    root_dir = os.path.join("pytest_" + start_time, "sac", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "--per_rank_batch_size=1",
        f"--buffer_size={int(os.environ['LT_DEVICES'])}",
        "--learning_starts=0",
        "--gradient_steps=1",
        "--root_dir=" + root_dir,
        "--run_name=" + run_name,
    ]
    if checkpoint_buffer:
        args.append("--checkpoint_buffer")

    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
        keys = {"agent", "qf_optimizer", "actor_optimizer", "alpha_optimizer", "args", "global_step"}
        if checkpoint_buffer:
            keys.add("rb")
        check_checkpoint(ckpt_path, keys, checkpoint_buffer)
        shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_sac_pixel_continuous(standard_args, checkpoint_buffer, start_time):
    task = importlib.import_module("sheeprl.algos.sac_pixel.sac_pixel_continuous")
    root_dir = os.path.join("pytest_" + start_time, "sac", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "--per_rank_batch_size=1",
        f"--buffer_size={int(os.environ['LT_DEVICES'])}",
        "--learning_starts=0",
        "--gradient_steps=1",
        "--root_dir=" + root_dir,
        "--run_name=" + run_name,
    ]
    if checkpoint_buffer:
        args.append("--checkpoint_buffer")

    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
        keys = {
            "agent",
            "encoder",
            "decoder",
            "qf_optimizer",
            "actor_optimizer",
            "alpha_optimizer",
            "encoder_optimizer",
            "decoder_optimizer",
            "args",
            "global_step",
            "batch_size",
        }
        if checkpoint_buffer:
            keys.add("rb")
        check_checkpoint(ckpt_path, keys, checkpoint_buffer)
        shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_sac_decoupled(standard_args, checkpoint_buffer, start_time):
    task = importlib.import_module("sheeprl.algos.sac.sac_decoupled")
    root_dir = os.path.join("pytest_" + start_time, "sac_decoupled", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "--per_rank_batch_size=1",
        "--learning_starts=0",
        "--gradient_steps=1",
        "--root_dir=" + root_dir,
        "--run_name=" + run_name,
    ]
    if checkpoint_buffer:
        args.append("--checkpoint_buffer")

    with mock.patch.object(sys, "argv", [task.__file__] + args):
        import torch.distributed.run as torchrun
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
        from torch.distributed.elastic.utils import get_socket_with_port

        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]

        for command in task.__all__:
            if command == "main":
                with pytest.raises(ChildFailedError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
                    torchrun_args = [
                        f"--nproc_per_node={os.environ['LT_DEVICES']}",
                        "--nnodes=1",
                        "--node-rank=0",
                        "--start-method=spawn",
                        "--master-addr=localhost",
                        f"--master-port={master_port}",
                    ] + sys.argv
                    torchrun.main(torchrun_args)

    if os.environ["LT_DEVICES"] != "1":
        with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
            keys = {"agent", "qf_optimizer", "actor_optimizer", "alpha_optimizer", "args", "global_step"}
            if checkpoint_buffer:
                keys.add("rb")
            check_checkpoint(ckpt_path, keys, checkpoint_buffer)
            shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
def test_ppo(standard_args, start_time):
    task = importlib.import_module("sheeprl.algos.ppo.ppo")
    root_dir = os.path.join("pytest_" + start_time, "ppo", os.environ["LT_DEVICES"])
    run_name = "test_ppo"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        f"--rollout_steps={os.environ['LT_DEVICES']}",
        "--per_rank_batch_size=1",
        f"--root_dir={root_dir}",
        f"--run_name={run_name}",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
        check_checkpoint(ckpt_path, {"actor", "critic", "optimizer", "args", "update_step", "scheduler"})
        shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
def test_ppo_decoupled(standard_args, start_time):
    task = importlib.import_module("sheeprl.algos.ppo.ppo_decoupled")
    root_dir = os.path.join("pytest_" + start_time, "ppo_decoupled", os.environ["LT_DEVICES"])
    run_name = "test_ppo_decoupled"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        f"--rollout_steps={os.environ['LT_DEVICES']}",
        "--per_rank_batch_size=1",
        "--update_epochs=1",
        f"--root_dir={root_dir}",
        f"--run_name={run_name}",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        import torch.distributed.run as torchrun
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
        from torch.distributed.elastic.utils import get_socket_with_port

        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]

        for command in task.__all__:
            if command == "main":
                with pytest.raises(ChildFailedError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
                    torchrun_args = [
                        f"--nproc_per_node={os.environ['LT_DEVICES']}",
                        "--nnodes=1",
                        "--node-rank=0",
                        "--start-method=spawn",
                        "--master-addr=localhost",
                        f"--master-port={master_port}",
                    ] + sys.argv
                    torchrun.main(torchrun_args)

    if os.environ["LT_DEVICES"] != "1":
        with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
            check_checkpoint(ckpt_path, {"agent", "optimizer", "args", "update_step", "scheduler"})
            shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    not (_IS_ATARI_AVAILABLE and _IS_ATARI_ROMS_AVAILABLE),
    reason="requires Atari games to be installed. "
    "Check https://gymnasium.farama.org/environments/atari/ for more infomation",
)
def test_ppo_atari(standard_args, start_time):
    task = importlib.import_module("sheeprl.algos.ppo_pixel.ppo_atari")
    root_dir = os.path.join("pytest_" + start_time, "ppo_atari", os.environ["LT_DEVICES"])
    run_name = "test_ppo_atari"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        f"--rollout_steps={os.environ['LT_DEVICES']}",
        "--per_rank_batch_size=1",
        "--env_id=BreakoutNoFrameskip-v4",
        f"--root_dir={root_dir}",
        f"--run_name={run_name}",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        import torch.distributed.run as torchrun
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
        from torch.distributed.elastic.utils import get_socket_with_port

        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]

        for command in task.__all__:
            if command == "main":
                with pytest.raises(ChildFailedError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
                    torchrun_args = [
                        f"--nproc_per_node={os.environ['LT_DEVICES']}",
                        "--nnodes=1",
                        "--node-rank=0",
                        "--start-method=spawn",
                        "--master-addr=localhost",
                        f"--master-port={master_port}",
                    ] + sys.argv
                    torchrun.main(torchrun_args)

    if os.environ["LT_DEVICES"] != "1":
        with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
            check_checkpoint(ckpt_path, {"agent", "optimizer", "args", "update_step", "scheduler"})
            shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
def test_ppo_continuous(standard_args, start_time):
    task = importlib.import_module("sheeprl.algos.ppo_continuous.ppo_continuous")
    root_dir = os.path.join("pytest_" + start_time, "ppo_continuous", os.environ["LT_DEVICES"])
    run_name = "test_ppo_continuous"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "--rollout_steps=1",
        "--per_rank_batch_size=1",
        f"--root_dir={root_dir}",
        f"--run_name={run_name}",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
        check_checkpoint(ckpt_path, {"actor", "critic", "optimizer", "args", "update_step", "scheduler"})
        shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
def test_ppo_recurrent(standard_args, start_time):
    task = importlib.import_module("sheeprl.algos.ppo_recurrent.ppo_recurrent")
    root_dir = os.path.join("pytest_" + start_time, "ppo_recurrent", os.environ["LT_DEVICES"])
    run_name = "test_ppo_recurrent"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        f"--rollout_steps={os.environ['LT_DEVICES']}",
        "--per_rank_batch_size=1",
        f"--root_dir={root_dir}",
        f"--run_name={run_name}",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
        check_checkpoint(ckpt_path, {"agent", "optimizer", "args", "update_step", "scheduler"})
        shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
def test_ppo_pixel_continuous(standard_args, start_time):
    task = importlib.import_module("sheeprl.algos.ppo_pixel.ppo_pixel_continuous")
    root_dir = os.path.join("pytest_" + start_time, "ppo_pixel_continuous", os.environ["LT_DEVICES"])
    run_name = "test_ppo_pixel_continuous"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        f"--rollout_steps={os.environ['LT_DEVICES']}",
        "--per_rank_batch_size=1",
        "--update_epochs=1",
        f"--root_dir={root_dir}",
        f"--run_name={run_name}",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        import torch.distributed.run as torchrun
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
        from torch.distributed.elastic.utils import get_socket_with_port

        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]

        for command in task.__all__:
            if command == "main":
                with pytest.raises(ChildFailedError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
                    torchrun_args = [
                        f"--nproc_per_node={os.environ['LT_DEVICES']}",
                        "--nnodes=1",
                        "--node-rank=0",
                        "--start-method=spawn",
                        "--master-addr=localhost",
                        f"--master-port={master_port}",
                    ] + sys.argv
                    torchrun.main(torchrun_args)

    if os.environ["LT_DEVICES"] != "1":
        with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
            check_checkpoint(ckpt_path, {"agent", "optimizer", "args", "update_step", "scheduler"})
            shutil.rmtree("pytest_" + start_time)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["AdventureNoFrameskip-v4", "CarRacing-v2"])
@pytest.mark.parametrize("checkpoint_buffer", [True, False])
def test_dreamer_v1(standard_args, env_id, checkpoint_buffer, start_time):
    task = importlib.import_module("sheeprl.algos.dreamer_v1.dreamer_v1")
    root_dir = os.path.join("pytest_" + start_time, "dreamer_v1", os.environ["LT_DEVICES"])
    run_name = "checkpoint_buffer" if checkpoint_buffer else "no_checkpoint_buffer"
    ckpt_path = os.path.join(root_dir, run_name)
    version = 0 if not os.path.isdir(ckpt_path) else len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, f"version_{version}", "checkpoint")
    args = standard_args + [
        "--per_rank_batch_size=1",
        "--per_rank_sequence_length=1",
        f"--buffer_size={int(os.environ['LT_DEVICES'])}",
        "--learning_starts=0",
        "--gradient_steps=1",
        "--env_id=" + env_id,
        "--root_dir=" + root_dir,
        "--run_name=" + run_name,
        "--dense_units=8",
        "--cnn_channels_multiplier=2",
    ]
    if checkpoint_buffer:
        args.append("--checkpoint_buffer")

    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}):
        keys = {
            "world_model",
            "actor",
            "critic",
            "world_optimizer",
            "actor_optimizer",
            "critic_optimizer",
            "expl_decay_steps",
            "args",
            "global_step",
            "batch_size",
        }
        if checkpoint_buffer:
            keys.add("rb")
        check_checkpoint(ckpt_path, keys, checkpoint_buffer)
        shutil.rmtree("pytest_" + start_time)
