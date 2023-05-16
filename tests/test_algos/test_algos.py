import importlib
import os
import sys
from contextlib import nullcontext
from unittest import mock

import pytest
import torch.distributed as dist
from lightning import Fabric
from lightning.fabric.fabric import _is_using_cli

from sheeprl.utils.imports import _IS_ATARI_AVAILABLE, _IS_ATARI_ROMS_AVAILABLE


@pytest.fixture(params=["1", "2", "3"])
def devices(request):
    return request.param


@pytest.fixture()
def standard_args():
    return ["--num_envs=1", "--dry_run"]


@pytest.fixture(autouse=True)
def mock_env_and_destroy(devices):
    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(devices)}, clear=True) as _fixture:
        yield _fixture
    if dist.is_initialized():
        dist.destroy_process_group()


def check_checkpoint(algo: str, target_keys: set):
    fabric = Fabric(accelerator="cpu")
    if not _is_using_cli():
        fabric.launch()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ckpt_path = f"{project_root}/logs/{algo}/"
    experiment_list = sorted(os.listdir(ckpt_path))
    assert len(experiment_list) > 0
    ckpt_path += experiment_list[-1] + "/"
    ckpt_path += os.listdir(ckpt_path)[-1] + "/version_0/checkpoint/"
    assert len(os.listdir(ckpt_path)) == 1
    state = fabric.load(ckpt_path + os.listdir(ckpt_path)[-1])
    ckpt_keys = set(state.keys())
    assert len(ckpt_keys.intersection(target_keys)) == len(ckpt_keys)


@pytest.mark.timeout(60)
def test_droq(standard_args):
    task = importlib.import_module("sheeprl.algos.droq.droq")
    args = standard_args + [
        "--per_rank_batch_size=1",
        f"--buffer_size={int(os.environ['LT_DEVICES'])}",
        "--learning_starts=0",
        "--gradient_steps=1",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}, clear=True):
        check_checkpoint(
            "droq", {"agent", "qf_optimizer", "actor_optimizer", "alpha_optimizer", "args", "rb", "global_step"}
        )


@pytest.mark.timeout(60)
def test_sac(standard_args):
    task = importlib.import_module("sheeprl.algos.sac.sac")
    args = standard_args + [
        "--per_rank_batch_size=1",
        f"--buffer_size={int(os.environ['LT_DEVICES'])}",
        "--learning_starts=0",
        "--gradient_steps=1",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}, clear=True):
        check_checkpoint(
            "sac", {"agent", "qf_optimizer", "actor_optimizer", "alpha_optimizer", "args", "rb", "global_step"}
        )


@pytest.mark.timeout(60)
def test_sac_decoupled(standard_args):
    task = importlib.import_module("sheeprl.algos.sac.sac_decoupled")
    args = standard_args + [
        "--per_rank_batch_size=1",
        "--learning_starts=0",
        "--gradient_steps=1",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        import torch.distributed.run as torchrun
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

        for command in task.__all__:
            if command == "main":
                with pytest.raises(ChildFailedError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
                    torchrun_args = [
                        f"--nproc_per_node={os.environ['LT_DEVICES']}",
                        "--nnodes=1",
                        "--standalone",
                    ] + sys.argv
                    torchrun.main(torchrun_args)

    if os.environ["LT_DEVICES"] != "1":
        with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}, clear=True):
            check_checkpoint(
                "sac_decoupled",
                {"agent", "qf_optimizer", "actor_optimizer", "alpha_optimizer", "args", "rb", "global_step"},
            )


@pytest.mark.timeout(60)
def test_ppo(standard_args):
    task = importlib.import_module("sheeprl.algos.ppo.ppo")
    args = standard_args + [f"--rollout_steps={os.environ['LT_DEVICES']}", "--per_rank_batch_size=1"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}, clear=True):
        check_checkpoint("ppo", {"actor", "critic", "optimizer", "args", "update_step", "scheduler"})


@pytest.mark.timeout(60)
def test_ppo_decoupled(standard_args):
    task = importlib.import_module("sheeprl.algos.ppo.ppo_decoupled")
    args = standard_args + [
        f"--rollout_steps={os.environ['LT_DEVICES']}",
        "--per_rank_batch_size=1",
        "--update_epochs=1",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        import torch.distributed.run as torchrun
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

        for command in task.__all__:
            if command == "main":
                with pytest.raises(ChildFailedError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
                    torchrun_args = [
                        f"--nproc_per_node={os.environ['LT_DEVICES']}",
                        "--nnodes=1",
                        "--standalone",
                    ] + sys.argv
                    torchrun.main(torchrun_args)

    if os.environ["LT_DEVICES"] != "1":
        with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}, clear=True):
            check_checkpoint("ppo_decoupled", {"agent", "optimizer", "args", "update_step", "scheduler"})


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    not (_IS_ATARI_AVAILABLE and _IS_ATARI_ROMS_AVAILABLE),
    reason="requires Atari games to be installed. "
    "Check https://gymnasium.farama.org/environments/atari/ for more infomation",
)
def test_ppo_atari(standard_args):
    task = importlib.import_module("sheeprl.algos.ppo.ppo_atari")
    args = standard_args + [
        f"--rollout_steps={os.environ['LT_DEVICES']}",
        "--per_rank_batch_size=1",
        "--env_id=BreakoutNoFrameskip-v4",
    ]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        import torch.distributed.run as torchrun
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

        for command in task.__all__:
            if command == "main":
                with pytest.raises(ChildFailedError) if os.environ["LT_DEVICES"] == "1" else nullcontext():
                    torchrun_args = [
                        f"--nproc_per_node={os.environ['LT_DEVICES']}",
                        "--nnodes=1",
                        "--standalone",
                    ] + sys.argv
                    torchrun.main(torchrun_args)

    if os.environ["LT_DEVICES"] != "1":
        with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}, clear=True):
            check_checkpoint("ppo_atari", {"agent", "optimizer", "args", "update_step", "scheduler"})


@pytest.mark.timeout(60)
def test_ppo_continuous(standard_args):
    task = importlib.import_module("sheeprl.algos.ppo_continuous.ppo_continuous")
    args = standard_args + ["--rollout_steps=1", "--per_rank_batch_size=1"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}, clear=True):
        check_checkpoint("ppo_continuous", {"actor", "critic", "optimizer", "args", "update_step", "scheduler"})


@pytest.mark.timeout(60)
def test_ppo_recurrent(standard_args):
    task = importlib.import_module("sheeprl.algos.ppo_recurrent.ppo_recurrent")
    args = standard_args + [f"--rollout_steps={os.environ['LT_DEVICES']}", "--per_rank_batch_size=1"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()

    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(1)}, clear=True):
        check_checkpoint("ppo_recurrent", {"agent", "optimizer", "args", "update_step", "scheduler"})
