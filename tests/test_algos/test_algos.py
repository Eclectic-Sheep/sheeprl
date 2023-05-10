import importlib
import os
import sys
from contextlib import nullcontext
from unittest import mock

import pytest
import torch.distributed as dist


@pytest.fixture(params=["1", "2"])
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


@pytest.mark.timeout(60)
def test_droq(standard_args):
    task = importlib.import_module("fabricrl.algos.droq.droq")
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


@pytest.mark.timeout(60)
def test_sac(standard_args):
    task = importlib.import_module("fabricrl.algos.sac.sac")
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


@pytest.mark.timeout(60)
def test_sac_decoupled(standard_args):
    task = importlib.import_module("fabricrl.algos.sac.sac_decoupled")
    args = standard_args + [
        "--per_rank_batch_size=1",
        f"--buffer_size={int(os.environ['LT_DEVICES'])}",
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


@pytest.mark.timeout(60)
def test_ppo(standard_args):
    task = importlib.import_module("fabricrl.algos.ppo.ppo")
    args = standard_args + ["--rollout_steps=1", "--per_rank_batch_size=1"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()


@pytest.mark.timeout(60)
def test_ppo_decoupled(standard_args):
    task = importlib.import_module("fabricrl.algos.ppo.ppo_decoupled")
    args = standard_args + ["--rollout_steps=1", "--per_rank_batch_size=1"]
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


@pytest.mark.timeout(60)
def test_ppo_atari(standard_args):
    task = importlib.import_module("fabricrl.algos.ppo.ppo_atari")
    args = standard_args + [
        "--rollout_steps=1",
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


@pytest.mark.timeout(60)
def test_ppo_continuous(standard_args):
    task = importlib.import_module("fabricrl.algos.ppo_continuous.ppo_continuous")
    args = standard_args + ["--rollout_steps=1", "--per_rank_batch_size=1"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()


@pytest.mark.timeout(60)
def test_ppo_recurrent(standard_args):
    task = importlib.import_module("fabricrl.algos.ppo_recurrent.ppo_recurrent")
    args = standard_args + ["--rollout_steps=1", "--per_rank_batch_size=1"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()
