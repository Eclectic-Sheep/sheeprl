import importlib
import os
import sys
from unittest import mock

import pytest
import torch.distributed as dist


@pytest.fixture(params=["1", "2"])
def devices(request):
    return request.param


@pytest.fixture()
def standard_args():
    return ["--num_envs=1", "--total_steps=64"]


@pytest.fixture(autouse=True)
def mock_env_and_destroy(devices):
    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": str(devices)}, clear=True) as _fixture:
        yield _fixture
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.timeout(60)
def test_droq(standard_args):
    task = importlib.import_module("fabricrl.algos.droq.droq")
    args = standard_args + ["--per_rank_batch_size=16", "--buffer_size=64", "--learning_starts=4", "--gradient_steps=1"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()


@pytest.mark.timeout(60)
def test_sac(standard_args):
    task = importlib.import_module("fabricrl.algos.sac.sac")
    args = standard_args + ["--per_rank_batch_size=16", "--buffer_size=64", "--learning_starts=4", "--gradient_steps=1"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()


@pytest.mark.timeout(60)
def test_ppo(standard_args):
    task = importlib.import_module("fabricrl.algos.ppo.ppo")
    args = standard_args + ["--rollout_steps=32", "--per_rank_batch_size=16"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()


@pytest.mark.timeout(60)
def test_ppo_continuous(standard_args):
    task = importlib.import_module("fabricrl.algos.ppo_continuous.ppo_continuous")
    args = standard_args + ["--rollout_steps=32", "--per_rank_batch_size=16"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()


@pytest.mark.timeout(60)
def test_ppo_recurrent(standard_args):
    task = importlib.import_module("fabricrl.algos.ppo_recurrent.ppo_recurrent")
    args = standard_args + ["--rollout_steps=32", "--per_rank_batch_size=16"]
    with mock.patch.object(sys, "argv", [task.__file__] + args):
        for command in task.__all__:
            if command == "main":
                task.__dict__[command]()
