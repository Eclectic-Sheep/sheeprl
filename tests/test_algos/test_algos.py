import importlib
import os
import sys
from contextlib import nullcontext
from unittest import mock

import pytest

from fabricrl.cli import tasks


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_DEVICES": "1"}, clear=True)
def test_algos():
    for module, algos in tasks.items():
        for algo in algos:
            algo_name = algo
            task = importlib.import_module(f"fabricrl.algos.{module}.{algo_name}")

            args = ["mock.py", "--num_envs=1", "--total_steps=64"]
            if "ppo" in algo:
                args += ["--rollout_steps=32", "--per_rank_batch_size=16"]
            elif "sac" in algo or "droq" in algo:
                args += [
                    "--buffer_size=64",
                    "--learning_starts=4",
                    "--gradient_steps=1",
                    "--per_rank_batch_size=16",
                ]
            with mock.patch.object(sys, "argv", args):
                for command in task.__all__:
                    if command == "main":
                        with pytest.raises(RuntimeError) if "decoupled" in algo else nullcontext():
                            task.__dict__[command]()
