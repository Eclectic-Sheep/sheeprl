import importlib
import os
import subprocess
import sys
import time
from unittest import mock

os.environ["LT_DEVICES"] = "1"
os.environ["LT_ACCELERATOR"] = "cpu"
import sheeprl

args = [
    "--actor_hidden_sizes=64",
    "--critic_hidden_sizes=64",
    "--lr=7e-4",
    "--total_steps=100000",
    "--rollout_steps=5",
    "--gae_lambda=1",
    "--max_grad_norm=0.5",
    "--per_rank_batch_size=5",
]
task = importlib.import_module("sheeprl.algos.a2c.a2c")
with mock.patch.object(sys, "argv", [task.__file__] + args):
    for command in task.__all__:
        if command == "main":
            t0 = time.perf_counter()
            task.__dict__[command]()
            t1 = time.perf_counter()
print(t1 - t0)
