import gc
import os
import shutil
import sys
import time
from pathlib import Path
from unittest import mock

import jsonlines
import torch

if __name__ == "__main__":
    from sheeprl import ROOT_DIR
    from sheeprl.cli import run

    os.environ["MUJOCO_GL"] = "osmesa"

    algos = ["dreamer_v3"]
    environments = [
        ("dmc", "walker", "walk"),
        ("dmc", "hopper", "hop"),
        ("dmc", "cartpole", "swingup_sparse"),
        ("dmc", "pendulum", "swingup"),
    ]

    for algo in algos:
        for env in environments:
            for seed in range(5):
                args = [
                    os.path.join(ROOT_DIR, "__main__.py"),
                    f"exp={algo}_{env[0]}_white_paper",
                    f"env.wrapper.domain_name={env[1]}",
                    f"env.wrapper.task_name={env[2]}",
                    f"seed={seed}",
                    "fabric.accelerator=cuda",
                    "checkpoint.keep_last=1",
                    "logger@metric.logger=csv",
                    "hydra.run.dir=./logs/runs/${root_dir}/${run_name}",
                ]
                with mock.patch.object(sys, "argv", args):
                    tic = time.perf_counter()
                    run()
                    t = time.perf_counter() - tic
                with jsonlines.open("times.jsonl", "a") as writer:
                    writer.write({"algo": algo, "env": env[1], "seed": seed, "time": t})
                gc.collect()
                torch.cuda.empty_cache()

                memmap_dir = Path("logs") / "runs" / algo / env[1] + "_" + env[2]
                memmap_dir /= sorted(os.listdir(memmap_dir))[-1]
                memmap_dir = memmap_dir / "version_0" / "memmap_buffer"
                if os.path.isdir(memmap_dir):
                    shutil.rmtree(memmap_dir)
