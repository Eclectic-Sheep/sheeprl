import gc
import os
import shutil
import sys
import time
from pathlib import Path
from unittest import mock

import torch

if __name__ == "__main__":
    from sheeprl import ROOT_DIR
    from sheeprl.cli import run

    os.environ["MUJOCO_GL"] = "egl"

    algos = {
        "ppo": [
            ("mujoco", "Walker2d-v3"),
            ("mujoco", "HalfCheetah-v3"),
            ("mujoco", "Hopper-v3"),
            ("mujoco", "Ant-v3"),
            ("atari", "PongNoFrameskip-v4"),
            ("atari", "BoxingNoFrameskip-v4"),
            ("atari", "FreewayNoFrameskip-v4"),
            ("atari", "BreakoutNoFrameskip-v4"),
        ],
        "sac": [
            ("mujoco", "Walker2d-v3"),
            ("mujoco", "HalfCheetah-v3"),
            ("mujoco", "Hopper-v3"),
            ("mujoco", "Ant-v3"),
        ],
        "a2c": [
            ("mujoco", "Walker2d-v3"),
            ("mujoco", "HalfCheetah-v3"),
            ("mujoco", "Hopper-v3"),
            ("mujoco", "Ant-v3"),
            ("atari", "PongNoFrameskip-v4"),
            ("atari", "BoxingNoFrameskip-v4"),
            ("atari", "FreewayNoFrameskip-v4"),
            ("atari", "BreakoutNoFrameskip-v4"),
        ],
        "dreamer_v3": [
            ("atari", "PongNoFrameskip-v4"),
            ("atari", "BoxingNoFrameskip-v4"),
            ("atari", "FreewayNoFrameskip-v4"),
            ("atari", "BreakoutNoFrameskip-v4"),
        ],
    }

    for algo, environments in algos.items():
        for env in environments:
            for seed in range(5):
                args = [
                    os.path.join(ROOT_DIR, "__main__.py"),
                    f"exp={algo}_{env[0]}_white_paper",
                    f"env.id={env[1]}",
                    f"seed={seed}",
                    "algo.total_steps=3_000",
                    "fabric.accelerator=cuda",
                    "checkpoint.keep_last=1",
                    "logger@metric.logger=csv",
                    "hydra.run.dir=./logs/runs/${root_dir}/${run_name}",
                ]
                with mock.patch.object(sys, "argv", args):
                    tic = time.perf_counter()
                    run()
                    print(time.perf_counter() - tic)
                gc.collect()
                torch.cuda.empty_cache()

                memmap_dir = Path("logs") / "runs" / algo / env[1]
                memmap_dir /= sorted(os.listdir(memmap_dir))[-1]
                memmap_dir = memmap_dir / "version_0" / "memmap_buffer"
                if os.path.isdir(memmap_dir):
                    shutil.rmtree(memmap_dir)
