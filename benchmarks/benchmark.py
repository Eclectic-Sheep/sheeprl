import os
import sys
import time
from unittest import mock

if __name__ == "__main__":
    from sheeprl import ROOT_DIR
    from sheeprl.cli import run

    # PPO Arguments
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        "exp=ppo_benchmarks",
        # Decomment below to run with 2 devices
        # "fabric.devices=2",
        # "env.num_envs=2",
        # "algo.per_rank_batch_size=128"
    ]

    # A2C Arguments
    # args = [
    #     os.path.join(ROOT_DIR, "__main__.py"),
    #     "exp=a2c_benchmarks",
    #     # Decomment below to run with 2 devices
    #     # "fabric.devices=2",
    #     # "env.num_envs=2",
    #     # "algo.per_rank_batch_size=10",
    #     # "algo.rollout_steps=20",
    # ]

    # SAC Arguments
    # args = [
    #     os.path.join(ROOT_DIR, "__main__.py"),
    #     "exp=sac_benchmarks",
    #     # Decomment below to run with 2 devices
    #     # "fabric.devices=2",
    #     # "env.num_envs=8",
    #     # "algo.per_rank_batch_size=512"
    # ]

    # Dreamer Arguments
    # args = [
    #     os.path.join(ROOT_DIR, "__main__.py"),
    #     # Select the Dreamer version you want to use for running the benchmarks
    #     "exp=dreamer_v1_benchmarks",
    #     # "exp=dreamer_v2_benchmarks",
    #     # "exp=dreamer_v3_benchmarks",
    # ]
    with mock.patch.object(sys, "argv", args):
        tic = time.perf_counter()
        run()
        print(time.perf_counter() - tic)
