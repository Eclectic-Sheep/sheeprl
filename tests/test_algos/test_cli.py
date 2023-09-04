import subprocess
import sys

import pytest


def test_fsdp_strategy_fail():
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            "lightning run model --strategy=fsdp --devices=1 sheeprl.py ppo",
            shell=True,
            check=True,
        )


def test_run_decoupled_algo():
    subprocess.run(
        "lightning run model --strategy=ddp --devices=2 sheeprl.py ppo_decoupled "
        "exp=ppo dry_run=True rollout_steps=1 cnn_keys.encoder=[rgb] mlp_keys.encoder=[state]",
        shell=True,
        check=True,
    )


def test_run_algo():
    subprocess.run(
        sys.executable
        + " sheeprl.py ppo exp=ppo dry_run=True rollout_steps=1 cnn_keys.encoder=[rgb] mlp_keys.encoder=[state]",
        shell=True,
        check=True,
    )
