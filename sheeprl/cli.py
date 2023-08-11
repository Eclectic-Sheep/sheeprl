"""Adapted from https://github.com/Lightning-Universe/lightning-flash/blob/master/src/flash/__main__.py"""

import functools
import importlib
import os
import warnings
from contextlib import closing
from typing import Optional
from unittest.mock import patch

import click
from lightning.fabric.fabric import _is_using_cli

from sheeprl.utils.registry import decoupled_tasks, tasks

CONTEXT_SETTINGS = dict(help_option_names=["--sheeprl_help"])


@click.group(no_args_is_help=True, add_help_option=True, context_settings=CONTEXT_SETTINGS)
def run():
    """SheepRL zero-code command line utility."""
    if not _is_using_cli():
        warnings.warn(
            "This script was launched without the Lightning CLI. Consider to launch the script with "
            "`lightning run model ...` to scale it with Fabric"
        )


def register_command(command, task, name: Optional[str] = None):
    @run.command(
        name if name is not None else command.__name__,
        context_settings=dict(
            help_option_names=[],
            ignore_unknown_options=True,
        ),
    )
    @click.argument("cli_args", nargs=-1, type=click.UNPROCESSED)
    @functools.wraps(command)
    def wrapper(cli_args):
        with patch("sys.argv", [task.__file__] + list(cli_args)) as sys_argv_mock:
            devices = os.environ.get("LT_DEVICES", None)
            strategy = os.environ.get("LT_STRATEGY", None)
            if strategy == "fsdp":
                raise ValueError(
                    "FSDPStrategy is currently not supported. Please launch the script with another strategy: "
                    "`lightning run model --strategy=... sheeprl.py ...`"
                )
            if name in decoupled_tasks and strategy is not None:
                warnings.warn(
                    "You are running a decoupled algorithm through the Lightning CLI "
                    "and you have specified a strategy: "
                    f"`lightning run model --strategy={strategy}`. "
                    "When a decoupled algorithm is run the default strategy will be "
                    "a `lightning.fabric.strategies.DDPStrategy`."
                )
                os.environ.pop("LT_STRATEGY")
            if name in decoupled_tasks and not _is_using_cli():
                import torch.distributed.run as torchrun
                from torch.distributed.elastic.utils import get_socket_with_port

                sock = get_socket_with_port()
                with closing(sock):
                    master_port = sock.getsockname()[1]
                nproc_per_node = "2" if devices is None else devices
                torchrun_args = [
                    f"--nproc_per_node={nproc_per_node}",
                    "--nnodes=1",
                    "--node-rank=0",
                    "--start-method=spawn",
                    "--master-addr=localhost",
                    f"--master-port={master_port}",
                ] + sys_argv_mock
                torchrun.main(torchrun_args)
            else:
                if not _is_using_cli() and devices is None:
                    os.environ["LT_DEVICES"] = "1"
                command()


for module, algos in tasks.items():
    for algo in algos:
        try:
            algo_name = algo
            task = importlib.import_module(f"{module}.{algo_name}")

            for command in task.__all__:
                command = task.__dict__[command]
                register_command(command, task, name=algo_name)
        except ImportError:
            pass
