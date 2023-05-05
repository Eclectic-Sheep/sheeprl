"""Adapted from https://github.com/Lightning-Universe/lightning-flash/blob/master/src/flash/__main__.py"""

import functools
import importlib
import warnings
from typing import Optional
from unittest.mock import patch

import click
from lightning.fabric.fabric import _is_using_cli


@click.group(no_args_is_help=True, add_help_option=False)
def run():
    """Fabric-RL zero-code command line utility."""
    if not _is_using_cli():
        warnings.warn(
            "This script was launched without the Lightning CLI. Consider to launch the script with "
            "`lightning run model ...` to scale it with Fabric"
        )


def register_command(command, name: Optional[str] = None):
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
        with patch("sys.argv", [name if name is not None else command.__name__] + list(cli_args)):
            command()


tasks = {
    "droq": ["droq"],
    "sac": ["sac", "sac_decoupled"],
    "ppo": ["ppo", "ppo_decoupled"],
    "ppo_recurrent": ["ppo_recurrent"],
}

for module, algos in tasks.items():
    for algo in algos:
        try:
            algo_name = algo
            task = importlib.import_module(f"fabricrl.algos.{module}.{algo_name}")

            for command in task.__all__:
                command = task.__dict__[command]
                register_command(command, name=algo_name)
        except ImportError:
            pass
