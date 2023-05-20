"""Inspired by https://github.com/Farama-Foundation/Gymnasium/blob/main/setup.py"""

import pathlib

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the sheeprl version."""
    path = CWD / "sheeprl" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("Bad version data in __init__.py")


setup(name="sheeprl", version=get_version())
