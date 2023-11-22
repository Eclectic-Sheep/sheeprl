"""Inspired by https://github.com/Farama-Foundation/Gymnasium/blob/main/setup.py"""

import pathlib
from pathlib import Path

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


setup(
    name="sheeprl",
    version=get_version(),
    long_description=(Path(__file__).parent / "docs" / "README.md").read_text(),
    long_description_content_type="text/markdown",
)
