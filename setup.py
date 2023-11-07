"""Inspired by https://github.com/Farama-Foundation/Gymnasium/blob/main/setup.py"""

import distutils
import distutils.command.clean
import glob
import os
import pathlib
import shutil
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT_DIR = Path(__file__).parent.resolve()
CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the sheeprl version."""
    path = CWD / "sheeprl" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("Bad version data in __init__.py")


def _get_packages():
    exclude = [
        "build*",
        "test*",
        "third_party*",
        "tools*",
    ]
    return find_packages(exclude=exclude)


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove tensordict extension
        for path in (ROOT_DIR / "sheeprl").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [ROOT_DIR / "build"]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


def get_extensions():
    extension = CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++17",
            "-fdiagnostics-color=always",
        ]
    }
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")
        extra_compile_args = {
            "cxx": [
                "-O0",
                "-fno-inline",
                "-g",
                "-std=c++17",
                "-fdiagnostics-color=always",
            ]
        }
        extra_link_args = ["-O0", "-g"]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "sheeprl", "csrc")
    extension_sources = {os.path.join(extensions_dir, p) for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))}
    sources = list(extension_sources)
    sources = ["./sheeprl/csrc/node/node.cpp", "./sheeprl/csrc/bind.cpp"]

    ext_modules = [
        extension(
            "sheeprl._" + Path(source).stem,
            sources,
            include_dirs=[this_dir],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        for source in sources
        if "bind" not in source
    ]

    return ext_modules


setup(
    name="sheeprl",
    version=get_version(),
    ext_modules=get_extensions(),
    packages=find_packages(exclude=("test", "examples", "assets")),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
)
