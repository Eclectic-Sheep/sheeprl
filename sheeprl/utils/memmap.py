"""Inspired by: https://github.com/pytorch/tensordict/blob/main/tensordict/memmap.py"""

from __future__ import annotations

import os
from pathlib import Path
from sys import getrefcount
from tempfile import _TemporaryFileWrapper
from typing import Any, Tuple

import numpy as np
from numpy.typing import DTypeLike


class MemmapArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(
        self, filename: str, dtype: DTypeLike, shape: None | int | Tuple[int, ...], mode: str, reset: bool = False
    ):
        path = Path(filename)
        path.touch(exist_ok=True)
        self._file = open(path, mode="r+")
        self._file.close()
        self._filename = str(path)
        self._dtype = dtype
        self._shape = shape
        self._mode = mode
        self._array = np.memmap(
            filename=self._filename,
            dtype=self._dtype,
            shape=self._shape,
            mode=self._mode,
        )
        if reset:
            self._array[:] = np.zeros_like(self._array)
        self._array_dir = self._array.__dir__()

    @property
    def array(self) -> np.memmap:
        if not os.path.isfile(self._filename):
            self._array = None
        if self._array is None:
            self._array = np.memmap(
                filename=self._filename,
                dtype=self._dtype,
                shape=self._shape,
                mode=self._mode,
            )
        return self._array

    @array.setter
    def array(self, v: np.memmap | np.ndarray):
        if not isinstance(v, (np.memmap, np.ndarray)):
            raise ValueError(f"The value to be set must be an instance of 'np.memmap' or 'np.ndarray', got '{type(v)}'")
        if isinstance(v, np.memmap):
            self._file = open(v.filename, mode="r+")
            self._file.close()
            self._filename = v.filename
            self._shape = v.shape
            self._mode = v.mode
        self._array[:] = np.reshape(v, self._array.shape)[:]

    def __del__(self) -> None:
        if getrefcount(self._file) <= 2:
            del self._file

    def __array__(self) -> np.memmap:
        return self.array

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(attr)
        if ("_array_dir" not in self.__dir__()) or (attr not in self.__getattribute__("_array_dir")):
            raise AttributeError(f"'MemmapArray' object has no attribute '{attr}'")
        array = self.__getattribute__("array")
        return getattr(array, attr)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["_file"].close()
        state["_file"] = None
        state["_array"] = None
        return state

    def __setstate__(self, state):
        filename = state["_filename"]
        if state["_file"] is None:
            tmpfile = _TemporaryFileWrapper(None, filename, delete=True)
            tmpfile.name = filename
            tmpfile._closer.name = filename
            state["_file"] = tmpfile
        self.__dict__.update(state)

    def __getitem__(self, idx: Any):
        return self.array[idx]

    def __setitem__(self, idx: Any, value: Any):
        self.array[idx] = value

    def __repr__(self) -> str:
        return f"MemmapArray(shape={self._shape}, dtype={self._dtype}, mode={self._mode}, filename={self._filename})"

    def __len__(self):
        return len(self.array)
