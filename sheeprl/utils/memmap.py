"""Inspired by: https://github.com/pytorch/tensordict/blob/main/tensordict/memmap.py"""

from __future__ import annotations

import os
import tempfile
import warnings
from io import TextIOWrapper
from pathlib import Path
from sys import getrefcount
from tempfile import _TemporaryFileWrapper
from typing import Any, Tuple

import numpy as np
from numpy.typing import DTypeLike


def is_shared(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and hasattr(array, "_mmap")


class MemmapArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(
        self,
        shape: None | int | Tuple[int, ...],
        dtype: DTypeLike = None,
        mode: str = "r+",
        reset: bool = False,
        filename: str | os.PathLike | None = None,
    ):
        """Create a memory-mapped array. The memory-mapped array is stored in a file on disk and is
        lazily loaded on demand. The array can be modified in-place and is automatically flushed to
        disk when the array is deleted. The ownership of the file can be transferred only when:

        * the array is created from an already mamory-mapped array (i.e., `MemmapArray.from_array`)
        * the array is set from an already memory-mapped array (i.e., `MemmapArray.array = ...`)

        Args:
            dtype (DTypeLike): the data type of the array.
            shape (None | int | Tuple[int, ...]): the shape of the array.
            mode (str, optional): the mode to open the file with. Defaults to "r+".
            reset (bool, optional): whether to reset the opened array to 0s. Defaults to False.
            filename (str | os.PathLike | None, optional): an optional filename. If the filename is None,
                then a temporary file will be opened.
                Defaults to None.
        """
        if filename is None:
            fd, path = tempfile.mkstemp(".memmap")
            self._filename = Path(path).resolve()
            self._file = _TemporaryFileWrapper(open(fd, mode="r+"), path, delete=False)
        else:
            path = Path(filename).resolve()
            if os.path.exists(path):
                warnings.warn(
                    "The specified filename already exists. "
                    "Please be aware that any modification will be possibly reflected.",
                    category=UserWarning,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
            self._filename = path
            self._file = open(path, mode="r+")
        os.close(self._file.fileno())
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
        self._has_ownership = True
        self._array_dir = self._array.__dir__()
        self.__array_interface__ = self._array.__array_interface__

    @property
    def filename(self) -> Path:
        """Return the filename of the memory-mapped array."""
        return self._filename

    @property
    def file(self) -> TextIOWrapper:
        """Return the file object of the memory-mapped array."""
        return self._file

    @property
    def dtype(self) -> DTypeLike:
        """Return the data type of the memory-mapped array."""
        return self._dtype

    @property
    def mode(self) -> str:
        """Return the mode of the memory-mapped array that has been opened with."""
        return self._mode

    @property
    def shape(self) -> None | int | Tuple[int, ...]:
        """Return the shape of the memory-mapped array."""
        return self._shape

    @property
    def has_ownership(self) -> bool:
        """Return whether the memory-mapped array has ownership of the file."""
        return self._has_ownership

    @has_ownership.setter
    def has_ownership(self, value: bool):
        """Set whether the memory-mapped array has ownership of the file."""
        self._has_ownership = value

    @property
    def array(self) -> np.memmap:
        """Return the memory-mapped array."""
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
        """Set the memory-mapped array. If the array to be set is already memory-mapped, then the ownership of the
        file will not be transferred to this memory-mapped array; this instance will lose previous
        ownership on its memory mapped file. Otherwise, the array will be copied into
        the memory-mapped array. In this last case, the shape of the array to be set must be the same as the
        shape of the memory-mapped array.

        Args:
            v (np.memmap | np.ndarray): the array to be set.

        Raises:
            ValueError: if the value to be set is not an instance of `np.memmap` or `np.ndarray`.
        """
        if not isinstance(v, (np.memmap, np.ndarray)):
            raise ValueError(f"The value to be set must be an instance of 'np.memmap' or 'np.ndarray', got '{type(v)}'")
        if is_shared(v):
            self.__del__()
            tmpfile = _TemporaryFileWrapper(None, v.filename, delete=True)
            tmpfile.name = v.filename
            tmpfile._closer.name = v.filename
            self._file = tmpfile
            self._filename = v.filename
            self._shape = v.shape
            self._dtype = v.dtype
            self._has_ownership = False
            self.__array_interface__ = v.__array_interface__
            self._array = np.memmap(
                filename=self._filename,
                dtype=self._dtype,
                shape=self._shape,
                mode=self._mode,
            )
        else:
            if self._array.size != v.size:
                raise ValueError(
                    "The shape of the value to be set must be the same as the shape of the memory-mapped array. "
                    f"Got {v.shape} and {self._shape}"
                )
            reshaped_v = np.reshape(v, self._shape)
            self._array[:] = reshaped_v
            self._array.flush()

    @classmethod
    def from_array(
        cls,
        array: np.ndarray | np.memmap | MemmapArray,
        mode: str = "r+",
        filename: str | os.PathLike | None = None,
    ) -> MemmapArray:
        """Create a memory-mapped array from an array. If the array is already memory-mapped, then the ownership of
        the file will not be transferred to this memory-mapped array; this instance will lose previous ownership on
        its memory mapped file. Otherwise, the array will be copied into the memory-mapped array. In this last case,
        the shape of the array to be set must be the same as the shape of the memory-mapped array.

        Args:
            array (np.ndarray | np.memmap | MemmapArray): the array to be set.
            mode (str, optional): the mode to open the file with. Defaults to "r+".
            filename (str | os.PathLike | None, optional): the filename. Defaults to None.

        Returns:
            MemmapArray: the memory-mapped array.
        """
        filename = Path(filename).resolve() if filename is not None else None
        is_memmap_array = isinstance(array, MemmapArray)
        is_shared_array = is_shared(array)
        if isinstance(array, (np.ndarray, MemmapArray)):
            out = cls(filename=filename, dtype=array.dtype, shape=array.shape, mode=mode, reset=False)
            if is_memmap_array or is_shared_array:
                if is_memmap_array:
                    array = array.array
                if filename is not None and filename == Path(array.filename).resolve():
                    out.array = array  # Lose previous ownership
                else:
                    out.array[:] = array[:]
            else:
                if filename is not None and os.path.exists(filename):
                    warnings.warn(
                        "The specified filename already exists. "
                        "Please be aware that any modification will be possibly reflected.",
                        category=UserWarning,
                    )
                out.array = array
            return out

    def __del__(self) -> None:
        """Delete the memory-mapped array. If the memory-mapped array has ownership of the file and no other
        reference to the memory-mapped array exists,
        then the memory-mapped array will be flushed to disk and both the memory-mapped array and
        the file will be closed. If the memory-mapped array is mapped to a temporary file then the file is
        removed.
        """
        if self._array is not None and self._has_ownership and getrefcount(self._file) <= 2:
            self._array.flush()
            self._array._mmap.close()
            del self._array._mmap
            self._array = None
            if isinstance(self._file, _TemporaryFileWrapper) and os.path.isfile(self._filename):
                os.unlink(self._filename)
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
        state["_file"] = None
        state["_array"] = None
        state["_has_ownership"] = False
        return state

    def __setstate__(self, state):
        filename = state["_filename"]
        if state["_file"] is None:
            tmpfile = _TemporaryFileWrapper(None, filename, delete=True)
            tmpfile.name = filename
            tmpfile._closer.name = filename
            state["_file"] = tmpfile
        self.__dict__.update(state)

    def __getitem__(self, idx: Any) -> np.ndarray:
        return self.array[idx]

    def __setitem__(self, idx: Any, value: Any):
        self.array[idx] = value

    def __repr__(self) -> str:
        return f"MemmapArray(shape={self._shape}, dtype={self._dtype}, mode={self._mode}, filename={self._filename})"

    def __len__(self):
        return len(self.array)
