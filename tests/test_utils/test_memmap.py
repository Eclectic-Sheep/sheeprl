import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sheeprl.utils.imports import _IS_WINDOWS
from sheeprl.utils.memmap import MemmapArray


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("shape", [[2], [1, 2]])
def test_memmap_data_type(dtype: np.dtype, shape):
    """Test that MemmapArray can be created with a given data type and shape."""
    a = np.array([1, 0], dtype=dtype).reshape(shape)
    m = MemmapArray.from_array(a)
    assert m.dtype == a.dtype
    assert (m == a).all()
    assert m.shape == a.shape


def test_memmap_del():
    a = np.array([1])
    m = MemmapArray.from_array(a)
    filename = m.filename
    assert os.path.isfile(filename)
    del m
    assert not os.path.isfile(filename)


def test_memmap_pickling():
    a = np.array([1])
    m1 = MemmapArray.from_array(a)
    filename = m1.filename
    m1_pickle = pickle.dumps(m1)
    assert m1._has_ownership
    m2 = pickle.loads(m1_pickle)
    assert m2.filename == m1.filename
    assert not m2._has_ownership
    del m1, m2
    assert not os.path.isfile(filename)


def test_memmap_array_get_not_none():
    a = np.ones((10,)) * 2
    m1 = MemmapArray.from_array(a)
    assert m1.array is not None


def test_memmap_array_get_none():
    a = np.ones((10,)) * 2
    m1 = MemmapArray.from_array(a)
    m1.__del__()
    with pytest.raises(Exception):
        m1.array


@pytest.mark.skipif(
    _IS_WINDOWS, reason="'test_memmap_array_get_none_linux_only' should be run and succeed only on Linux"
)
def test_memmap_array_get_none_linux_only():
    a = np.ones((10,)) * 2
    m1 = MemmapArray.from_array(a)
    m2 = MemmapArray.from_array(m1, filename=m1.filename)
    del m1
    with pytest.raises(FileNotFoundError):
        m2.array


def test_memmap_array_set_from_numpy():
    a = np.ones((10,)) * 2
    m1 = MemmapArray.from_array(a)
    a = np.ones((10,)) * 3
    m1.array = a
    assert (m1.array == a).all()
    a = np.ones((10,)) * 4
    assert not (m1.array == a).all()
    del m1


def test_memmap_array_set_from_numpy_wrong_shape():
    a = np.ones((10,)) * 2
    m1 = MemmapArray.from_array(a)
    a = np.ones((11,))
    with pytest.raises(
        ValueError, match="The shape of the value to be set must be the same as the shape of the memory-mapped array. "
    ):
        m1.array = a
    del m1


def test_memmap_array_set_from_np_memmap():
    a = np.ones((10,)) * 2
    tmpfd, filename = tempfile.mkstemp(".memmap")
    os.close(tmpfd)
    memmap = np.memmap(filename, shape=a.shape, dtype=a.dtype)
    memmap[:] = a[:]
    m = MemmapArray(dtype=memmap.dtype, shape=memmap.shape)
    assert m.has_ownership
    m.array = memmap
    m.array[:] = m.array * 2
    assert not m.has_ownership
    del m
    assert os.path.isfile(filename)
    assert (memmap == 4).all()
    memmap._mmap.close()
    del memmap
    Path.unlink(Path(filename), missing_ok=True)


def test_memmap_array_set_from_memmap_array():
    a = np.ones((10,)) * 2
    m1 = MemmapArray.from_array(a)
    m2 = MemmapArray(dtype=m1.dtype, shape=m1.shape, mode=m1.mode)
    filename = m1.filename
    assert m2.has_ownership
    with pytest.raises(
        ValueError,
        match="The value to be set must be an instance of 'np.memmap' or 'np.ndarray', "
        "got '<class 'sheeprl.utils.memmap.MemmapArray'>'",
    ):
        m2.array = m1
    m2.array = m1.array
    m2.array[:] = m2 * 2
    assert not m2.has_ownership
    del m2
    assert os.path.isfile(filename)
    assert (m1.array == 4).all()
    del m1
    assert not os.path.isfile(filename)


def test_memmap_from_array_memmap_array_different_filename():
    a = np.ones((10,)) * 2
    m1 = MemmapArray.from_array(a)
    m2 = MemmapArray.from_array(m1)
    m1_filename = m1.filename
    m2_filename = m2.filename
    assert m1.has_ownership
    assert m2.has_ownership
    assert m1_filename != m2_filename
    assert (m1.array == m2.array).all()
    del m1
    del m2
    assert not os.path.isfile(m1_filename)
    assert not os.path.isfile(m2_filename)


def test_memmap_from_array_memmap_array():
    a = np.ones((10,)) * 2
    m1 = MemmapArray.from_array(a)
    m2 = MemmapArray.from_array(m1, filename=m1.filename)
    filename = m1.filename
    assert m1.has_ownership
    assert not m2.has_ownership
    del m2
    assert os.path.isfile(filename)
    del m1
    assert not os.path.isfile(filename)


@pytest.mark.parametrize("mode", ["r", "r+", "w+", "c", "readonly", "readwrite", "write", "copyonwrite"])
def test_memmap_mode(mode):
    ma = MemmapArray(shape=10, dtype=np.float32, filename="./test_memmap_mode/test.memmap")
    ma[:] = np.ones(10) * 1.5
    del ma

    ma = MemmapArray(shape=10, dtype=np.float32, filename="./test_memmap_mode/test.memmap", mode=mode)
    if mode in ("r", "readonly", "r+", "readwrite", "c", "copyonwrite"):
        # data in memmap persists
        assert (ma.array == 1.5).all()
    elif mode in ("w+", "write"):
        # memmap is initialized to zero
        assert (ma.array == 0).all()

    if mode in ("r", "readonly"):
        with pytest.raises(ValueError):
            ma[:] = np.ones(10) * 2.5
        del ma
    else:
        ma[:] = np.ones(10) * 2.5
        assert (ma.array == 2.5).all()
        del ma

        mt2 = MemmapArray(shape=10, dtype=np.float32, filename="./test_memmap_mode/test.memmap")
        if mode in ("c", "copyonwrite"):
            # tensor was only mutated in memory, not on disk
            assert (mt2.array == 1.5).all()
        else:
            assert (mt2.array == 2.5).all()
        del mt2
    shutil.rmtree("./test_memmap_mode")
