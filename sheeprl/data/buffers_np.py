import os
import typing
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from sheeprl.utils.utils import NUMPY_TO_TORCH_DTYPE_DICT


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: Optional[Union[str, os.PathLike]] = None,
        memmap_mode: str = "r+",
    ):
        """A standard replay buffer implementation. Internally this is represented by a
        dictionary mapping string to numpy arrays.

        Args:
            buffer_size (int): the buffer size.
            n_envs (int, optional): the number of environments. Defaults to 1.
            obs_keys (Sequence[str], optional): whether to memory-mapping the buffer. Defaults to ("observations",).
            memmap (bool, optional): whether to memory-map the numpy arrays saved in the buffer. Defaults to False.
            memmap_dir (Optional[Union[str, os.PathLike]], optional): the memory-mapped files directory.
                Defaults to None.
            memmap_mode (str, optional): memory-map mode.
                Possible values are: "r+", "w+", "c", "copyonwrite", "readwrite", "write".
                Defaults to "r+".
        """
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if n_envs <= 0:
            raise ValueError(f"The number of environments must be greater than zero, got: {n_envs}")
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        self._obs_keys = obs_keys
        self._memmap = memmap
        self._memmap_dir = memmap_dir
        self._memmap_mode = memmap_mode
        self._buf: Dict[str, ArrayLike] = {}
        if self._memmap:
            if self._memmap_mode not in ("r+", "w+", "c", "copyonwrite", "readwrite", "write"):
                raise ValueError(
                    'Accepted values for memmap_mode are "r+", "readwrite", "w+", "write", "c" or '
                    '"copyonwrite". PyTorch does not support tensors backed by read-only '
                    'NumPy arrays, so "r" and "readonly" are not supported.'
                )
            if self._memmap_dir is None:
                raise ValueError(
                    "The buffer is set to be memory-mapped but the `memmap_dir` attribute is None. "
                    "Set the `memmap_dir` to a known directory.",
                )
            else:
                self._memmap_dir = Path(self._memmap_dir)
                self._memmap_dir.mkdir(parents=True, exist_ok=True)
        self._pos = 0
        self._full = False
        self._memmap_specs = {}
        self._rng: np.random.Generator = np.random.default_rng()

    @property
    def buffer(self) -> Dict[str, np.ndarray]:
        return self._buf

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def full(self) -> bool:
        return self._full

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def empty(self) -> bool:
        return (self.buffer is not None and len(self.buffer) == 0) or self.buffer is None

    @property
    def is_memmap(self) -> bool:
        return self._memmap

    def __len__(self) -> int:
        return self.buffer_size

    @torch.no_grad()
    def to_tensor(
        self,
        dtype: Optional[torch.dtype] = None,
        clone: bool = False,
        device: Union[str, torch.dtype] = "cpu",
    ) -> Dict[str, Tensor]:
        """Converts the replay buffer to a dictionary mapping string to torch.Tensor.

        Args:
            dtype (Optional[torch.dtype], optional): the torch dtype to convert the arrays to.
                If None, then the dtypes of the numpy arrays is maintained.
                Defaults to None.
            clone (bool, optional): whether to clone the converted tensors.
                Defaults to False.
            device (Union[str, torch.dtype], optional): the torch device to move the tensors to.
                Defaults to "cpu".

        Returns:
            Dict[str, Tensor]: the converted buffer.
        """
        buf = {}
        for k, v in self.buffer.items():
            torch_v = torch.as_tensor(
                v,
                dtype=NUMPY_TO_TORCH_DTYPE_DICT[v.dtype] if dtype is None else dtype,
                device=device,
            )
            if clone:
                torch_v = torch_v.clone()
            buf[k] = torch_v
        return buf

    @typing.overload
    def add(self, data: "ReplayBuffer") -> None:
        ...

    @typing.overload
    def add(self, data: Dict[str, ArrayLike]) -> None:
        ...

    def add(self, data: Union["ReplayBuffer", Dict[str, ArrayLike]]) -> None:
        """_summary_

        Args:
            data (Union[&quot;ReplayBuffer&quot;, Dict[str, ArrayLike]]): data to add.
        """
        if isinstance(data, ReplayBuffer):
            data = data.buffer
        elif not isinstance(data, dict):
            raise ValueError(
                f"`data` must be a dictionary containing Numpy arrays, but `data` is of type `{type(data)}`"
            )
        elif isinstance(data, dict):
            for k, v in data.items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"`data` must be a dictionary containing Numpy arrays. Found key `{k}` "
                        f"containing a value of type `{type(v)}`"
                    )
        if data is None:
            raise ValueError("The `data` replay buffer must be not None")
        last_key = next(iter(data.keys()))
        last_batch_shape = next(iter(data.values())).shape[:2]
        for i, (k, v) in enumerate(data.items()):
            if len(v.shape) < 2:
                raise RuntimeError(
                    "`data` must have at least 2: [sequence_length, n_envs, ...]. " f"Shape of `{k}` is {v.shape}"
                )
            if i > 0:
                current_key = k
                current_batch_shape = v.shape[:2]
                if current_batch_shape != last_batch_shape:
                    raise RuntimeError(
                        "Every array in `data` must be congruent in the first 2 dimensions: "
                        f"found key `{last_key}` with shape `{last_batch_shape}` "
                        f"and `{current_key}` with shape `{current_batch_shape}`"
                    )
                last_key = current_key
                last_batch_shape = current_batch_shape
        data_len = next(iter(data.values())).shape[0]
        next_pos = (self._pos + data_len) % self._buffer_size
        if next_pos <= self._pos or (data_len >= self._buffer_size and not self._full):
            idxes = np.array(list(range(self._pos, self._buffer_size)) + list(range(0, next_pos)))
        else:
            idxes = np.array(range(self._pos, next_pos))
        if data_len > self._buffer_size:
            data_to_store = data[-self._buffer_size - next_pos :]
        else:
            data_to_store = data
        if self._memmap and self.empty:
            for k, v in data_to_store.items():
                path = Path(self._memmap_dir / f"{k}.memmap")
                path.touch(exist_ok=True)
                fd = open(path, mode="r+")
                fd.close()
                filename = str(path)
                self._memmap_specs[filename] = {
                    "key": k,
                    "file_descriptor": fd,
                    "shape": (self._buffer_size, self._n_envs, *v.shape[2:]),
                    "dtype": v.dtype,
                }
                self.buffer[k] = np.memmap(
                    filename=filename,
                    dtype=v.dtype,
                    shape=(self._buffer_size, self._n_envs, *v.shape[2:]),
                    mode=self._memmap_mode,
                )
                self.buffer[k][idxes, :] = data_to_store[k]
        elif self.empty:
            for k, v in data_to_store.items():
                self.buffer[k] = np.empty(shape=(self._buffer_size, self._n_envs, *v.shape[2:]), dtype=v.dtype)
                self.buffer[k][idxes, :] = data_to_store[k]
        else:
            for k, v in data_to_store.items():
                self.buffer[k][idxes, :] = data_to_store[k]
        if self._pos + data_len >= self._buffer_size:
            self._full = True
        self._pos = next_pos

    @torch.no_grad()
    def sample(
        self, batch_size: int, sample_next_obs: bool = False, clone: bool = False, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the replay buffer.

        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        Args:
            batch_size (int): Number of element to sample
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.

        Returns:
            sample: the sampled dictionary, containing the sampled array,
            one for every key, with a shape of [batch_size, 1]
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        if not self._full and self._pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. Please add at least one sample calling `self.add()`"
            )
        if self._full:
            first_range_end = self._pos - 1 if sample_next_obs else self._pos
            second_range_end = self.buffer_size if first_range_end >= 0 else self.buffer_size + first_range_end
            valid_idxes = np.array(list(range(0, first_range_end)) + list(range(self._pos, second_range_end)))
            batch_idxes = valid_idxes[self._rng.integers(0, len(valid_idxes), size=(batch_size,))]
        else:
            max_pos_to_sample = self._pos - 1 if sample_next_obs else self._pos
            if max_pos_to_sample == 0:
                raise RuntimeError(
                    "You want to sample the next observations, but one sample has been added to the buffer. "
                    "Make sure that at least two samples are added."
                )
            batch_idxes = self._rng.integers(0, max_pos_to_sample, size=(batch_size,))
        return self._get_samples(batch_idxes=batch_idxes, sample_next_obs=sample_next_obs, clone=clone)

    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        clone: bool = False,
        sample_next_obs: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.dtype] = "cpu",
        **kwargs,
    ) -> Dict[str, Tensor]:
        samples = self.sample(batch_size=batch_size, sample_next_obs=sample_next_obs, clone=clone)
        for k, v in samples.items():
            torch_v = torch.as_tensor(
                v,
                dtype=NUMPY_TO_TORCH_DTYPE_DICT[v.dtype] if dtype is None else dtype,
                device=device,
            )
            samples[k] = torch_v
        return samples

    @torch.no_grad()
    def _get_samples(
        self, batch_idxes: ArrayLike, sample_next_obs: bool = False, clone: bool = False
    ) -> Dict[str, np.ndarray]:
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        env_idxes = self._rng.integers(0, self.n_envs, size=(len(batch_idxes),))
        buf: Dict[str, np.ndarray] = {}
        for k, v in self.buffer.items():
            buf[k] = v[batch_idxes, env_idxes]
            if clone:
                buf[k] = buf[k].copy()
            if sample_next_obs:
                buf[f"next_{k}"] = v[(batch_idxes + 1) % self._buffer_size, env_idxes]
                if clone:
                    buf[f"next_{k}"] = buf[f"next_{k}"].copy()
        return buf

    def __getitem__(self, key: str) -> np.ndarray:
        if not isinstance(key, str):
            raise TypeError("`key` must be a string")
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        return self.buffer.get(key)

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        self.buffer.update({key: value})

    def __del__(self) -> None:
        if self._memmap:
            for filename in self._memmap_specs.keys():
                del self._memmap_specs[filename]["file_descriptor"]
                key = self._memmap_specs[filename]["key"]
                self._buf[key] = None

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if self._memmap:
            for filename in state["_memmap_specs"].keys():
                state["_memmap_specs"][filename]["file_descriptor"] = None
                key = state["_memmap_specs"][filename]["key"]
                # We remove the buffer entry: this can be reloaded upon unpickling by reading the
                # related file
                state["_buf"][key] = None
        return state

    def __setstate__(self, state):
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        if state["_memmap"]:
            for filename in state["_memmap_specs"].keys():
                state["_memmap_specs"][filename]["file_descriptor"] = open(filename, "r+")
                state["_memmap_specs"][filename]["file_descriptor"].close()
                key = state["_memmap_specs"][filename]["key"]
                state["_buf"][key] = np.memmap(
                    filename=filename,
                    dtype=state["_memmap_specs"][filename]["dtype"],
                    shape=state["_memmap_specs"][filename]["shape"],
                    mode=state["_memmap_mode"],
                )
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
