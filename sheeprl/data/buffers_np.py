from __future__ import annotations

import logging
import os
import shutil
import typing
import uuid
from pathlib import Path
from typing import Dict, Optional, Sequence, Type, Union

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
        **kwargs,
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
        from_numpy: bool = False,
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
            if from_numpy:
                torch_v = torch.from_numpy(v).to(
                    dtype=NUMPY_TO_TORCH_DTYPE_DICT[v.dtype] if dtype is None else dtype,
                    device=device,
                )
            else:
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
    def add(self, data: "ReplayBuffer", validate_args: bool = False) -> None:
        ...

    @typing.overload
    def add(self, data: Dict[str, ArrayLike], validate_args: bool = False) -> None:
        ...

    def add(self, data: Union["ReplayBuffer", Dict[str, ArrayLike]], validate_args: bool = False) -> None:
        """_summary_

        Args:
            data (Union[&quot;ReplayBuffer&quot;, Dict[str, ArrayLike]]): data to add.
        """
        if validate_args:
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
        if next_pos <= self._pos or (data_len > self._buffer_size and not self._full):
            idxes = np.array(list(range(self._pos, self._buffer_size)) + list(range(0, next_pos)))
        else:
            idxes = np.array(range(self._pos, next_pos))
        if data_len > self._buffer_size:
            data_to_store = {k: v[-self._buffer_size - next_pos :] for k, v in data.items()}
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
                self.buffer[k][idxes] = data_to_store[k]
        elif self.empty:
            for k, v in data_to_store.items():
                self.buffer[k] = np.empty(shape=(self._buffer_size, self._n_envs, *v.shape[2:]), dtype=v.dtype)
                self.buffer[k][idxes] = data_to_store[k]
        else:
            for k, v in data_to_store.items():
                self.buffer[k][idxes] = data_to_store[k]
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
            valid_idxes = np.array(
                list(range(0, first_range_end)) + list(range(self._pos, second_range_end)), dtype=np.intp
            )
            batch_idxes = valid_idxes[self._rng.integers(0, len(valid_idxes), size=(batch_size,), dtype=np.intp)]
        else:
            max_pos_to_sample = self._pos - 1 if sample_next_obs else self._pos
            if max_pos_to_sample == 0:
                raise RuntimeError(
                    "You want to sample the next observations, but one sample has been added to the buffer. "
                    "Make sure that at least two samples are added."
                )
            batch_idxes = self._rng.integers(0, max_pos_to_sample, size=(batch_size,), dtype=np.intp)
        return self._get_samples(batch_idxes=batch_idxes, sample_next_obs=sample_next_obs, clone=clone)

    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        clone: bool = False,
        sample_next_obs: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.dtype] = "cpu",
        from_numpy: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        samples = self.sample(batch_size=batch_size, sample_next_obs=sample_next_obs, clone=clone, **kwargs)
        for k, v in samples.items():
            if from_numpy:
                torch_v = torch.from_numpy(v).to(
                    dtype=NUMPY_TO_TORCH_DTYPE_DICT[v.dtype] if dtype is None else dtype,
                    device=device,
                )
            else:
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
        samples: Dict[str, np.ndarray] = {}
        env_idxes = self._rng.integers(0, self.n_envs, size=(len(batch_idxes),), dtype=np.intp)
        flattened_idxes = (batch_idxes * self.n_envs + env_idxes).flat
        if sample_next_obs:
            flattened_next_idxes = (((batch_idxes + 1) % self._buffer_size) * self.n_envs + env_idxes).flat
        for k, v in self.buffer.items():
            samples[k] = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_idxes, axis=0)
            if clone:
                samples[k] = samples[k].copy()
            if sample_next_obs:
                samples[f"next_{k}"] = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_next_idxes, axis=0)
                if clone:
                    samples[f"next_{k}"] = samples[f"next_{k}"].copy()
        return samples

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


class SequentialReplayBuffer(ReplayBuffer):
    """A replay buffer which internally uses a TensorDict and returns sequential samples.

    Args:
        buffer_size (int): The buffer size.
        n_envs (int, optional): The number of environments. Defaults to 1.
        device (Union[torch.device, str], optional): The device where the buffer is created. Defaults to "cpu".
    """

    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: Optional[Union[str, os.PathLike]] = None,
        memmap_mode: str = "r+",
        **kwargs,
    ):
        super().__init__(buffer_size, n_envs, obs_keys, memmap, memmap_dir, memmap_mode, **kwargs)

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        sequence_length: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the sequential replay buffer,
        each one is a sequence of a consecutive items.

        Custom sampling when using memory efficient variant,
        as the first element of the sequence cannot be in a position
        greater than (pos - sequence_length) % buffer_size.
        See comments in the code for more information.

        Args:
            batch_size (int): Number of element to sample
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.
            clone (bool): whether to clone the sampled TensorDict.
            sequence_length (int): the length of the sequence of each element. Defaults to 1.
            n_samples (int): the number of samples to perform. Defaults to 1.

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of [n_samples, sequence_length, batch_size, ...]
            for every element in it
        """
        # the batch_size can be fused with the number of samples to have single batch size
        batch_dim = batch_size * n_samples

        # Sanity checks
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(f"`batch_size` ({batch_size}) and `n_samples` ({n_samples}) must be both greater than 0")
        if not self.full and self._pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. Please add at least one sample calling `self.add()`"
            )
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        if not self.full and self._pos - sequence_length + 1 < 1:
            raise ValueError(f"Cannot sample a sequence of length {sequence_length}. Data added so far: {self._pos}")
        if self.full and sequence_length > len(self.buffer):
            raise ValueError(
                f"The sequence length ({sequence_length}) is greater than the buffer size ({len(self.buffer)})"
            )

        # Do not sample the element with index `self.pos` as the transitions is invalid
        if self.full:
            # when the buffer is full, it is necessary to avoid the starting index
            # to be between (self.pos - sequence_length)
            # and self.pos, so it is possible to sample
            # the starting index between (0, self.pos - sequence_length) and (self.pos, self.buffer_size)
            first_range_end = self._pos - sequence_length + 1
            # end of the second range, if the first range is empty, then the second range ends
            # in (buffer_size + (self._pos - sequence_length + 1)), otherwise the sequence will contain
            # invalid values
            second_range_end = self.buffer_size if first_range_end >= 0 else self.buffer_size + first_range_end
            valid_idxes = np.array(
                list(range(0, first_range_end)) + list(range(self._pos, second_range_end)), dtype=np.intp
            )
            # start_idxes are the indices of the first elements of the sequences
            start_idxes = valid_idxes[self._rng.integers(0, len(valid_idxes), size=(batch_dim,), dtype=np.intp)]
        else:
            # when the buffer is not full, we need to start the sequence so that it does not go out of bounds
            start_idxes = self._rng.integers(0, self._pos - sequence_length + 1, size=(batch_dim,), dtype=np.intp)

        # chunk_length contains the relative indices of the sequence (0, 1, ..., sequence_length-1)
        chunk_length = np.arange(sequence_length, dtype=np.intp).reshape(1, -1)
        idxes = (start_idxes.reshape(-1, 1) + chunk_length) % self.buffer_size

        # (n_samples, sequence_length, batch_size)
        return self._get_samples(
            idxes, batch_size, n_samples, sequence_length, sample_next_obs=sample_next_obs, clone=clone
        )

    def _get_samples(
        self,
        batch_idxes: ArrayLike,
        batch_size: int,
        n_samples: int,
        sequence_length: int,
        sample_next_obs: bool = False,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        batch_shape = (batch_size * n_samples, sequence_length)  # [Batch_size * N_samples, Seq_len]
        flattened_batch_idxes = np.ravel(batch_idxes)

        # Each sequence must come from the same environment
        if self._n_envs == 1:
            env_idxes = np.zeros((np.prod(batch_shape),), dtype=np.intp)
        else:
            env_idxes = self._rng.integers(0, self.n_envs, size=(batch_shape[0],), dtype=np.intp)
            env_idxes = np.reshape(env_idxes, (-1, 1))
            env_idxes = np.tile(env_idxes, (1, sequence_length))
            env_idxes = np.ravel(env_idxes)

        # Flatten indexes
        flattened_idxes = (flattened_batch_idxes * self._n_envs + env_idxes).flat

        # Get samples
        samples: Dict[str, np.ndarray] = {}
        for k, v in self.buffer.items():
            # Retrieve the items by flattening the indices
            # (b1_s1, b1_s2, b1_s3, ..., bn_s1, bn_s2, bn_s3, ...)
            # where bm_sk is the k-th elements in the sequence of the m-th batch
            flattened_v = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_idxes, axis=0)
            # Properly reshape the items:
            # [
            #   [b1_s1, b1_s2, ...],
            #   [b2_s1, b2_s2, ...],
            #   ...,
            #   [bn_s1, bn_s2, ...]
            # ]
            batched_v = np.reshape(flattened_v, (n_samples, batch_size, sequence_length) + flattened_v.shape[1:])
            # Reshape back to # [N_samples, Seq_len, Batch_size]
            samples[k] = np.swapaxes(
                batched_v,
                axis1=1,
                axis2=2,
            )
            if clone:
                samples[k] = batched_v.copy()
            if sample_next_obs:
                flattened_next_v = v[(flattened_batch_idxes + 1) % self._buffer_size, env_idxes]
                batched_next_v = np.reshape(
                    flattened_next_v, (n_samples, batch_size, sequence_length) + flattened_next_v.shape[1:]
                )
                samples[f"next_{k}"] = np.swapaxes(
                    batched_next_v,
                    axis1=1,
                    axis2=2,
                )
                if clone:
                    samples[f"next_{k}"] = samples[f"next_{k}"].copy()
        return samples


class EnvIndipendentReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: Optional[Union[str, os.PathLike]] = None,
        memmap_mode: str = "r+",
        buffer_cls: Type[ReplayBuffer] = ReplayBuffer,
        **kwargs,
    ):
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if n_envs <= 0:
            raise ValueError(f"The number of environments must be greater than zero, got: {n_envs}")
        if memmap:
            if memmap_mode not in ("r+", "w+", "c", "copyonwrite", "readwrite", "write"):
                raise ValueError(
                    'Accepted values for memmap_mode are "r+", "readwrite", "w+", "write", "c" or '
                    '"copyonwrite". PyTorch does not support tensors backed by read-only '
                    'NumPy arrays, so "r" and "readonly" are not supported.'
                )
            if memmap_dir is None:
                raise ValueError(
                    "The buffer is set to be memory-mapped but the `memmap_dir` attribute is None. "
                    "Set the `memmap_dir` to a known directory.",
                )
            else:
                memmap_dir = Path(memmap_dir)
                memmap_dir.mkdir(parents=True, exist_ok=True)
        self._buf: Sequence[ReplayBuffer] = [
            buffer_cls(
                buffer_size=buffer_size,
                n_envs=1,
                obs_keys=obs_keys,
                memmap=memmap,
                memmap_dir=memmap_dir / f"env_{i}",
                memmap_mode=memmap_mode,
                **kwargs,
            )
            for i in range(n_envs)
        ]
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        self._rng: np.random.Generator = np.random.default_rng()
        self._concat_along_axis = 2 if buffer_cls == SequentialReplayBuffer else 0

    @property
    def buffer(self) -> Sequence[ReplayBuffer]:
        return tuple(self._buf)

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def full(self) -> Sequence[bool]:
        return tuple([b.full for b in self.buffer])

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def empty(self) -> Sequence[bool]:
        return tuple([b.empty for b in self.buffer])

    @property
    def is_memmap(self) -> Sequence[bool]:
        return tuple([b.is_memmap for b in self.buffer])

    def __len__(self) -> int:
        return self.buffer_size

    def add(
        self, data: Dict[str, ArrayLike], indices: Optional[Sequence[int]] = None, validate_args: bool = False
    ) -> None:
        """_summary_

        Args:
            data (Union[&quot;ReplayBuffer&quot;, Dict[str, ArrayLike]]): data to add.
        """
        if indices is None:
            indices = tuple(range(self.n_envs))
        for env_data_idx, env_idx in enumerate(indices):
            env_data = {k: v[:, env_data_idx : env_data_idx + 1] for k, v in data.items()}
            self._buf[env_idx].add(env_data, validate_args=validate_args)

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        sequence_length: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the sequential replay buffer,
        each one is a sequence of a consecutive items.

        Custom sampling when using memory efficient variant,
        as the first element of the sequence cannot be in a position
        greater than (pos - sequence_length) % buffer_size.
        See comments in the code for more information.

        Args:
            batch_size (int): Number of element to sample
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.
            clone (bool): whether to clone the sampled TensorDict.
            n_samples (int): the number of samples to perform. Defaults to 1.
            sequence_length (int): the length of the sequence of each element. Defaults to 1.

        Returns:
            TensorDictBase: the sampled TensorDictBase with a `batch_size` of [n_samples, sequence_length, batch_size]
        """
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(f"`batch_size` ({batch_size}) and `n_samples` ({n_samples}) must be both greater than 0")
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")

        bs_per_buf = np.bincount(self._rng.integers(0, self._n_envs, (batch_size,)))
        per_buf_samples = [
            b.sample(
                batch_size=bs,
                sample_next_obs=sample_next_obs,
                clone=clone,
                n_samples=n_samples,
                sequence_length=sequence_length,
            )
            for b, bs in zip(self._buf, bs_per_buf)
            if bs > 0
        ]
        samples = {}
        for k in per_buf_samples[0].keys():
            samples[k] = np.concatenate([s[k] for s in per_buf_samples], axis=self._concat_along_axis)
        return samples

    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        sequence_length: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.dtype] = "cpu",
        from_numpy: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        samples = self.sample(
            batch_size=batch_size,
            sample_next_obs=sample_next_obs,
            clone=clone,
            n_samples=n_samples,
            sequence_length=sequence_length,
            concat_along_axis=self._concat_along_axis,
        )
        for k, v in samples.items():
            if from_numpy:
                torch_v = torch.from_numpy(v).to(
                    dtype=NUMPY_TO_TORCH_DTYPE_DICT[v.dtype] if dtype is None else dtype,
                    device=device,
                )
            else:
                torch_v = torch.as_tensor(
                    v,
                    dtype=NUMPY_TO_TORCH_DTYPE_DICT[v.dtype] if dtype is None else dtype,
                    device=device,
                )
            samples[k] = torch_v
        return samples


class EpisodeBuffer:
    """A replay buffer that stores separately the episodes.

    Args:
        buffer_size (int): The capacity of the buffer.
        sequence_length (int): The length of the sequences of the samples
            (an episode cannot be shorter than the episode length).
        n_envs (int): The number of environments.
            Default to 1.
        obs_keys (Sequence[str]): The observations keys to store in the buffer.
            Default to ("observations",).
        prioritize_ends (bool): Whether to prioritize the ends of the episodes when sampling.
            Default to False.
        memmap (bool): Whether to memory-mapping the buffer.
            Default to False.
        memmap_dir (Union[str, os.PathLike], optional): The directory for the memmap.
            Default to None.
        memmap_mode (str, optional): memory-map mode.
            Possible values are: "r+", "w+", "c", "copyonwrite", "readwrite", "write".
            Defaults to "r+".
    """

    def __init__(
        self,
        buffer_size: int,
        sequence_length: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        prioritize_ends: bool = False,
        memmap: bool = False,
        memmap_dir: Optional[Union[str, os.PathLike]] = None,
        memmap_mode: str = "r+",
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if sequence_length <= 0:
            raise ValueError(f"The sequence length must be greater than zero, got: {sequence_length}")
        if buffer_size < sequence_length:
            raise ValueError(
                "The sequence length must be lower than the buffer size, "
                f"got: bs = {buffer_size} and sl = {sequence_length}"
            )
        self._n_envs = n_envs
        self._obs_keys = obs_keys
        self._buffer_size = buffer_size
        self._sequence_length = sequence_length
        self._prioritize_ends = prioritize_ends

        # Contain the specifications of the memmaped episodes
        self._episode_specs = []
        # One list for each environment that contains open episodes:
        # one open episode per environment
        self._open_episodes = [[] for _ in range(n_envs)]
        # Contain the cumulative length of the episodes in the buffer
        self._cum_lengths: Sequence[int] = []
        # List of stored episodes
        self._buf: Sequence[Dict[str, ArrayLike]] = []

        self._memmap = memmap
        self._memmap_dir = memmap_dir
        self._memmap_mode = memmap_mode
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
        self._chunk_length = np.arange(sequence_length, dtype=np.intp).reshape(1, -1)

    def _get_buf(self) -> Sequence[Dict[str, ArrayLike]]:
        if self._memmap:
            for i, ep_spec in enumerate(self._episode_specs):
                for filename, file_specs in ep_spec.items():
                    key = file_specs["key"]
                    if key in self._buf[i] and self._buf[i][key] is None:
                        file_specs["file_descriptor"] = open(filename, "r+")
                        file_specs["file_descriptor"].close()
                        self._buf[i][key] = np.memmap(
                            filename=filename,
                            dtype=file_specs["dtype"],
                            shape=file_specs["shape"],
                            mode=self._memmap_mode,
                        )
        return self._buf

    buf = property(_get_buf)

    @property
    def prioritize_ends(self) -> bool:
        return self._prioritize_ends

    @prioritize_ends.setter
    def prioritize_ends(self, prioritize_ends: bool) -> None:
        self._prioritize_ends = prioritize_ends

    @property
    def buffer(self) -> Optional[Dict[str, ArrayLike]]:
        if len(self.buf) > 0:
            return {k: np.concatenate([v[k] for v in self._buf]) for k in self._obs_keys}
        else:
            return {}

    @property
    def obs_keys(self) -> Sequence[str]:
        return self._obs_keys

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def is_memmap(self) -> bool:
        return self._memmap

    @property
    def full(self) -> bool:
        return self._cum_lengths[-1] + self._sequence_length > self._buffer_size if len(self.buf) > 0 else False

    def __len__(self) -> int:
        return self._cum_lengths[-1] if len(self.buf) > 0 else 0

    @typing.overload
    def add(self, data: "ReplayBuffer", validate_args: bool = False) -> None:
        ...

    @typing.overload
    def add(self, data: Dict[str, ArrayLike], validate_args: bool = False) -> None:
        ...

    def add(self, data: Union["ReplayBuffer", Dict[str, ArrayLike]], validate_args: bool = False) -> None:
        """_summary_

        Args:
            data (Union[&quot;ReplayBuffer&quot;, Dict[str, ArrayLike]]): data to add.
        """
        if isinstance(data, ReplayBuffer):
            data = data.buffer
        if validate_args:
            if data is None:
                raise ValueError("The `data` replay buffer must be not None")
            if not isinstance(data, dict):
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

            if "dones" not in data:
                raise RuntimeError(f"The episode must contain the `dones` key, got: {data.keys()}")

        # For each environment
        for env in range(self._n_envs):
            # Take the data from a single environment
            env_data = {k: v[:, env] for k, v in data.items()}
            done = env_data["dones"]
            # Take episode ends
            episode_ends = done.nonzero()[0].tolist()
            # If there is not any done, then add the data to the respective open episode
            if len(episode_ends) == 0:
                self._open_episodes[env].append(env_data)
            else:
                # In case there is at leas one done, then split the environment data into episodes
                episode_ends.append(len(done))
                start = 0
                # For each episode in the received data
                for ep_end_idx in episode_ends:
                    stop = ep_end_idx
                    # Take the episode from the data
                    episode = {k: env_data[k][start : stop + 1] for k in self._obs_keys}
                    # If the episode length is greater than zero, then add it to the open episode
                    # of the corresponding environment.
                    if len(episode["dones"]) > 0:
                        self._open_episodes[env].append(episode)
                    start = stop + 1
                    # If the open episode is not empty and the last element is a done, then save the episode
                    # in the buffer and clear the open episode
                    if len(self._open_episodes[env]) > 0 and self._open_episodes[env][-1]["dones"][-1] == 1:
                        self._save_episode(self._open_episodes[env])
                        self._open_episodes[env] = []

    def _save_episode(self, episode_chunks: Sequence[Dict[str, ArrayLike]]) -> None:
        if len(episode_chunks) == 0:
            raise RuntimeError("Invalid episode, an empty sequence is given. You must pass a non-empty sequence.")
        # Concatenate all the chunks of the episode
        episode = {k: [] for k in self._obs_keys}
        for chunk in episode_chunks:
            for k in self._obs_keys:
                episode[k].append(chunk[k])
        episode = {k: np.concatenate(episode[k], axis=0) for k in self._obs_keys}

        # Control the validity of the episode
        ep_len = episode["dones"].shape[0]
        if len(episode["dones"].nonzero()[0]) != 1 or episode["dones"][-1] != 1:
            raise RuntimeError(f"The episode must contain exactly one done, got: {len(np.nonzero(episode['dones']))}")
        if ep_len < self._sequence_length:
            raise RuntimeError(f"Episode too short (at least {self._sequence_length} steps), got: {ep_len} steps")
        if ep_len > self._buffer_size:
            raise RuntimeError(f"Episode too long (at most {self._buffer_size} steps), got: {ep_len} steps")

        # If the buffer is full, then remove the oldest episodes
        if self.full or len(self) + ep_len > self._buffer_size:
            # Compute the index of the last episode to remove
            cum_lengths = np.array(self._cum_lengths)
            mask = (len(self) - cum_lengths + ep_len) <= self._buffer_size
            last_to_remove = mask.argmax()
            # Remove all memmaped episodes
            if self._memmap and self._memmap_dir is not None:
                for _ in range(last_to_remove + 1):
                    for k, file_info in self._episode_specs[0].items():
                        file_info["file_descriptor"].close()
                        file_info["file_descriptor"] = None
                        filename = k
                    self._buf[0] = None
                    self._episode_specs[0] = None
                    del self._buf[0]
                    del self._episode_specs[0]
                    try:
                        shutil.rmtree(os.path.dirname(filename))
                    except Exception as e:
                        logging.error(e)
            else:
                self._buf = self._buf[last_to_remove + 1 :]
            # Update the cum_lengths lists
            cum_lengths = cum_lengths[last_to_remove + 1 :] - cum_lengths[last_to_remove]
            self._cum_lengths = cum_lengths.tolist()
        self._cum_lengths.append(len(self) + ep_len)
        episode_to_store = episode
        if self._memmap:
            episode_dir = self._memmap_dir / f"episode_{str(uuid.uuid4())}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            episode_specs = {}
            episode_to_store = {}
            for k, v in episode.items():
                path = Path(episode_dir / f"{k}.memmap")
                path.touch(exist_ok=True)
                fd = open(path, mode="r+")
                fd.close()
                filename = str(path)
                episode_specs[filename] = {
                    "key": k,
                    "file_descriptor": fd,
                    "shape": v.shape,
                    "dtype": v.dtype,
                }
                episode_to_store[k] = np.memmap(
                    filename=filename,
                    dtype=v.dtype,
                    shape=v.shape,
                    mode=self._memmap_mode,
                )
                episode_to_store[k][:] = episode[k]
            self._episode_specs.append(episode_specs)
        self._buf.append(episode_to_store)
        _ = self.buf

    def sample(
        self,
        batch_size: int,
        n_samples: int = 1,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Sample trajectories from the replay buffer.

        Args:
            batch_size (int): Number of element in the batch.
            n_samples (bool): The number of samples to be retrieved.
                Defaults to 1.
            clone (bool): Whether to clone the samples.
                Default to False.

        Returns:
            TensorDictBase: the sampled TensorDictBase with a `batch_size` of [batch_size, 1]
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be greater than 0, got: {batch_size}")
        if n_samples <= 0:
            raise ValueError(f"The number of samples must be greater than 0, got: {n_samples}")
        if len(self) == 0:
            raise RuntimeError(
                "No sample has been added to the buffer. Please add at least one sample calling `self.add()`"
            )

        nsample_per_eps = np.bincount(np.random.randint(0, len(self.buf), (batch_size * n_samples,))).astype(np.intp)
        samples = {k: [] for k in self._obs_keys}
        for i, n in enumerate(nsample_per_eps):
            ep_len = self._buf[i]["dones"].shape[0]
            # Define the maximum index that can be sampled in the episodes
            upper = ep_len - self._sequence_length + 1
            # If you want to prioritize ends, then all the indices of the episode
            # can be sampled as starting index
            if self._prioritize_ends:
                upper += self._sequence_length
            # Sample the starting indices and upper bound with `ep_len - self._sequence_length`
            start_idxes = np.minimum(
                np.random.randint(0, upper, size=(n,)).reshape(-1, 1), ep_len - self._sequence_length, dtype=np.intp
            )
            # Compute the indices of the sequences
            indices = start_idxes + self._chunk_length
            # Retrieve the data
            for k in self._obs_keys:
                samples[k].append(self._buf[i][k][indices])
        # Concatenate all the trajectories on the batch dimension and properly reshape them
        samples = {
            k: np.moveaxis(
                np.concatenate(samples[k], axis=0).reshape(
                    n_samples, batch_size, self._sequence_length, *samples[k][0].shape[2:]
                ),
                2,
                1,
            )
            for k in self._obs_keys
            if len(samples[k]) > 0
        }
        if clone:
            return {k: v.clone() for k, v in samples.items()}
        return samples

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if self._memmap:
            for i in range(len(state["_episode_specs"])):
                for filename in state["_episode_specs"][i].keys():
                    state["_episode_specs"][i][filename]["file_descriptor"] = None
                    key = state["_episode_specs"][i][filename]["key"]
                    # We remove the buffer entry: this can be reloaded upon unpickling by reading the
                    # related file
                    state["_buf"][i][key] = None
        return state

    def __setstate__(self, state):
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        if state["_memmap"]:
            for i in range(len(state["_episode_specs"])):
                for filename in state["_episode_specs"][i].keys():
                    state["_episode_specs"][i][filename]["file_descriptor"] = open(filename, "r+")
                    state["_episode_specs"][i][filename]["file_descriptor"].close()
                    key = state["_episode_specs"][i][filename]["key"]
                    state["_buf"][i][key] = np.memmap(
                        filename=filename,
                        dtype=state["_episode_specs"][i][filename]["dtype"],
                        shape=state["_episode_specs"][i][filename]["shape"],
                        mode=state["_memmap_mode"],
                    )
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
