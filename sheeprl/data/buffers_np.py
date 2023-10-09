from __future__ import annotations

import os
import typing
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
        samples: Dict[str, np.ndarray] = {}
        for k, v in self.buffer.items():
            samples[k] = v[batch_idxes, env_idxes]
            if clone:
                samples[k] = samples[k].copy()
            if sample_next_obs:
                samples[f"next_{k}"] = v[(batch_idxes + 1) % self._buffer_size, env_idxes]
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
            valid_idxes = np.array(list(range(0, first_range_end)) + list(range(self._pos, second_range_end)))
            # start_idxes are the indices of the first elements of the sequences
            start_idxes = valid_idxes[self._rng.integers(0, len(valid_idxes), size=(batch_dim,))]
        else:
            # when the buffer is not full, we need to start the sequence so that it does not go out of bounds
            start_idxes = self._rng.integers(0, self._pos - sequence_length + 1, size=(batch_dim,))

        # chunk_length contains the relative indices of the sequence (0, 1, ..., sequence_length-1)
        chunk_length = np.arange(sequence_length).reshape(1, -1)
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
            env_idxes = np.zeros((np.prod(batch_shape),), dtype=batch_idxes.dtype)
        else:
            env_idxes = self._rng.integers(0, self.n_envs, size=(batch_shape[0],))
            env_idxes = np.reshape(env_idxes, (-1, 1))
            env_idxes = np.tile(env_idxes, (1, sequence_length))
            env_idxes = np.ravel(env_idxes)

        # Get samples
        samples: Dict[str, np.ndarray] = {}
        for k, v in self.buffer.items():
            # Retrieve the items by flattening the indices
            # (b1_s1, b1_s2, b1_s3, ..., bn_s1, bn_s2, bn_s3, ...)
            # where bm_sk is the k-th elements in the sequence of the m-th batch
            flattened_v = v[flattened_batch_idxes, env_idxes]
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
            torch_v = torch.as_tensor(
                v,
                dtype=NUMPY_TO_TORCH_DTYPE_DICT[v.dtype] if dtype is None else dtype,
                device=device,
            )
            samples[k] = torch_v
        return samples
