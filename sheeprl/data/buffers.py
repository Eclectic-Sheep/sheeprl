from __future__ import annotations

import os
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
import torch
from torch import Tensor

from sheeprl.utils.memmap import MemmapArray
from sheeprl.utils.utils import NUMPY_TO_TORCH_DTYPE_DICT


class ReplayBuffer:
    batch_axis: int = 1

    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
        seed: int | np.random.SeedSequence | None = None,
        **kwargs,
    ):
        """A standard replay buffer implementation. Internally this is represented by a
        dictionary mapping string to numpy arrays. The first dimension of the arrays is the
        buffer size, while the second dimension is the number of environments.

        Args:
            buffer_size (int): the buffer size.
            n_envs (int, optional): the number of environments. Defaults to 1.
            obs_keys (Sequence[str], optional): names of the observation keys. Those are used
                to sample the next-observation. Defaults to ("observations",).
            memmap (bool, optional): whether to memory-map the numpy arrays saved in the buffer. Defaults to False.
            memmap_dir (str | os.PathLike | None, optional): the memory-mapped files directory.
                Defaults to None.
            memmap_mode (str, optional): memory-map mode.
                Possible values are: "r+", "w+", "c", "copyonwrite", "readwrite", "write".
                Defaults to "r+".
            seed (int | np.random.SeedSequence | None, optional): the seed of the random number generator
                used to sample from the buffer. Defaults to None.
            kwargs: additional keyword arguments.
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
        self._buf: Dict[str, np.ndarray | MemmapArray] = {}
        if self._memmap:
            if self._memmap_mode not in ("r+", "w+", "c", "copyonwrite", "readwrite", "write"):
                raise ValueError(
                    'Accepted values for memmap_mode are "r+", "readwrite", "w+", "write", "c" or '
                    '"copyonwrite". PyTorch does not support tensors backed by read-only '
                    'NumPy arrays, so "r" and "readonly" are not supported.'
                )
            if self._memmap_dir is None:
                raise ValueError(
                    "The buffer is set to be memory-mapped but the 'memmap_dir' attribute is None. "
                    "Set the 'memmap_dir' to a known directory.",
                )
            else:
                self._memmap_dir = Path(self._memmap_dir)
                self._memmap_dir.mkdir(parents=True, exist_ok=True)
        self._pos = 0
        self._full = False
        self._memmap_specs = {}
        self._rng: np.random.Generator = np.random.default_rng(seed)

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
        device: str | torch.dtype = "cpu",
        from_numpy: bool = False,
    ) -> Dict[str, Tensor]:
        """Converts the replay buffer to a dictionary mapping string to torch.Tensor.

        Args:
            dtype (Optional[torch.dtype], optional): the torch dtype to convert the arrays to.
                If None, then the dtypes of the numpy arrays is maintained.
                Defaults to None.
            clone (bool, optional): whether to clone the converted tensors.
                Defaults to False.
            device (str | torch.dtype, optional): the torch device to move the tensors to.
                Defaults to "cpu".
            from_numpy (bool, optional): whether to convert the numpy arrays to torch tensors
                with the 'torch.from_numpy' function. Defaults to False.

        Returns:
            Dict[str, Tensor]: the converted buffer.
        """
        buf = {}
        for k, v in self.buffer.items():
            buf[k] = get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy)
        return buf

    @typing.overload
    def add(self, data: "ReplayBuffer", validate_args: bool = False) -> None: ...

    @typing.overload
    def add(self, data: Dict[str, np.ndarray], validate_args: bool = False) -> None: ...

    def add(self, data: "ReplayBuffer" | Dict[str, np.ndarray], validate_args: bool = False) -> None:
        """Add data to the replay buffer. If the replay buffer is full, then the oldest data is overwritten.
        If data is a dictionary, then the keys must be strings and the values must be numpy arrays of shape
        [sequence_length, n_envs, ...].

        Args:
            data (ReplayBuffer | Dict[str, np.ndarray]): the data to add to the replay buffer.
            validate_args (bool, optional): whether to validate the arguments. Defaults to False.

        Raises:
            ValueError: if the data is not a dictionary containing numpy arrays.
            ValueError: if the data is not a dictionary containing numpy arrays.
            RuntimeError: if the data does not have at least 2 dimensions.
            RuntimeError: if the data is not congruent in the first 2 dimensions.
        """
        if isinstance(data, ReplayBuffer):
            data = data.buffer
        if validate_args:
            if not isinstance(data, dict):
                raise ValueError(
                    f"'data' must be a dictionary containing Numpy arrays, but 'data' is of type '{type(data)}'"
                )
            elif isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(v, np.ndarray):
                        raise ValueError(
                            f"'data' must be a dictionary containing Numpy arrays. Found key '{k}' "
                            f"containing a value of type '{type(v)}'"
                        )
            last_key = next(iter(data.keys()))
            last_batch_shape = next(iter(data.values())).shape[:2]
            for i, (k, v) in enumerate(data.items()):
                if len(v.shape) < 2:
                    raise RuntimeError(
                        "'data' must have at least 2 dimensions: [sequence_length, n_envs, ...]. "
                        f"Shape of '{k}' is {v.shape}"
                    )
                if i > 0:
                    current_key = k
                    current_batch_shape = v.shape[:2]
                    if current_batch_shape != last_batch_shape:
                        raise RuntimeError(
                            "Every array in 'data' must be congruent in the first 2 dimensions: "
                            f"found key '{last_key}' with shape '{last_batch_shape}' "
                            f"and '{current_key}' with shape '{current_batch_shape}'"
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
                self.buffer[k] = MemmapArray(
                    filename=Path(self._memmap_dir / f"{k}.memmap"),
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

    def sample(
        self, batch_size: int, sample_next_obs: bool = False, clone: bool = False, n_samples: int = 1, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the replay buffer. If the replay buffer is not full, then the samples are taken
        from the first 'self.pos' elements. Otherwise, the samples are taken from all the elements.
        When 'sample_next_obs' is True we sample until 'self.pos - 1' to avoid sampling the last observation,
        which would be invalid.
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        Args:
            batch_size (int): Number of element to sample
            sample_next_obs (bool): whether to sample the next observations from the 'self.obs_keys' keys.
                Defaults to False.
            clone (bool): whether to clone the sampled numpy arrays. Defaults to False.
            n_samples (int): the number of samples to perform. Defaults to 1.

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of [n_samples, batch_size, ...].
        """
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(f"'batch_size' ({batch_size}) and 'n_samples' ({n_samples}) must be both greater than 0")
        if not self._full and self._pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. Please add at least one sample calling 'self.add()'"
            )
        if self._full:
            # Every row can be sampled, except the last inserted one if the next observation is needed:
            # the valid rows are the ones in [pos, pos + n_valid) (modulo the buffer size)
            n_valid = self.buffer_size - int(sample_next_obs)
            batch_idxes = (
                self._pos + self._rng.integers(0, n_valid, size=(batch_size * n_samples,), dtype=np.intp)
            ) % self.buffer_size
        else:
            max_pos_to_sample = self._pos - 1 if sample_next_obs else self._pos
            if max_pos_to_sample == 0:
                raise RuntimeError(
                    "You want to sample the next observations, but one sample has been added to the buffer. "
                    "Make sure that at least two samples are added."
                )
            batch_idxes = self._rng.integers(0, max_pos_to_sample, size=(batch_size * n_samples,), dtype=np.intp)
        return {
            k: v.reshape(n_samples, batch_size, *v.shape[1:])
            for k, v in self._get_samples(batch_idxes=batch_idxes, sample_next_obs=sample_next_obs, clone=clone).items()
        }

    def _get_samples(
        self, batch_idxes: np.ndarray, sample_next_obs: bool = False, clone: bool = False
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
            if k in self._obs_keys and sample_next_obs:
                samples[f"next_{k}"] = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_next_idxes, axis=0)
                if clone:
                    samples[f"next_{k}"] = samples[f"next_{k}"].copy()
        return samples

    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        clone: bool = False,
        sample_next_obs: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: str | torch.dtype = "cpu",
        from_numpy: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Sample elements from the replay buffer and convert them to torch tensors.

        Args:
            batch_size (int): Number of elements to sample.
            clone (bool): whether to clone the sampled numpy arrays. Defaults to False.
            sample_next_obs (bool): whether to sample the next observations from the 'self.obs_keys' keys.
                Defaults to False.
            dtype (Optional[torch.dtype], optional): the torch dtype to convert the arrays to. If None,
                then the dtypes of the numpy arrays is maintained. Defaults to None.
            device (str | torch.dtype, optional): the torch device to move the tensors to. Defaults to "cpu".
            from_numpy (bool, optional): whether to convert the numpy arrays to torch tensors
                with the 'torch.from_numpy' function. If False, then the numpy arrays are converted
                with the 'torch.as_tensor' function. Defaults to False.
            kwargs: additional keyword arguments to be passed to the 'self.sample' method.

        Returns:
            Dict[str, Tensor]: the sampled dictionary, containing the sampled array,
            one for every key, with a shape of [n_samples, batch_size, ...]
        """
        n_samples = kwargs.pop("n_samples", 1)
        samples = self.sample(
            batch_size=batch_size, sample_next_obs=sample_next_obs, clone=clone, n_samples=n_samples, **kwargs
        )
        return {
            k: get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy) for k, v in samples.items()
        }

    def __getitem__(self, key: str) -> np.ndarray | np.memmap | MemmapArray:
        if not isinstance(key, str):
            raise TypeError("'key' must be a string")
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        return self.buffer.get(key)

    def __setitem__(self, key: str, value: np.ndarray | np.memmap | MemmapArray) -> None:
        if not isinstance(value, (np.ndarray, MemmapArray)):
            raise ValueError(
                "The value to be set must be an instance of 'np.ndarray', 'np.memmap' "
                f"or '{MemmapArray.__module__}.{MemmapArray.__qualname__}', "
                f"got {type(value)}"
            )
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        if value.shape[:2] != (self._buffer_size, self._n_envs):
            raise RuntimeError(
                "'value' must have at least two dimensions of dimension [buffer_size, n_envs, ...]. "
                f"Shape of 'value' is {value.shape}"
            )
        if self._memmap:
            if isinstance(value, np.ndarray):
                filename = Path(self._memmap_dir / f"{key}.memmap")
            elif isinstance(value, MemmapArray):
                filename = value.filename
            value_to_add = MemmapArray.from_array(value, filename=filename, mode=self._memmap_mode)
        else:
            if isinstance(value, np.ndarray):
                value_to_add = np.copy(value)
            elif isinstance(value, MemmapArray):
                value_to_add = np.copy(value.array)
        self.buffer.update({key: value_to_add})


class SequentialReplayBuffer(ReplayBuffer):
    batch_axis: int = 2

    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
        seed: int | np.random.SeedSequence | None = None,
        **kwargs,
    ):
        """A sequential replay buffer implementation. Internally this is represented by a
        dictionary mapping string to numpy arrays. The first dimension of the arrays is the
        buffer length, while the second dimension is the number of environments. The sequentiality comes
        from the fact that the samples are sampled as sequences of consecutive elements.

        Args:
            buffer_size (int): the buffer size.
            n_envs (int, optional): the number of environments. Defaults to 1.
            obs_keys (Sequence[str], optional): names of the observation keys. Those are used
                to sample the next-observation. Defaults to ("observations",).
            memmap (bool, optional): whether to memory-map the numpy arrays saved in the buffer. Defaults to False.
            memmap_dir (str | os.PathLike | None, optional): the memory-mapped files directory.
                Defaults to None.
            memmap_mode (str, optional): memory-map mode. Possible values are: "r+", "w+", "c", "copyonwrite",
                "readwrite", "write". Defaults to "r+".
            seed (int | np.random.SeedSequence | None, optional): the seed of the random number generator
                used to sample from the buffer. Defaults to None.
            kwargs: additional keyword arguments.
        """
        super().__init__(buffer_size, n_envs, obs_keys, memmap, memmap_dir, memmap_mode, seed=seed, **kwargs)

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        sequence_length: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.

        Args:
            batch_size (int): Number of element to sample
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.
            clone (bool): whether to clone the sampled tensors.
            n_samples (int): the number of samples to perform. Defaults to 1.
            sequence_length (int): the length of the sequence of each element. Defaults to 1.

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of
            [n_samples, sequence_length, batch_size, ...].
        """
        # the batch_size can be fused with the number of samples to have single batch size
        batch_dim = batch_size * n_samples

        # Sanity checks
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(f"'batch_size' ({batch_size}) and 'n_samples' ({n_samples}) must be both greater than 0")
        if not self.full and self._pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. Please add at least one sample calling 'self.add()'"
            )
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        if not self.full and self._pos - sequence_length + 1 < 1:
            raise ValueError(f"Cannot sample a sequence of length {sequence_length}. Data added so far: {self._pos}")
        if self.full and sequence_length > self.__len__():
            raise ValueError(
                f"The sequence length ({sequence_length}) is greater than the buffer size ({self.__len__()})"
            )

        # Do not sample the element with index 'self.pos' as the transitions is invalid
        if self.full:
            # when the buffer is full, a sequence must not cross the position of the next insertion,
            # where the newest data are followed by the oldest ones: the valid starting indices are the ones
            # in [pos, pos + buffer_size - sequence_length] (modulo the buffer size)
            n_valid = self.buffer_size - sequence_length + 1
            start_idxes = (
                self._pos + self._rng.integers(0, n_valid, size=(batch_dim,), dtype=np.intp)
            ) % self.buffer_size
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
        batch_idxes: np.ndarray,
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
                samples[k] = samples[k].copy()
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


class EnvIndependentReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
        buffer_cls: Type[ReplayBuffer] = ReplayBuffer,
        seed: int | np.random.SeedSequence | None = None,
        **kwargs,
    ):
        """A replay buffer implementation that is composed of multiple independent replay buffers.

        Args:
            buffer_size (int): the buffer size.
            n_envs (int, optional): the number of environments. Defaults to 1.
            obs_keys (Sequence[str], optional): names of the observation keys. Those are used
                to sample the next-observation. Defaults to ("observations",).
            memmap (bool, optional): whether to memory-map the numpy arrays saved in the buffer. Defaults to False.
            memmap_dir (str | os.PathLike | None, optional): the memory-mapped files directory.
                Defaults to None.
            memmap_mode (str, optional): memory-map mode. Possible values are: "r+", "w+", "c", "copyonwrite",
                "readwrite", "write". Defaults to "r+".
            buffer_cls (Type[ReplayBuffer], optional): the replay buffer class to use. Defaults to ReplayBuffer.
            seed (int | np.random.SeedSequence | None, optional): the seed from which the independent random number
                generators of this buffer and of every environment buffer are derived. Defaults to None.
            kwargs: additional keyword arguments.
        """
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
                    "The buffer is set to be memory-mapped but the 'memmap_dir' attribute is None. "
                    "Set the 'memmap_dir' to a known directory.",
                )
            else:
                memmap_dir = Path(memmap_dir)
                memmap_dir.mkdir(parents=True, exist_ok=True)
        seed_sequences = np.random.SeedSequence(seed).spawn(n_envs + 1)
        self._buf: Sequence[ReplayBuffer] = [
            buffer_cls(
                buffer_size=buffer_size,
                n_envs=1,
                obs_keys=obs_keys,
                memmap=memmap,
                memmap_dir=memmap_dir / f"env_{i}" if memmap else None,
                memmap_mode=memmap_mode,
                seed=seed_sequences[i],
                **kwargs,
            )
            for i in range(n_envs)
        ]
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        self._rng: np.random.Generator = np.random.default_rng(seed_sequences[-1])
        self._concat_along_axis = buffer_cls.batch_axis

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

    @typing.overload
    def add(self, data: "ReplayBuffer", validate_args: bool = False) -> None: ...

    @typing.overload
    def add(self, data: Dict[str, np.ndarray], validate_args: bool = False) -> None: ...

    def add(
        self,
        data: "ReplayBuffer" | Dict[str, np.ndarray],
        indices: Optional[Sequence[int]] = None,
        validate_args: bool = False,
    ) -> None:
        """Add data to the replay buffers specified by the 'indices'. If 'indices' is None, then the data is added
        one for every environment. The length of indices must be equal to the second dimension of the arrays in 'data',
        which is the number of environments. If data is a dictionary, then the keys must be strings
        and the values must be numpy arrays of shape [sequence_length, n_envs, ...].


        Args:
            data (Union[ReplayBuffer, Dict[str, np.ndarray]]): the data to add to the replay buffers.
            indices (Optional[Sequence[int]], optional): the indices of the replay buffers to add the data to.
                Defaults to None.
            validate_args (bool, optional): whether to validate the arguments. Defaults to False.
        """
        if indices is None:
            indices = tuple(range(self.n_envs))
        elif len(indices) != next(iter(data.values())).shape[1]:
            raise ValueError(
                f"The length of 'indices' ({len(indices)}) must be equal to the second dimension of the "
                f"arrays in 'data' ({next(iter(data.values())).shape[1]})"
            )
        for env_data_idx, env_idx in enumerate(indices):
            env_data = {k: v[:, env_data_idx : env_data_idx + 1] for k, v in data.items()}
            self._buf[env_idx].add(env_data, validate_args=validate_args)

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Samples data from the buffer. The returned samples are sampled given the 'buffer_cls' class
        used to initialize the buffer. The samples are concatenated along the 'buffer_cls.batch_axis' axis.

        Args:
            batch_size (int): The number of samples to draw from the buffer.
            sample_next_obs (bool): Whether to sample the next observation or the current observation.
            clone (bool): Whether to clone the data or return a reference to the original data.
            n_samples (int): The number of samples to draw for each batch element.
            **kwargs: Additional keyword arguments to pass to the underlying buffer's `sample` method.

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of
            [n_samples, sequence_length, batch_size, ...] if 'buffer_cls' is a 'SequentialReplayBuffer',
            otherwise [n_samples, batch_size, ...] if 'buffer_cls' is a 'ReplayBuffer'.
        """
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(f"'batch_size' ({batch_size}) and 'n_samples' ({n_samples}) must be both greater than 0")
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")

        # The environment of every element of every batch, drawn independently: the batches of one call don't take the
        # same number of elements from each environment
        env_idxes = self._rng.integers(0, self._n_envs, (n_samples, batch_size))
        axis = self._concat_along_axis
        samples: Dict[str, np.ndarray] = {}
        for env, buf in enumerate(self._buf):
            sample_idxes, element_idxes = np.nonzero(env_idxes == env)
            if len(sample_idxes) == 0:
                continue
            # All the elements of this environment at once, along the batch axis of a single sample
            env_samples = buf.sample(
                batch_size=len(sample_idxes),
                sample_next_obs=sample_next_obs,
                clone=clone,
                n_samples=1,
                **kwargs,
            )
            for k, v in env_samples.items():
                if k not in samples:
                    shape = list(v.shape)
                    shape[0], shape[axis] = n_samples, batch_size
                    samples[k] = np.empty(shape, dtype=v.dtype)
                # Write them where they were drawn: index the samples and the batch axis together
                np.moveaxis(samples[k], axis, 1)[sample_idxes, element_idxes] = np.moveaxis(v[0], axis - 1, 0)
        return samples

    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: str | torch.dtype = "cpu",
        from_numpy: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Sample elements from the replay buffer and convert them to torch tensors.

        Args:
            batch_size (int): Number of elements to sample.
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.
            clone (bool): whether to clone the sampled tensors.
            n_samples (int): the number of samples per batch_size to retrieve. Defaults to 1.
            dtype (Optional[torch.dtype], optional): the torch dtype to convert the arrays to. If None,
                then the dtypes of the numpy arrays is maintained. Defaults to None.
            device (str | torch.dtype, optional): the torch device to move the tensors to. Defaults to "cpu".
            from_numpy (bool, optional): whether to convert the numpy arrays to torch tensors
                with the 'torch.from_numpy' function. If False, then the numpy arrays are converted
                with the 'torch.as_tensor' function. Defaults to False.
            kwargs: additional keyword arguments to be passed to the 'self.sample' method.

        Returns:
            Dict[str, Tensor]: the sampled dictionary, containing the sampled array,
            one for every key, with a shape of [n_samples, sequence_length, batch_size, ...] if 'buffer_cls' is a
            'SequentialReplayBuffer', otherwise [n_samples, batch_size, ...] if 'buffer_cls' is a 'ReplayBuffer'.
        """
        samples = self.sample(
            batch_size=batch_size,
            sample_next_obs=sample_next_obs,
            clone=clone,
            n_samples=n_samples,
            **kwargs,
        )
        return {
            k: get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy) for k, v in samples.items()
        }


class EpisodeBuffer:
    """A replay buffer of whole episodes. The episodes of every key are stored one after the other in a circular
    storage of `buffer_size` steps: the oldest episodes make room for the new ones, which continue from the start of
    the storage when they reach its end. When memory-mapped, the storage of every key is a file in `memmap_dir`,
    whatever the number of episodes; in memory, it grows with the episodes up to `buffer_size` steps.

    Args:
        buffer_size (int): The capacity of the buffer.
        minimum_episode_length (int): The length of the sequences of the samples: the shorter episodes, and the ones
            longer than the buffer, are skipped with a warning.
        n_envs (int): The number of environments.
            Default to 1.
        obs_keys (Sequence[str]): The observations keys to store in the buffer.
            Default to ("observations",).
        prioritize_ends (bool): Whether to prioritize the ends of the episodes when sampling.
            Default to False.
        memmap (bool): Whether to memory-mapping the buffer.
            Default to False.
        memmap_dir (str | os.PathLike, optional): The directory for the memmap.
            Default to None.
        memmap_mode (str, optional): memory-map mode.
            Possible values are: "r+", "w+", "c", "copyonwrite", "readwrite", "write".
            Defaults to "r+".
    """

    batch_axis: int = 2

    def __init__(
        self,
        buffer_size: int,
        minimum_episode_length: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        prioritize_ends: bool = False,
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if minimum_episode_length <= 0:
            raise ValueError(f"The sequence length must be greater than zero, got: {minimum_episode_length}")
        if buffer_size < minimum_episode_length:
            raise ValueError(
                "The sequence length must be lower than the buffer size, "
                f"got: bs = {buffer_size} and sl = {minimum_episode_length}"
            )
        self._n_envs = n_envs
        self._obs_keys = obs_keys
        self._buffer_size = buffer_size
        self._minimum_episode_length = minimum_episode_length
        self._prioritize_ends = prioritize_ends

        # One list for each environment that contains open episodes:
        # one open episode per environment
        self._open_episodes = [[] for _ in range(n_envs)]
        # Contain the cumulative length of the episodes in the buffer
        self._cum_lengths: Sequence[int] = []
        # The storage of every key, created with the first episode, the position in it of every stored episode and
        # the position of the next one
        self._storage: Dict[str, np.ndarray | MemmapArray] = {}
        self._starts: List[int] = []
        self._pos = 0

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

    @property
    def prioritize_ends(self) -> bool:
        return self._prioritize_ends

    @prioritize_ends.setter
    def prioritize_ends(self, prioritize_ends: bool) -> None:
        self._prioritize_ends = prioritize_ends

    @property
    def buffer(self) -> Sequence[Dict[str, np.ndarray]]:
        """The stored episodes, from the oldest one: dictionaries of arrays of shape [episode_length, ...], views of
        the storage, or copies for the episodes that continue from its end to its start."""
        lengths = np.diff(self._cum_lengths, prepend=0)
        episodes = []
        for start, length in zip(self._starts, lengths):
            ranges = self._ranges(start, length)
            episodes.append(
                {
                    k: v[ranges[0]] if len(ranges) == 1 else np.concatenate([v[r] for r in ranges], axis=0)
                    for k, v in self._storage.items()
                }
            )
        return episodes

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
    def minimum_episode_length(self) -> int:
        return self._minimum_episode_length

    @property
    def is_memmap(self) -> bool:
        return self._memmap

    @property
    def full(self) -> bool:
        return len(self) + self._minimum_episode_length > self._buffer_size if len(self._cum_lengths) > 0 else False

    def __len__(self) -> int:
        return self._cum_lengths[-1] if len(self._cum_lengths) > 0 else 0

    @property
    def _capacity(self) -> int:
        """The number of steps of the storage: `buffer_size`, or fewer while an in-memory storage grows."""
        return next(iter(self._storage.values())).shape[0] if len(self._storage) > 0 else 0

    def _ranges(self, start: int, length: int) -> Sequence[slice]:
        """The slices of the storage of the `length` steps from `start`: two when they continue from its end."""
        end = start + length
        if end <= self._capacity:
            return [slice(start, end)]
        return [slice(start, self._capacity), slice(0, end - self._capacity)]

    @typing.overload
    def add(
        self, data: "ReplayBuffer", env_idxes: Sequence[int] | None = None, validate_args: bool = False
    ) -> None: ...

    @typing.overload
    def add(
        self,
        data: Dict[str, np.ndarray],
        env_idxes: Sequence[int] | None = None,
        validate_args: bool = False,
    ) -> None: ...

    def add(
        self,
        data: "ReplayBuffer" | Dict[str, np.ndarray],
        env_idxes: Sequence[int] | None = None,
        validate_args: bool = False,
    ) -> None:
        """Add data to the replay buffer in episodes. If data is a dictionary, then the keys must be strings
        and the values must be numpy arrays of shape [sequence_length, n_envs, ...].

        Args:
            data (ReplayBuffer | Dict[str, np.ndarray]]): data to add.
            env_idxes (Sequence[int], optional): the indices of the environments in which to add the data.
                Default to None.
            validate_args (bool): whether to validate the arguments or not.
                Default to None.
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

            if "terminated" not in data and "truncated" not in data:
                raise RuntimeError(
                    f"The episode must contain the `terminated` and the `truncated` keys, got: {data.keys()}"
                )

            if env_idxes is not None and (np.array(env_idxes) >= self._n_envs).any():
                raise ValueError(
                    f"The indices of the environment must be integers in [0, {self._n_envs}), given {env_idxes}"
                )

        # For each environment
        if env_idxes is None:
            env_idxes = range(self._n_envs)
        for i, env in enumerate(env_idxes):
            # Take the data from a single environment
            env_data = {k: v[:, i] for k, v in data.items()}
            done = np.logical_or(env_data["terminated"], env_data["truncated"])
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
                    episode = {k: env_data[k][start : stop + 1] for k in env_data.keys()}
                    # If the episode length is greater than zero, then add it to the open episode
                    # of the corresponding environment.
                    if len(np.logical_or(episode["terminated"], episode["truncated"])) > 0:
                        self._open_episodes[env].append(episode)
                    start = stop + 1
                    # If the open episode is not empty and the last element is a done, then save the episode
                    # in the buffer and clear the open episode
                    should_save = len(self._open_episodes[env]) > 0 and np.logical_or(
                        self._open_episodes[env][-1]["terminated"][-1], self._open_episodes[env][-1]["truncated"][-1]
                    )
                    if should_save:
                        self._save_episode(self._open_episodes[env])
                        self._open_episodes[env] = []

    def _save_episode(self, episode_chunks: Sequence[Dict[str, np.ndarray | MemmapArray]]) -> None:
        if len(episode_chunks) == 0:
            raise RuntimeError("Invalid episode, an empty sequence is given. You must pass a non-empty sequence.")
        # Concatenate all the chunks of the episode
        episode = {k: [] for k in episode_chunks[0].keys()}
        for chunk in episode_chunks:
            for k in chunk.keys():
                episode[k].append(chunk[k])
        episode = {k: np.concatenate(v, axis=0) for k, v in episode.items()}

        # Control the validity of the episode
        ends = np.logical_or(episode["terminated"], episode["truncated"])
        ep_len = ends.shape[0]
        if len(ends.nonzero()[0]) != 1 or ends[-1] != 1:
            raise RuntimeError(f"The episode must contain exactly one done, got: {len(np.nonzero(ends))}")
        # As DreamerV2 does, the episodes shorter than the sampled sequences (they cannot be sampled) are skipped, and
        # so are the ones longer than the buffer (they cannot be stored)
        if ep_len < self._minimum_episode_length or ep_len > self._buffer_size:
            warnings.warn(
                f"Skipping the episodes shorter than {self._minimum_episode_length} steps "
                f"or longer than {self._buffer_size} steps (the buffer size)"
            )
            return
        if len(self._storage) > 0 and episode.keys() != self._storage.keys():
            raise RuntimeError(
                f"Every episode must have the same keys: the buffer holds {list(self._storage.keys())}, "
                f"got: {list(episode.keys())}"
            )

        # If the buffer is full, then remove the oldest episodes: their steps make room for the new one
        if self.full or len(self) + ep_len > self._buffer_size:
            # Compute the index of the last episode to remove
            cum_lengths = np.array(self._cum_lengths)
            mask = (len(self) - cum_lengths + ep_len) <= self._buffer_size
            last_to_remove = mask.argmax()
            self._starts = self._starts[last_to_remove + 1 :]
            # Update the cum_lengths lists
            cum_lengths = cum_lengths[last_to_remove + 1 :] - cum_lengths[last_to_remove]
            self._cum_lengths = cum_lengths.tolist()
        self._store(episode, ep_len)

    def _store(self, episode: Dict[str, np.ndarray | MemmapArray], ep_len: int) -> None:
        """Write the episode in the storage, after the newest one: the buffer has room for it."""
        if len(self._storage) == 0:
            # A memory-mapped storage holds the whole buffer from the start (its file grows on the disk as it is
            # written), while an in-memory one grows with the episodes
            for k, v in episode.items():
                shape = (self._buffer_size if self._memmap else ep_len, *v.shape[1:])
                if self._memmap:
                    self._storage[k] = MemmapArray(
                        filename=Path(self._memmap_dir / f"{k}.memmap"),
                        dtype=v.dtype,
                        shape=shape,
                        mode=self._memmap_mode,
                    )
                else:
                    self._storage[k] = np.empty(shape, dtype=v.dtype)
        elif self._capacity < self._buffer_size and self._pos + ep_len > self._capacity:
            # The storage only grows before it holds `buffer_size` steps: until then, no episode has been removed,
            # and the episodes are in its first `self._pos` steps
            capacity = min(self._buffer_size, max(2 * self._capacity, self._pos + ep_len))
            for k, v in self._storage.items():
                self._storage[k] = np.empty((capacity, *v.shape[1:]), dtype=v.dtype)
                self._storage[k][: v.shape[0]] = v
        start = self._pos % self._capacity
        for k, v in episode.items():
            written = 0
            for steps in self._ranges(start, ep_len):
                self._storage[k][steps] = v[written : written + steps.stop - steps.start]
                written += steps.stop - steps.start
        self._starts.append(start)
        self._cum_lengths.append(len(self) + ep_len)
        self._pos = start + ep_len

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Up to sheeprl 0.7.0 every episode had its own arrays (memory-mapped to a directory of its own): the
        # episodes of those buffers are copied into the storage, while their files are left where they are
        episodes = state.pop("_buf", None)
        self.__dict__.update(state)
        if episodes is not None:
            self._storage, self._starts, self._pos = {}, [], 0
            lengths = np.diff(self._cum_lengths, prepend=0)
            self._cum_lengths = []
            for episode, ep_len in zip(episodes, lengths):
                # Every file is opened only to be copied: the episodes could be more than the files a process can open
                episode = {
                    k: (
                        np.memmap(v.filename, dtype=v.dtype, shape=v.shape, mode="r")
                        if isinstance(v, MemmapArray)
                        else v
                    )
                    for k, v in episode.items()
                }
                self._store(episode, ep_len)

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        n_samples: int = 1,
        clone: bool = False,
        sequence_length: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample trajectories from the replay buffer.

        Args:
            batch_size (int): Number of element in the batch.
            sample_next_obs (bool): Whether to sample the next obs.
                Default to False.
            n_samples (bool): The number of samples per batch_size to be retrieved.
                Defaults to 1.
            clone (bool): Whether to clone the samples.
                Default to False.
            sequence_length (int): The length of the sequences to sample.
                Default to 1.

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of
            [n_samples, sequence_length, batch_size, ...].
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be greater than 0, got: {batch_size}")
        if n_samples <= 0:
            raise ValueError(f"The number of samples must be greater than 0, got: {n_samples}")
        lengths = np.diff(self._cum_lengths, prepend=0)
        if sample_next_obs:
            valid_episodes = np.flatnonzero(lengths > sequence_length)
        else:
            valid_episodes = np.flatnonzero(lengths >= sequence_length)
        if len(valid_episodes) == 0:
            raise RuntimeError(
                "No valid episodes has been added to the buffer. Please add at least one episode of length greater "
                f"than or equal to {sequence_length} calling `self.add()`"
            )

        # The episode of every sequence, drawn independently: each of the `n_samples` batches is a uniform sample
        episodes = valid_episodes[np.random.randint(0, len(valid_episodes), (batch_size * n_samples,))]
        ep_lens = lengths[episodes] - 1 if sample_next_obs else lengths[episodes]
        # Define the maximum index that can be sampled in the episodes
        upper = ep_lens - sequence_length + 1
        # If you want to prioritize ends, then all the indices of the episode
        # can be sampled as starting index
        if self._prioritize_ends:
            upper += sequence_length
        # Sample the starting indices and upper bound with `ep_len - sequence_length`
        start_idxes = np.minimum(np.random.randint(0, upper), ep_lens - sequence_length, dtype=np.intp)
        # The steps of the sequences in the storage, where an episode can continue from its end to its start
        first_steps = np.array(self._starts, dtype=np.intp)[episodes] + start_idxes
        indices = (first_steps.reshape(-1, 1) + np.arange(sequence_length, dtype=np.intp)) % self._capacity
        samples = {}
        for k, v in self._storage.items():
            array = v.array if isinstance(v, MemmapArray) else v
            samples[k] = np.take(array, indices.flat, axis=0)
            if sample_next_obs and k in self._obs_keys:
                samples[f"next_{k}"] = np.take(array, ((indices + 1) % self._capacity).flat, axis=0)
        # Reshape the sequences to [n_samples, sequence_length, batch_size, ...]
        for k, v in samples.items():
            samples[k] = np.moveaxis(v.reshape(n_samples, batch_size, sequence_length, *v.shape[1:]), 2, 1)
            if clone:
                samples[k] = samples[k].copy()
        return samples

    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        n_samples: int = 1,
        clone: bool = False,
        sequence_length: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: str | torch.dtype = "cpu",
        from_numpy: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Sample elements from the replay buffer and convert them to torch tensors.

        Args:
            batch_size (int): Number of elements to sample.
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.
            clone (bool): whether to clone the sampled tensors.
            n_samples (int): the number of samples per batch_size. Defaults to 1.
            sequence_length (int): the length of the sequence of each element. Defaults to 1.
            dtype (Optional[torch.dtype], optional): the torch dtype to convert the arrays to. If None,
                then the dtypes of the numpy arrays is maintained. Defaults to None.
            device (str | torch.dtype, optional): the torch device to move the tensors to. Defaults to "cpu".
            from_numpy (bool, optional): whether to convert the numpy arrays to torch tensors
                with the 'torch.from_numpy' function. If False, then the numpy arrays are converted
                with the 'torch.as_tensor' function. Defaults to False.
            kwargs: additional keyword arguments to be passed to the 'self.sample' method.
        """
        samples = self.sample(batch_size, sample_next_obs, n_samples, clone, sequence_length)
        return {
            k: get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy) for k, v in samples.items()
        }


def get_tensor(
    array: np.ndarray | MemmapArray,
    dtype: Optional[torch.dtype] = None,
    clone: bool = False,
    device: str | torch.dtype = "cpu",
    from_numpy: bool = False,
) -> Tensor:
    if isinstance(array, MemmapArray):
        array = array.array
    if clone:
        array = array.copy()
    if from_numpy:
        torch_v = torch.from_numpy(array).to(
            dtype=NUMPY_TO_TORCH_DTYPE_DICT[array.dtype] if dtype is None else dtype,
            device=device,
        )
    else:
        torch_v = torch.as_tensor(
            array,
            dtype=NUMPY_TO_TORCH_DTYPE_DICT[array.dtype] if dtype is None else dtype,
            device=device,
        )
    return torch_v
