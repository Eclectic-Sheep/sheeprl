from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from sheeprl.utils.memmap import MemmapArray
from sheeprl.utils.utils import NUMPY_TO_TORCH_DTYPE_DICT, TORCH_TO_NUMPY_DTYPE_DICT


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """A standard replay buffer implementation. Internally this is represented by a
        dictionary mapping string to numpy arrays. The first dimension of the arrays is the
        buffer size, while the second dimension is the number of environments. With `device`, the arrays are tensors
        in the memory of that device (e.g. a GPU), gathered there (`gather`): the steps are copied there as they are
        added.

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
            device (str | torch.device | None, optional): the device whose memory holds the buffer, which is then not
                memory-mapped: when its storage is created (at the first `add`), it must fit in the free memory of the
                device. Defaults to None (NumPy arrays in the memory of the CPU).
            kwargs: additional keyword arguments.
        """
        if device is not None and memmap:
            raise ValueError("A replay buffer in the memory of a device can't be memory-mapped")
        self._device = torch.device(device) if device is not None else None
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
        # The row of the next step, whether every row was written, and the steps added, of every environment
        self._env_pos = np.zeros(n_envs, dtype=np.int64)
        self._env_full = np.zeros(n_envs, dtype=bool)
        self._env_added = np.zeros(n_envs, dtype=np.int64)
        self._memmap_specs = {}

    @property
    def buffer(self) -> Dict[str, np.ndarray]:
        return self._buf

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def full(self) -> bool:
        """Whether every row of every environment was written."""
        return bool(self._env_full.all())

    @property
    def positions(self) -> np.ndarray:
        """The row of the next step of every environment."""
        return self._env_pos.copy()

    @property
    def env_full(self) -> np.ndarray:
        """Whether every row of every environment was written."""
        return self._env_full.copy()

    @property
    def env_added(self) -> np.ndarray:
        """The steps added to every environment."""
        return self._env_added.copy()

    @property
    def lockstep(self) -> bool:
        """Whether the environments are at the same row: always, unless steps are added to some of them only."""
        return bool((self._env_pos == self._env_pos[0]).all() and (self._env_full == self._env_full[0]).all())

    def _lockstep(self, value: np.ndarray) -> Any:
        if not self.lockstep:
            raise RuntimeError(
                "The environments of the buffer are at different rows (steps are added to some of them only): "
                "read `positions`, `env_full` and `env_added`"
            )
        return value[0].item()

    @property
    def _pos(self) -> int:
        """The row of the next step of the environments, in lockstep."""
        return self._lockstep(self._env_pos)

    @property
    def _full(self) -> bool:
        return self._lockstep(self._env_full)

    @property
    def _added(self) -> int:
        return self._lockstep(self._env_added)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def empty(self) -> bool:
        return (self.buffer is not None and len(self.buffer) == 0) or self.buffer is None

    @property
    def is_memmap(self) -> bool:
        return self._memmap

    @property
    def device(self) -> Optional[torch.device]:
        """The device whose memory holds the buffer, `None` for the memory of the CPU."""
        return self._device

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
        self._apply_checkpoint_truncation()
        buf = {}
        for k, v in self.buffer.items():
            buf[k] = get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy)
        return buf

    def add(
        self,
        data: "ReplayBuffer" | Dict[str, np.ndarray],
        env_idxes: Optional[Sequence[int]] = None,
        validate_args: bool = False,
    ) -> None:
        """Add data to the replay buffer. If the replay buffer is full, then the oldest data is overwritten.
        If data is a dictionary, then the keys must be strings and the values must be numpy arrays of shape
        [sequence_length, n_envs, ...].

        Args:
            data (ReplayBuffer | Dict[str, np.ndarray]): the data to add to the replay buffer.
            env_idxes (Sequence[int], optional): the environments of the columns of the data, written at their own
                rows (e.g. the first steps of the environments that ended an episode). Defaults to None (all of them).
            validate_args (bool, optional): whether to validate the arguments. Defaults to False.

        Raises:
            ValueError: if the data is not a dictionary containing numpy arrays.
            ValueError: if the data is not a dictionary containing numpy arrays.
            RuntimeError: if the data does not have at least 2 dimensions.
            RuntimeError: if the data is not congruent in the first 2 dimensions.
        """
        if isinstance(data, ReplayBuffer):
            data = data.buffer
        envs = None if env_idxes is None else np.asarray(env_idxes, dtype=np.intp).reshape(-1)
        if envs is not None:
            if len(envs) != next(iter(data.values())).shape[1]:
                raise ValueError(
                    f"The length of 'env_idxes' ({len(envs)}) must be equal to the second dimension of the "
                    f"arrays in 'data' ({next(iter(data.values())).shape[1]})"
                )
            if len(np.unique(envs)) != len(envs) or (envs < 0).any() or (envs >= self._n_envs).any():
                raise ValueError(f"The environments must be distinct integers in [0, {self._n_envs}), got {env_idxes}")
            if len(envs) == self._n_envs and (envs == np.arange(self._n_envs)).all():
                envs = None
        self._check_add(data, validate_args, n_envs=None if envs is None else len(envs))
        self._apply_checkpoint_truncation()
        data_len = next(iter(data.values())).shape[0]
        # Only the last `buffer_size` steps are kept, in the rows they would take if added one at a time: the step of
        # index `i` of the data goes in the row `(pos + i) % buffer_size` of its environment
        kept = min(data_len, self._buffer_size)
        data_to_store = {k: v[-self._buffer_size :] for k, v in data.items()} if data_len > kept else data
        steps = (data_len - kept) + np.arange(kept)
        written = np.arange(self._n_envs) if envs is None else envs
        if envs is None and self.lockstep:
            # The rows of every environment
            idxes = (self._env_pos[0] + steps) % self._buffer_size
        else:
            # The rows of every environment written: [Steps, Environments]
            idxes = (self._env_pos[written][np.newaxis] + steps[:, np.newaxis]) % self._buffer_size
        if self.empty:
            self._allocate(data_to_store)
        if self._device is not None:
            # Normal tensors, also when the steps are played in inference mode: the buffer is changed outside of it too
            with torch.inference_mode(False):
                index = torch.as_tensor(idxes, device=self._device)
                if idxes.ndim == 2:
                    index = (index, torch.as_tensor(written, device=self._device))
                for k, v in data_to_store.items():
                    # Cast to the dtype of the buffer, as NumPy does
                    value = torch.as_tensor(v, device=self._device).to(self.buffer[k].dtype)
                    # The leading dimensions of size one are dropped, as NumPy does when it broadcasts (`_check_add`)
                    while value.dim() > self.buffer[k].dim() and value.shape[0] == 1:
                        value = value[0]
                    self.buffer[k][index] = value
        else:
            index = idxes if idxes.ndim == 1 else (idxes, written)
            for k, v in data_to_store.items():
                self.buffer[k][index] = v
        self._env_full[written] |= self._env_pos[written] + data_len >= self._buffer_size
        self._env_pos[written] = (self._env_pos[written] + data_len) % self._buffer_size
        self._env_added[written] += data_len

    def _allocate(self, data: Dict[str, np.ndarray]) -> None:
        """Create the arrays of the buffer, for the keys of `data`: memory-mapped, in the memory of the CPU or in the
        one of the device."""
        if self._device is not None:
            # Normal tensors, also when the steps are played in inference mode
            with torch.inference_mode(False):
                self._allocate_on_device(data)
        elif self._memmap:
            for k, v in data.items():
                self.buffer[k] = MemmapArray(
                    filename=Path(self._memmap_dir / f"{k}.memmap"),
                    dtype=v.dtype,
                    shape=(self._buffer_size, self._n_envs, *v.shape[2:]),
                    mode=self._memmap_mode,
                )
        else:
            for k, v in data.items():
                self.buffer[k] = np.empty(shape=(self._buffer_size, self._n_envs, *v.shape[2:]), dtype=v.dtype)

    def to(self, device: str | torch.device | None) -> "ReplayBuffer":
        """The buffer in the memory of `device` (`None`: NumPy arrays in the memory of the CPU), e.g. a buffer loaded
        from a checkpoint (on the CPU) in a run that keeps it on a device, or the other way around. A memory-mapped
        buffer moved to a device is no longer memory-mapped."""
        device = torch.device(device) if device is not None else None
        if not self.empty and device is not None:
            needed = sum(int(np.prod(v.shape)) * np.dtype(_numpy_dtype(v)).itemsize for v in self._buf.values())
            _check_free_memory(device, needed)
        with torch.inference_mode(False):
            for k, v in self._buf.items():
                if device is None:
                    self._buf[k] = v.cpu().numpy() if torch.is_tensor(v) else v
                else:
                    self._buf[k] = torch.as_tensor(v.array if isinstance(v, MemmapArray) else v).to(device)
        if device is not None:
            self._memmap = False
        self._device = device
        return self

    def _allocate_on_device(self, data: Dict[str, np.ndarray]) -> None:
        """Create the tensors of the buffer on its device, for the keys of `data`, if they fit in its free memory."""
        specs = {
            k: ((self._buffer_size, self._n_envs, *np.shape(v)[2:]), NUMPY_TO_TORCH_DTYPE_DICT[np.asarray(v).dtype])
            for k, v in data.items()
        }
        needed = sum(
            int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size() for shape, dtype in specs.values()
        )
        _check_free_memory(self._device, needed)
        for k, (shape, dtype) in specs.items():
            self._buf[k] = torch.empty(shape, dtype=dtype, device=self._device)

    def _check_add(
        self, data: Dict[str, np.ndarray], validate_args: bool = False, n_envs: Optional[int] = None
    ) -> None:
        """Raise the errors that adding `data` would raise, before anything is written: an add that fails leaves the
        buffer as it was."""
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
        rows = min(data_len, self._buffer_size)
        for k, v in data.items():
            # The columns of the environments written (all of them by default)
            columns = self._n_envs if n_envs is None else n_envs
            if self.empty:
                target = (rows, columns, *np.shape(v)[2:])
            elif k not in self._buf:
                raise KeyError(f"The buffer has no key '{k}': its keys are {list(self._buf.keys())}")
            else:
                target = (rows, columns, *self._buf[k].shape[2:])
            # The rows written are the last `buffer_size` ones, broadcast as numpy does (also dropping the leading
            # dimensions of size one)
            shape = np.shape(v)
            if data_len > self._buffer_size:
                shape = (min(shape[0], self._buffer_size), *shape[1:])
            while len(shape) > len(target) and shape[0] == 1:
                shape = shape[1:]
            try:
                fits = np.broadcast_shapes(shape, target) == target
            except ValueError:
                fits = False
            if not fits:
                raise ValueError(
                    f"The data of the key '{k}', of shape {np.shape(v)}, cannot be written in rows of shape {target}"
                )

    def _apply_checkpoint_truncation(self) -> None:
        """A memory-mapped buffer is checkpointed by reference to its files (`CheckpointCallback`): the truncation of
        its last step, made for the checkpoint, is undone on the files after it. The loaded buffer writes it again when
        it is used (added to, sampled or read), not when it is loaded: the checkpoints are also loaded only to be
        evaluated."""
        if self.__dict__.get("_checkpoint_truncation") and not self.empty:
            self._checkpoint_truncation = False
            self._buf["truncated"][(self._env_pos - 1) % self._buffer_size, np.arange(self._n_envs)] = 1

    def gather(
        self,
        rows: np.ndarray,
        env_idxes: np.ndarray,
        sequence_length: Optional[int] = None,
        sample_next_obs: bool = False,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        """The steps at the rows `rows` of the environments `env_idxes`, of shape `[len(rows), ...]`, or, with
        `sequence_length`, the sequences of that many steps that start there, of shape
        `[len(rows), sequence_length, ...]`. With `sample_next_obs`, the steps after them too, as `next_<key>`: of the
        observation keys for the steps, of every key for the sequences."""
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        self._apply_checkpoint_truncation()
        if self._device is not None:
            return self._gather_on_device(rows, env_idxes, sequence_length, sample_next_obs)
        if sequence_length is None:
            samples: Dict[str, np.ndarray] = {}
            flattened_idxes = (rows * self.n_envs + env_idxes).flat
            if sample_next_obs:
                flattened_next_idxes = (((rows + 1) % self._buffer_size) * self.n_envs + env_idxes).flat
            for k, v in self.buffer.items():
                samples[k] = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_idxes, axis=0)
                if clone:
                    samples[k] = samples[k].copy()
                if k in self._obs_keys and sample_next_obs:
                    samples[f"next_{k}"] = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_next_idxes, axis=0)
                    if clone:
                        samples[f"next_{k}"] = samples[f"next_{k}"].copy()
            return samples
        # The rows of every sequence, one after the other: (b1_s1, b1_s2, ..., bn_s1, bn_s2, ...), where bm_sk is the
        # k-th step of the m-th sequence, each from a single environment
        chunk_length = np.arange(sequence_length, dtype=np.intp).reshape(1, -1)
        flattened_rows = np.ravel((rows.reshape(-1, 1) + chunk_length) % self.buffer_size)
        env_idxes = np.repeat(env_idxes, sequence_length)
        flattened_idxes = (flattened_rows * self._n_envs + env_idxes).flat
        shape = (len(rows), sequence_length)
        samples = {}
        for k, v in self.buffer.items():
            flattened_v = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_idxes, axis=0)
            samples[k] = np.reshape(flattened_v, shape + flattened_v.shape[1:])
            if clone:
                samples[k] = samples[k].copy()
            if sample_next_obs:
                flattened_next_v = v[(flattened_rows + 1) % self._buffer_size, env_idxes]
                samples[f"next_{k}"] = np.reshape(flattened_next_v, shape + flattened_next_v.shape[1:])
                if clone:
                    samples[f"next_{k}"] = samples[f"next_{k}"].copy()
        return samples

    def _gather_on_device(
        self, rows: np.ndarray, env_idxes: np.ndarray, sequence_length: Optional[int], sample_next_obs: bool
    ) -> Dict[str, Tensor]:
        """`gather` on the device: the same steps, as tensors (new ones, as with `clone`)."""
        rows = torch.as_tensor(rows, device=self._device)
        env_idxes = torch.as_tensor(env_idxes, device=self._device)
        if sequence_length is not None:
            rows = (rows.reshape(-1, 1) + torch.arange(sequence_length, device=self._device)) % self._buffer_size
            env_idxes = env_idxes.reshape(-1, 1).expand_as(rows)
        next_rows = (rows + 1) % self._buffer_size
        samples = {}
        for k, v in self.buffer.items():
            samples[k] = v[rows, env_idxes]
            # As `gather`: the next steps of the observation keys for the steps, of every key for the sequences
            if sample_next_obs and (sequence_length is not None or k in self._obs_keys):
                samples[f"next_{k}"] = v[next_rows, env_idxes]
        return samples

    def __getitem__(self, key: str) -> np.ndarray | np.memmap | MemmapArray:
        if not isinstance(key, str):
            raise TypeError("'key' must be a string")
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        self._apply_checkpoint_truncation()
        return self.buffer.get(key)

    def __setitem__(self, key: str, value: np.ndarray | np.memmap | MemmapArray) -> None:
        if self._device is not None and isinstance(value, (np.ndarray, MemmapArray, Tensor)):
            if self.empty:
                raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
            if tuple(value.shape[:2]) != (self._buffer_size, self._n_envs):
                raise RuntimeError(
                    "'value' must have at least two dimensions of dimension [buffer_size, n_envs, ...]. "
                    f"Shape of 'value' is {value.shape}"
                )
            value = value.array if isinstance(value, MemmapArray) else value
            with torch.inference_mode(False):
                self.buffer.update({key: torch.as_tensor(value, device=self._device).clone()})
            return
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


def get_tensor(
    array: np.ndarray | MemmapArray,
    dtype: Optional[torch.dtype] = None,
    clone: bool = False,
    device: str | torch.dtype = "cpu",
    from_numpy: bool = False,
) -> Tensor:
    if isinstance(array, Tensor):
        # Already a tensor (of a buffer in the memory of a device)
        array = array.to(device=device, dtype=dtype)
        return array.clone() if clone else array
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


def _numpy_dtype(array: Any) -> np.dtype:
    return TORCH_TO_NUMPY_DTYPE_DICT[array.dtype] if torch.is_tensor(array) else array.dtype


def _check_free_memory(device: torch.device, needed: int) -> None:
    """Raise if `needed` bytes don't fit in the free memory of the GPU `device`."""
    if device.type == "cuda":
        free = torch.cuda.mem_get_info(device)[0]
        if needed > free:
            raise RuntimeError(
                f"The replay buffer needs {needed / 2**30:.2f} GB on {device}, but {free / 2**30:.2f} GB are "
                "free: keep it in the memory of the CPU (`buffer.on_device=False`) or make it smaller (`buffer.size`)"
            )
