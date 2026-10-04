from __future__ import annotations

import os
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
import torch
from torch import Tensor

from sheeprl.data.samplers import (
    EnvIndependentSampler,
    EpisodeQueue,
    EpisodeSampler,
    OnlineQueue,
    SequenceSampler,
    TransitionSampler,
)
from sheeprl.utils.memmap import MemmapArray
from sheeprl.utils.utils import NUMPY_TO_TORCH_DTYPE_DICT, TORCH_TO_NUMPY_DTYPE_DICT


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
            seed (int | np.random.SeedSequence | None, optional): the seed of the random number generator
                used to sample from the buffer. Defaults to None.
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
        # The generator and the online queue of `sample`: the samplers of a `ReplayStore` have their own
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._online = OnlineQueue()

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

    def env_view(self, env: int) -> "_EnvView":
        """The environment `env` as a buffer of one environment, for the samplers of the environments (its rows are
        gathered from this buffer)."""
        return _EnvView(self, env)

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

    @typing.overload
    def add(self, data: "ReplayBuffer", validate_args: bool = False) -> None: ...

    @typing.overload
    def add(self, data: Dict[str, np.ndarray], validate_args: bool = False) -> None: ...

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
        if isinstance(env_idxes, bool):
            # Up to sheeprl 0.8.2 the second argument was `validate_args`
            env_idxes, validate_args = None, env_idxes
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

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        online: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the replay buffer (`TransitionSampler`, with the generator and the online queue of the
        buffer). If the replay buffer is not full, then the samples are taken from the first 'self.pos' elements.
        Otherwise, the samples are taken from all the elements. When 'sample_next_obs' is True we sample until
        'self.pos - 1' to avoid sampling the last observation, which would be invalid.
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        Args:
            batch_size (int): Number of element to sample
            sample_next_obs (bool): whether to sample the next observations from the 'self.obs_keys' keys.
                Defaults to False.
            clone (bool): whether to clone the sampled numpy arrays. Defaults to False.
            n_samples (int): the number of samples to perform. Defaults to 1.
            online (bool): whether the samples start with the elements of the online queue, the oldest first, and only
                the rest of them is sampled uniformly: every element added is sampled once soon after (see
                `OnlineQueue`). Defaults to False.

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of [n_samples, batch_size, ...].
        """
        sampler = TransitionSampler(sample_next_obs, online, rng=self._rng, queue=self._online)
        return sampler.sample(self, batch_size, n_samples, clone=clone)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if "_env_pos" not in state:
            # Up to sheeprl 0.8.2 the environments were written in lockstep, at the row `_pos`. Up to sheeprl 0.8.0 the
            # buffers didn't count the steps added: only their remainder by the buffer size, the position of the next
            # one, matters
            pos, full = state.pop("_pos"), state.pop("_full")
            added = state.pop("_added", pos + (state["_buffer_size"] if full else 0))
            state["_env_pos"] = np.full(state["_n_envs"], pos, dtype=np.int64)
            state["_env_full"] = np.full(state["_n_envs"], full, dtype=bool)
            state["_env_added"] = np.full(state["_n_envs"], added, dtype=np.int64)
        # Up to sheeprl 0.8.2 the buffers were in the memory of the CPU
        state.setdefault("_device", None)
        # The online queue isn't checkpointed: it restarts empty, with the steps added after the loading (up to sheeprl
        # 0.8.2 it was kept in `_online_origin` and `_online_next`)
        state.pop("_online_origin", None)
        state.pop("_online_next", None)
        state["_online"] = OnlineQueue(state["_env_added"].copy())
        self.__dict__.update(state)

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
        online: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries (`SequenceSampler`, with the generator and the online queue of the buffer).

        Args:
            batch_size (int): Number of element to sample
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.
            clone (bool): whether to clone the sampled tensors.
            n_samples (int): the number of samples to perform. Defaults to 1.
            sequence_length (int): the length of the sequence of each element. Defaults to 1.
            online (bool): whether the samples start with the sequences of the online queue, the oldest first, and only
                the rest of them is sampled uniformly: every step added is sampled once soon after (see
                `OnlineQueue`). Defaults to False.

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of
            [n_samples, sequence_length, batch_size, ...].
        """
        sampler = SequenceSampler(sequence_length, sample_next_obs, online, rng=self._rng, queue=self._online)
        return sampler.sample(self, batch_size, n_samples, clone=clone)


class EnvIndependentReplayBuffer(ReplayBuffer):
    """A `ReplayBuffer` sampled by `EnvIndependentSampler`: every step or sequence of its samples comes from a single
    environment, drawn independently, then from the rows of that environment (written at its own row: see the
    `env_idxes` of `add`).

    It is kept for its `sample` and for the checkpoints of sheeprl up to 0.8.2, which hold a buffer per environment
    (converted to a single storage when they are loaded).

    Args:
        buffer_size (int): the steps of every environment.
        n_envs (int, optional): the number of environments. Defaults to 1.
        obs_keys (Sequence[str], optional): names of the observation keys. Those are used
            to sample the next-observation. Defaults to ("observations",).
        memmap (bool, optional): whether to memory-map the numpy arrays saved in the buffer. Defaults to False.
        memmap_dir (str | os.PathLike | None, optional): the memory-mapped files directory.
            Defaults to None.
        memmap_mode (str, optional): memory-map mode. Possible values are: "r+", "w+", "c", "copyonwrite",
            "readwrite", "write". Defaults to "r+".
        buffer_cls (Type[ReplayBuffer], optional): how `sample` reads the environments: steps (`ReplayBuffer`) or
            sequences (`SequentialReplayBuffer`). Defaults to ReplayBuffer.
        seed (int | np.random.SeedSequence | None, optional): the seed from which the independent random number
            generators of `sample` (the one of the environments and the one of every environment) are derived.
            Defaults to None.
        kwargs: additional keyword arguments of `ReplayBuffer` (`device`).
    """

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
        if n_envs <= 0:
            raise ValueError(f"The number of environments must be greater than zero, got: {n_envs}")
        seed_sequences = np.random.SeedSequence(seed).spawn(n_envs + 1)
        super().__init__(
            buffer_size, n_envs, obs_keys, memmap, memmap_dir, memmap_mode, seed=seed_sequences[-1], **kwargs
        )
        # The generators and the online queues of the environments in `sample`
        self._env_rngs = [np.random.default_rng(s) for s in seed_sequences[:-1]]
        self._env_online = [OnlineQueue() for _ in range(n_envs)]
        self._concat_along_axis = buffer_cls.batch_axis

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        online: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Samples data from the buffer (`EnvIndependentSampler`, with the generators and the online queues of the
        buffer). The returned samples are sampled given the 'buffer_cls' class used to initialize the buffer:
        sequences for a `SequentialReplayBuffer`, steps for a `ReplayBuffer`.

        Args:
            batch_size (int): The number of samples to draw from the buffer.
            sample_next_obs (bool): Whether to sample the next observation or the current observation.
            clone (bool): Whether to clone the data or return a reference to the original data.
            n_samples (int): The number of samples to draw for each batch element.
            online (bool): whether the samples start with the elements of the online queues of the environments, the
                oldest first (the ones that start first, in the order of their environments), and only the rest of
                them is sampled uniformly (see `OnlineQueue`). Defaults to False.
            **kwargs: Additional keyword arguments of the underlying buffer's `sample` method (`sequence_length`).

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of
            [n_samples, sequence_length, batch_size, ...] if 'buffer_cls' is a 'SequentialReplayBuffer',
            otherwise [n_samples, batch_size, ...] if 'buffer_cls' is a 'ReplayBuffer'.
        """
        sequence_length = kwargs.get("sequence_length", 1) if self._concat_along_axis == 2 else None
        env_samplers = [
            (
                TransitionSampler(sample_next_obs, rng=rng, queue=queue)
                if sequence_length is None
                else SequenceSampler(sequence_length, sample_next_obs, rng=rng, queue=queue)
            )
            for rng, queue in zip(self._env_rngs, self._env_online)
        ]
        sampler = EnvIndependentSampler(
            self._n_envs, sequence_length, sample_next_obs, online, rng=self._rng, env_samplers=env_samplers
        )
        return sampler.sample(self, batch_size, n_samples, clone=clone)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if isinstance(state.get("_buf"), list):
            state = _merge_env_buffers(state)
        super().__setstate__(state)
        # The online queues aren't checkpointed: they restart empty, with the steps added after the loading
        self._env_online = [OnlineQueue(self._env_added[i : i + 1].copy()) for i in range(self._n_envs)]


def _merge_env_buffers(state: Dict[str, Any]) -> Dict[str, Any]:
    """The state of an `EnvIndependentReplayBuffer` of sheeprl up to 0.8.2, which held a buffer per environment, as the
    one of a single storage: the arrays of the environments side by side (memory-mapped in the parent directory of
    the ones of the environments, if they were), with their rows and their generators."""
    buffers: List[ReplayBuffer] = state["_buf"]
    first = buffers[0]
    n_envs, buffer_size = len(buffers), state["_buffer_size"]
    for b in buffers:
        b._apply_checkpoint_truncation()
    memmap_dir = Path(first._memmap_dir).parent if first.is_memmap else None
    arrays: Dict[str, np.ndarray | MemmapArray] = {}
    filled = [b for b in buffers if not b.empty]
    for k, v in (filled[0].buffer.items() if filled else ()):
        shape = (buffer_size, n_envs, *v.shape[2:])
        if memmap_dir is not None:
            arrays[k] = MemmapArray(
                filename=memmap_dir / f"{k}.memmap", dtype=v.dtype, shape=shape, mode=first._memmap_mode
            )
        else:
            arrays[k] = np.empty(shape, dtype=v.dtype)
        for env, b in enumerate(buffers):
            if not b.empty:
                arrays[k][:, env] = np.asarray(b.buffer[k])[:, 0]
    return {
        "_buffer_size": buffer_size,
        "_n_envs": n_envs,
        "_obs_keys": first._obs_keys,
        "_memmap": memmap_dir is not None,
        "_memmap_dir": memmap_dir,
        "_memmap_mode": first._memmap_mode,
        "_buf": arrays,
        "_memmap_specs": {},
        "_rng": state["_rng"],
        "_device": None,
        "_env_pos": np.array([b._env_pos[0] for b in buffers], dtype=np.int64),
        "_env_full": np.array([b._env_full[0] for b in buffers], dtype=bool),
        "_env_added": np.array([b._env_added[0] for b in buffers], dtype=np.int64),
        "_env_rngs": [b._rng for b in buffers],
        "_concat_along_axis": state["_concat_along_axis"],
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
        # The episodes stored, and the online queue of `sample`: the samplers of a `ReplayStore` have their own
        self._stored = 0
        self._online = EpisodeQueue()

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
    def device(self) -> None:
        """The episodes are in the memory of the CPU."""
        return None

    def to(self, device: str | torch.device | None) -> "EpisodeBuffer":
        if device is not None:
            raise ValueError("The episode buffer is kept in the memory of the CPU")
        return self

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
            # No episode continues from the end of a storage smaller than `buffer_size` (it grows to hold the new one
            # where it starts), so the episodes are in its first `self._pos` steps and keep their positions
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
        self._stored += 1

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Up to sheeprl 0.7.0 every episode had its own arrays (memory-mapped to a directory of its own): the
        # episodes of those buffers are copied into the storage, while their files are left where they are
        episodes = state.pop("_buf", None)
        # Up to sheeprl 0.8.0 the episodes stored weren't counted
        state.setdefault("_stored", len(state.get("_starts", [])))
        self.__dict__.update(state)
        if episodes is not None:
            self._storage, self._starts, self._pos, self._stored = {}, [], 0, 0
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
        # The online queue isn't checkpointed: it restarts empty, with the episodes stored after the loading (up to
        # sheeprl 0.8.2 it was kept in `_online_episode` and `_online_sequence`)
        self.__dict__.pop("_online_episode", None)
        self.__dict__.pop("_online_sequence", None)
        self._online = EpisodeQueue(self._stored)

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        n_samples: int = 1,
        clone: bool = False,
        sequence_length: int = 1,
        online: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample trajectories from the replay buffer (`EpisodeSampler`, with the online queue of the buffer and the
        global generator of NumPy).

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
            online (bool): whether the samples start with the sequences of the online queue, the oldest first, and only
                the rest of them is sampled uniformly: every step of the stored episodes (but their first ones, fewer
                than a sequence) is sampled once soon after (see `EpisodeQueue`). Defaults to False.

        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of
            [n_samples, sequence_length, batch_size, ...].
        """
        sampler = EpisodeSampler(sequence_length, sample_next_obs, self._prioritize_ends, online, queue=self._online)
        return sampler.sample(self, batch_size, n_samples, clone=clone)

    def gather(
        self, first_steps: np.ndarray, sequence_length: int, sample_next_obs: bool = False, clone: bool = False
    ) -> Dict[str, np.ndarray]:
        """The sequences of `sequence_length` steps that start at the steps `first_steps` of the storage, where an
        episode can continue from its end to its start, of shape `[len(first_steps), sequence_length, ...]`. With
        `sample_next_obs`, the steps after them too, as `next_<key>` of the observation keys."""
        indices = (first_steps.reshape(-1, 1) + np.arange(sequence_length, dtype=np.intp)) % self._capacity
        shape = (len(first_steps), sequence_length)
        samples = {}
        for k, v in self._storage.items():
            array = v.array if isinstance(v, MemmapArray) else v
            samples[k] = np.take(array, indices.flat, axis=0).reshape(shape + array.shape[1:])
            if sample_next_obs and k in self._obs_keys:
                next_v = np.take(array, ((indices + 1) % self._capacity).flat, axis=0)
                samples[f"next_{k}"] = next_v.reshape(shape + array.shape[1:])
        if clone:
            samples = {k: v.copy() for k, v in samples.items()}
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
        online: bool = False,
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
            online (bool): whether the samples start with the sequences of the online queue (see `sample`).
                Defaults to False.
            dtype (Optional[torch.dtype], optional): the torch dtype to convert the arrays to. If None,
                then the dtypes of the numpy arrays is maintained. Defaults to None.
            device (str | torch.dtype, optional): the torch device to move the tensors to. Defaults to "cpu".
            from_numpy (bool, optional): whether to convert the numpy arrays to torch tensors
                with the 'torch.from_numpy' function. If False, then the numpy arrays are converted
                with the 'torch.as_tensor' function. Defaults to False.
            kwargs: additional keyword arguments to be passed to the 'self.sample' method.
        """
        samples = self.sample(batch_size, sample_next_obs, n_samples, clone, sequence_length, online=online)
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


class _EnvView:
    """An environment of a `ReplayBuffer`, as a buffer of one environment: its row, whether it was filled and its steps
    added, which its sampler reads (`EnvIndependentSampler`)."""

    n_envs = 1
    lockstep = True

    def __init__(self, buffer: ReplayBuffer, env: int) -> None:
        self.buffer_size = buffer.buffer_size
        self._pos = int(buffer._env_pos[env])
        self.full = self._full = bool(buffer._env_full[env])
        self._added = int(buffer._env_added[env])
        self.env_added = buffer._env_added[env : env + 1].copy()
