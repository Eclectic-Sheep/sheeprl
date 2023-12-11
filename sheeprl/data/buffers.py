import os
import shutil
import typing
import uuid
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from tensordict import MemmapTensor, TensorDict
from tensordict.tensordict import TensorDictBase
from torch import Size, Tensor, device


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        device: Union[device, str] = "cpu",
        memmap: bool = False,
        memmap_dir: Optional[Union[str, os.PathLike]] = None,
        obs_keys: Sequence[str] = ("observations",),
    ):
        """A replay buffer which internally uses a TensorDict.

        Args:
            buffer_size (int): The buffer size.
            n_envs (int, optional): The number of environments. Defaults to 1.
            device (Union[torch.device, str], optional): The device where the buffer is created. Defaults to "cpu".
            memmap (bool, optional): Whether to memory-mapping the buffer.
        """
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if n_envs <= 0:
            raise ValueError(f"The number of environments must be greater than zero, got: {n_envs}")
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        if isinstance(device, str):
            device = torch.device(device=device)
        self._device = device
        self._memmap = memmap
        self._memmap_dir = memmap_dir
        if self._memmap:
            if memmap_dir is None:
                warnings.warn(
                    "The buffer will be memory-mapped into the `/tmp` folder, this means that there is the"
                    " possibility to lose the saved files. Set the `memmap_dir` to a known directory.",
                    UserWarning,
                )
            else:
                self._memmap_dir = Path(self._memmap_dir)
                self._memmap_dir.mkdir(parents=True, exist_ok=True)
            self._buf = None
        else:
            self._buf = TensorDict({}, batch_size=[buffer_size, n_envs], device=device)
        self._pos = 0
        self._full = False
        self.obs_keys = obs_keys

    @property
    def buffer(self) -> Optional[TensorDictBase]:
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
    def shape(self) -> Optional[Size]:
        if self.buffer is None:
            return None
        return self.buffer.shape

    @property
    def device(self) -> device:
        return self._device

    def __len__(self) -> int:
        return self.buffer_size

    @typing.overload
    def add(self, data: "ReplayBuffer") -> None:
        ...

    @typing.overload
    def add(self, data: TensorDictBase) -> None:
        ...

    def add(self, data: Union["ReplayBuffer", TensorDictBase]) -> None:
        """Add data to the buffer.

        Args:
            data: data to add.

        Raises:
            RuntimeError: the number of dimensions (the batch_size of the TensorDictBase) must be 2:
            one for the number of environments and one for the sequence length.
        """
        if isinstance(data, ReplayBuffer):
            data = data.buffer
        elif not isinstance(data, TensorDictBase):
            raise TypeError("`data` must be a TensorDictBase or a sheeprl.data.ReplayBuffer")
        if data is None:
            raise RuntimeError("The `data` replay buffer must be not None")
        if len(data.shape) != 2:
            raise RuntimeError(
                "`data` must have 2 batch dimensions: [sequence_length, n_envs]. "
                "`sequence_length` and `n_envs` should be 1. Shape is: {}".format(data.shape)
            )
        data = data.to(self.device)
        data_len = data.shape[0]
        next_pos = (self._pos + data_len) % self._buffer_size
        if next_pos < self._pos or (data_len >= self._buffer_size and not self._full):
            idxes = torch.tensor(
                list(range(self._pos, self._buffer_size)) + list(range(0, next_pos)), device=self.device
            )
        else:
            idxes = torch.tensor(range(self._pos, next_pos), device=self.device)
        if data_len > self._buffer_size:
            data_to_store = data[-self._buffer_size - next_pos :]
        else:
            data_to_store = data
        if self._memmap and self._buf is None:
            self._buf = TensorDict(
                {
                    k: MemmapTensor(
                        (self._buffer_size, self._n_envs, *v.shape[2:]),
                        dtype=v.dtype,
                        device=v.device,
                        filename=None if self._memmap_dir is None else self._memmap_dir / f"{k}.memmap",
                    )
                    for k, v in data_to_store.items()
                },
                batch_size=[self._buffer_size, self._n_envs],
                device=self.device,
            )
            self._buf.memmap_(prefix=self._memmap_dir)
        self._buf[idxes, :] = data_to_store
        if self._pos + data_len >= self._buffer_size:
            self._full = True
        self._pos = next_pos

    def sample(self, batch_size: int, sample_next_obs: bool = False, clone: bool = False, **kwargs) -> TensorDictBase:
        """Sample elements from the replay buffer.

        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        Args:
            batch_size (int): Number of element to sample
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.
            clone (bool): whether to clone the sampled TensorDict

        Returns:
            TensorDictBase: the sampled TensorDictBase with a `batch_size` of [batch_size, 1]
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
            valid_idxes = torch.tensor(
                list(range(0, first_range_end)) + list(range(self._pos, second_range_end)),
                device=self.device,
            )
            batch_idxes = valid_idxes[torch.randint(0, len(valid_idxes), size=(batch_size,), device=self.device)]
        else:
            max_pos_to_sample = self._pos - 1 if sample_next_obs else self._pos
            if max_pos_to_sample == 0:
                raise RuntimeError(
                    "You want to sample the next observations, but one sample has been added to the buffer. "
                    "Make sure that at least two samples are added."
                )
            batch_idxes = torch.randint(0, max_pos_to_sample, size=(batch_size,), device=self.device)
        sample = self._get_samples(batch_idxes, sample_next_obs=sample_next_obs).unsqueeze(-1)
        if clone:
            return sample.clone()
        return sample

    def _get_samples(self, batch_idxes: Tensor, sample_next_obs: bool = False) -> TensorDictBase:
        env_idxes = torch.randint(0, self.n_envs, size=(len(batch_idxes),))
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        buf = self._buf[batch_idxes, env_idxes]
        if sample_next_obs:
            for k in self.obs_keys:
                buf[f"next_{k}"] = self._buf[k][(batch_idxes + 1) % self._buffer_size, env_idxes]
        return buf

    def __getitem__(self, key: str) -> torch.Tensor:
        if not isinstance(key, str):
            raise TypeError("`key` must be a string")
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        return self._buf.get(key)

    def __setitem__(self, key: str, t: Tensor) -> None:
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        self._buf.set(key, t, inplace=True)


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
        device: Union[device, str] = "cpu",
        memmap: bool = False,
        memmap_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        super().__init__(buffer_size, n_envs, device, memmap, memmap_dir)

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        sequence_length: int = 1,
        n_samples: int = 1,
    ) -> TensorDictBase:
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
            TensorDictBase: the sampled TensorDictBase with a `batch_size` of [n_samples, sequence_length, batch_size]
        """
        # the batch_size can be fused with the number of samples to have single batch size
        batch_dim = batch_size * n_samples

        # Controls
        if batch_dim <= 0:
            raise ValueError("Batch size must be greater than 0")
        if not self._full and self._pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. Please add at least one sample calling `self.add()`"
            )
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        if not self._full and self._pos - sequence_length + 1 < 1:
            raise ValueError(f"too long sequence length ({sequence_length})")
        if self.full and sequence_length > self._buf.shape[0]:
            raise ValueError(f"too long sequence length ({sequence_length})")

        # Do not sample the element with index `self.pos` as the transitions is invalid
        if self._full:
            # when the buffer is full, it is necessary to avoid the starting index between (self.pos - sequence_length)
            # and self.pos, so it is possible to sample the starting index between (0, self.pos - sequence_length) and
            # between (self.pos, self.buffer_size)
            first_range_end = self._pos - sequence_length + 1
            # end of the second range, if the first range is empty, then the second range ends
            # in (buffer_size + (self._pos - sequence_length + 1)), otherwise the sequence will contain
            # invalid values
            second_range_end = self.buffer_size if first_range_end >= 0 else self.buffer_size + first_range_end
            valid_idxes = torch.tensor(
                list(range(0, first_range_end)) + list(range(self._pos, second_range_end)),
                device=self.device,
            )
            # start_idxes are the indices of the first elements of the sequences
            start_idxes = valid_idxes[torch.randint(0, len(valid_idxes), size=(batch_dim,), device=self.device)]
        else:
            # when the buffer is not full, we need to start the sequence so that it does not go out of bounds
            start_idxes = torch.randint(0, self._pos - sequence_length + 1, size=(batch_dim,), device=self.device)

        # chunk_length contains the relative indices of the sequence (0, 1, ..., sequence_length-1)
        chunk_length = torch.arange(sequence_length, device=self.device).reshape(1, -1)
        idxes = (start_idxes.reshape(-1, 1) + chunk_length) % self.buffer_size

        # (n_samples, sequence_length, batch_size)
        sample = self._get_samples(idxes).reshape(n_samples, batch_size, sequence_length).permute(0, -1, -2)
        if clone:
            return sample.clone()
        return sample

    def _get_samples(self, batch_idxes: Tensor, sample_next_obs: bool = False) -> TensorDictBase:
        """Retrieves the items and return the TensorDict of sampled items.

        Args:
            batch_idxes (Tensor): the indices to retrieve of dimension (batch_dim, sequence_length).
            sample_next_obs (bool): whether to sample the next observations from the 'observations' key.
                Defaults to False.

        Returns:
            TensorDictBase: the sampled TensorDictBase with a `batch_size` of [batch_dim, sequence_length]
        """
        unflatten_shape = batch_idxes.shape
        # each sequence must come from the same environment
        env_idxes = (
            torch.randint(0, self.n_envs, size=(unflatten_shape[0],)).view(-1, 1).repeat(1, unflatten_shape[1]).view(-1)
        )
        # retrieve the items by flattening the indices
        # (b1_s1, b1_s2, b1_s3, ..., bn_s1, bn_s2, bn_s3, ...)
        # where bm_sk is the k-th elements in the sequence of the m-th batch
        sample = self._buf[batch_idxes.flatten(), env_idxes]
        # properly reshape the items:
        # [
        #   [b1_s1, b1_s2, ...],
        #   [b2_s1, b2_s2, ...],
        #   ...,
        #   [bn_s1, bn_s2, ...]
        # ]
        return sample.view(*unflatten_shape)


class EpisodeBuffer:
    """A replay buffer that stores separately the episodes.

    Args:
        buffer_size (int): The capacity of the buffer.
        sequence_length (int): The length of the sequences of the samples
            (an episode cannot be shorter than the episode length).
        device (Union[torch.device, str]): The device where the buffer is created. Defaults to "cpu".
        memmap (bool): Whether to memory-mapping the buffer.
    """

    def __init__(
        self,
        buffer_size: int,
        sequence_length: int,
        device: Union[device, str] = "cpu",
        memmap: bool = False,
        memmap_dir: Optional[Union[str, os.PathLike]] = None,
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
        self._buffer_size = buffer_size
        self._sequence_length = sequence_length
        self._buf: List[TensorDictBase] = []
        self._cum_lengths: List[int] = []
        if isinstance(device, str):
            device = torch.device(device=device)
        self._device = device
        self._memmap = memmap
        self._memmap_dir = memmap_dir
        if memmap_dir is None:
            warnings.warn(
                "The buffer will be memory-mapped into the `/tmp` folder, this means that there is the"
                " possibility to lose the saved files. Set the `memmap_dir` to a known directory.",
                UserWarning,
            )
        else:
            self._memmap_dir = Path(self._memmap_dir)
            self._memmap_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_length = torch.arange(sequence_length, device=self.device).reshape(1, -1)

    @property
    def buffer(self) -> Optional[List[TensorDictBase]]:
        return self._buf

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def device(self) -> device:
        return self._device

    @property
    def is_memmap(self) -> bool:
        return self._memmap

    @property
    def full(self) -> bool:
        return self._cum_lengths[-1] + self._sequence_length > self._buffer_size if len(self._buf) > 0 else False

    def __getitem__(self, key: int) -> torch.Tensor:
        if not isinstance(key, int):
            raise TypeError("`key` must be an integer")
        return self._buf[key]

    def __len__(self) -> int:
        return self._cum_lengths[-1] if len(self._buf) > 0 else 0

    def add(self, episode: TensorDictBase) -> None:
        """Add an episode to the buffer.

        Args:
            episode (TensorDictBase): data to add.

        Raises:
            RuntimeError:
                - The episode must contain exactly one done at the end of the episode.
                - The length of the episode must be at least sequence lenght.
                - The length of the episode cannot be greater than the buffer size.
        """
        if len(torch.nonzero(episode["dones"])) != 1:
            raise RuntimeError(
                f"The episode must contain exactly one done, got: {len(torch.nonzero(episode['dones']))}"
            )
        if episode["dones"][-1] != 1:
            raise RuntimeError(f"The last step must contain a done, got: {episode['dones'][-1]}")
        if episode.shape[0] < self._sequence_length:
            raise RuntimeError(
                f"Episode too short (at least {self._sequence_length} steps), got: {episode.shape[0]} steps"
            )
        if episode.shape[0] > self._buffer_size:
            raise RuntimeError(f"Episode too long (at most {self._buffer_size} steps), got: {episode.shape[0]} steps")

        ep_len = episode.shape[0]
        if self.full or len(self) + ep_len > self._buffer_size:
            cum_lengths = np.array(self._cum_lengths)
            mask = (len(self) - cum_lengths + ep_len) <= self._buffer_size
            last_to_remove = mask.argmax()
            # Remove all memmaped episodes
            if self._memmap and self._memmap_dir is not None:
                for _ in range(last_to_remove + 1):
                    filename = self._buf[0][self._buf[0].sorted_keys[0]].filename
                    for k in self._buf[0].sorted_keys:
                        f = self._buf[0][k].file
                        if f is not None:
                            f.close()
                    del self._buf[0]
                    shutil.rmtree(os.path.dirname(filename))
            else:
                self._buf = self._buf[last_to_remove + 1 :]
            cum_lengths = cum_lengths[last_to_remove + 1 :] - cum_lengths[last_to_remove]
            self._cum_lengths = cum_lengths.tolist()
        self._cum_lengths.append(len(self) + ep_len)
        if self._memmap:
            episode_dir = None
            if self._memmap_dir is not None:
                episode_dir = self._memmap_dir / f"episode_{str(uuid.uuid4())}"
                episode_dir.mkdir(parents=True, exist_ok=True)
            for k, v in episode.items():
                episode[k] = MemmapTensor.from_tensor(
                    v,
                    filename=None if episode_dir is None else episode_dir / f"{k}.memmap",
                    transfer_ownership=False,
                )
            episode.memmap_(prefix=episode_dir)
        episode.to(self.device)
        self._buf.append(episode)

    def sample(
        self,
        batch_size: int,
        n_samples: int = 1,
        prioritize_ends: bool = False,
        clone: bool = False,
    ) -> TensorDictBase:
        """Sample trajectories from the replay buffer.

        Args:
            batch_size (int): Number of element in the batch.
            n_samples (bool): The number of samples to be retrieved.
                Defaults to 1.
            prioritize_ends (bool): Whether to clone prioritize the ends of the episodes.
                Defaults to False.

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

        nsample_per_eps = torch.bincount(torch.randint(0, len(self._buf), (batch_size * n_samples,)))
        samples = []
        for i, n in enumerate(nsample_per_eps):
            ep_len = self._buf[i].shape[0]
            upper = ep_len - self._sequence_length + 1
            if prioritize_ends:
                upper += self._sequence_length
            start_idxes = torch.min(
                torch.randint(0, upper, size=(n,)).reshape(-1, 1), torch.tensor(ep_len - self._sequence_length)
            )
            indices = start_idxes + self._chunk_length
            samples.append(self._buf[i][indices])
        samples = torch.cat(samples, 0).reshape(n_samples, batch_size, self._sequence_length).permute(0, -1, -2)
        if clone:
            return samples.clone()
        return samples


class AsyncReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        device: Union[device, str] = "cpu",
        memmap: bool = False,
        memmap_dir: Optional[Union[str, os.PathLike]] = None,
        sequential: bool = False,
    ):
        """An async replay buffer which internally uses a TensorDict. This replay buffer
        saves a experiences independently for every environment. When new data has to be added, it expects
        the TensorDict or the ReplayBuffer to be added to have a 2D shape dimension as [T, B], where T`
        represents the sequence length, while `B` is the number of environments to be batched.

        Args:
            buffer_size (int): The buffer size.
            n_envs (int, optional): The number of environments. Defaults to 1.
            device (Union[torch.device, str], optional): The device where the buffer is created. Defaults to "cpu".
            memmap (bool, optional): Whether to memory-mapping the buffer.
        """
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if n_envs <= 0:
            raise ValueError(f"The number of environments must be greater than zero, got: {n_envs}")
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        if isinstance(device, str):
            device = torch.device(device=device)
        self._device = device
        self._memmap = memmap
        self._memmap_dir = memmap_dir
        self._sequential = sequential
        self._buf: Optional[Sequence[ReplayBuffer]] = None
        if self._memmap_dir is not None:
            self._memmap_dir = Path(self._memmap_dir)
        if self._memmap:
            if memmap_dir is None:
                warnings.warn(
                    "The buffer will be memory-mapped into the `/tmp` folder, this means that there is the"
                    " possibility to lose the saved files. Set the `memmap_dir` to a known directory.",
                    UserWarning,
                )
            else:
                self._memmap_dir.mkdir(parents=True, exist_ok=True)

    @property
    def buffer(self) -> Optional[Sequence[ReplayBuffer]]:
        return tuple(self._buf)

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def full(self) -> Optional[Sequence[bool]]:
        if self.buffer is None:
            return None
        return tuple([b.full for b in self.buffer])

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def shape(self) -> Optional[Sequence[Size]]:
        if self.buffer is None:
            return None
        return tuple([b.shape for b in self.buffer])

    @property
    def device(self) -> Optional[Sequence[device]]:
        if self.buffer is None:
            return None
        return self._device

    def __len__(self) -> int:
        return self.buffer_size

    def add(self, data: TensorDictBase, indices: Optional[Sequence[int]] = None) -> None:
        """Add data to the buffer.

        Args:
            data: data to add.
            indices (Sequence[int], optional): the indices where to add the data.
                If None, then data will be added on every indices.
                Defaults to None.

        Raises:
            RuntimeError: the number of dimensions (the batch_size of the TensorDictBase) must be 2:
            one for the number of environments and one for the sequence length.
        """
        if not isinstance(data, TensorDictBase):
            raise TypeError("`data` must be a TensorDictBase")
        if data is None:
            raise RuntimeError("The `data` parameter must be not None")
        if len(data.shape) != 2:
            raise RuntimeError(
                "`data` must have 2 batch dimensions: [sequence_length, n_envs]. "
                "`sequence_length` and `n_envs` should be 1. Shape is: {}".format(data.shape)
            )
        if self._buf is None:
            buf_cls = SequentialReplayBuffer if self._sequential else ReplayBuffer
            self._buf = [
                buf_cls(
                    self.buffer_size,
                    n_envs=1,
                    device=self._device,
                    memmap=self._memmap,
                    memmap_dir=self._memmap_dir / f"env_{i}" if self._memmap_dir is not None else None,
                )
                for i in range(self._n_envs)
            ]
        if indices is None:
            indices = tuple(range(self.n_envs))
        for env_data_idx, env_idx in enumerate(indices):
            self._buf[env_idx].add(data[:, env_data_idx : env_data_idx + 1])

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        sequence_length: int = 1,
        n_samples: int = 1,
    ) -> TensorDictBase:
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
            TensorDictBase: the sampled TensorDictBase with a `batch_size` of [n_samples, sequence_length, batch_size]
        """
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(f"`batch_size` ({batch_size}) and `n_samples` ({n_samples}) must be both greater than 0")
        if self._buf is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")

        bs_per_buf = torch.bincount(torch.randint(0, self._n_envs, (batch_size,)))
        samples = [
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
        return torch.cat(samples, dim=2 if self._sequential else 0)
