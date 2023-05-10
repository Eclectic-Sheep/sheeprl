import typing
from typing import Union

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import Size, Tensor, device


class ReplayBuffer:
    def __init__(self, buffer_size: int, n_envs: int = 1, device: Union[device, str] = "cpu"):
        """A replay buffer which internally uses a TensorDict.

        Args:
            buffer_size (int): The buffer size.
            n_envs (int, optional): The number of environments. Defaults to 1.
            device (Union[torch.device, str], optional): The device where the buffer is created. Defaults to "cpu".
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
        self._buf = TensorDict({}, batch_size=[buffer_size, n_envs], device=device)
        self._pos = 0
        self._full = False

    @property
    def buffer(self) -> TensorDictBase:
        return self._buf

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def full(self) -> int:
        return self._full

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def shape(self) -> Size:
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
            raise TypeError("`data` must be a TensorDictBase or a fabricrl.data.ReplayBuffer")
        if len(data.shape) != 2:
            raise RuntimeError(
                "`data` must have 2 batch dimensions: [sequence_length, n_envs]. "
                "`sequence_length` and `n_envs` should be 1. Shape is: {}".format(data.shape)
            )
        data_len = data.shape[0]
        next_pos = (self._pos + data_len) % self._buffer_size
        if next_pos < self._pos or (self._pos + data_len >= self._buffer_size and self._pos == 0):
            idxes = torch.tensor(
                list(range(self._pos, self._buffer_size)) + list(range(0, next_pos)), device=self.device
            )
        else:
            idxes = torch.tensor(range(self._pos, next_pos), device=self.device)
        self._buf[idxes, :] = data
        if self._pos + data_len >= self._buffer_size:
            self._full = True
        self._pos = next_pos

    def sample(self, batch_size: int, sample_next_obs: bool = False, clone: bool = False) -> TensorDictBase:
        """Sample elements from the replay buffer.

        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        Args:
            batch_size (int): batch_size (int): Number of element to sample
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
        if self._full and batch_size > self._buf.shape[0]:
            raise ValueError(f"Batch size {batch_size} is larger than the replay buffer size ({self._buf.shape[0]})")
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self._full:
            if sample_next_obs:
                batch_idxes = (
                    torch.randint(1, self._buffer_size, size=(batch_size,), device=self.device) + self._pos
                ) % self._buffer_size
            else:
                batch_idxes = torch.randint(0, self._buffer_size, size=(batch_size,), device=self.device)
        else:
            batch_idxes = torch.randint(0, self._pos, size=(batch_size,), device=self.device)
        sample = self._get_samples(batch_idxes, sample_next_obs=sample_next_obs).unsqueeze(-1)
        if clone:
            return sample.clone()
        return sample

    def _get_samples(self, batch_idxes: Tensor, sample_next_obs: bool = False) -> TensorDictBase:
        env_idxes = torch.randint(0, self.n_envs, size=(len(batch_idxes),))
        buf = self._buf[batch_idxes, env_idxes]
        if sample_next_obs:
            buf["next_observations"] = self._buf["observations"][(batch_idxes + 1) % self._buffer_size, env_idxes]
        return buf

    def __getitem__(self, key: str) -> torch.Tensor:
        if not isinstance(key, str):
            raise TypeError("`key` must be a string")
        return self._buf.get(key)

    def __setitem__(self, key: str, t: Tensor) -> None:
        self.buffer.set(key, t, inplace=True)
