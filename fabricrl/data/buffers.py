from typing import Union

import torch
from tensordict import TensorDict
from torch import Size, Tensor


class ReplayBuffer:
    """Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(self, buffer_size: int, n_envs: int = 1, device: Union[torch.device, str] = "cpu"):
        """_summary_

        Args:
            buffer_size (int): _description_
            n_envs (int, optional): _description_. Defaults to 1.
            device (Union[torch.device, str], optional): _description_. Defaults to "cpu".
        """
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        if isinstance(device, str):
            device = torch.device(device=device)
        self._device = device
        self._buf = TensorDict({}, batch_size=[buffer_size, n_envs], device=device)
        self._pos = 0
        self._full = False

    @property
    def buffer(self) -> TensorDict:
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
    def device(self) -> torch.device:
        return self._device

    def __len__(self) -> int:
        return self.buffer_size

    def add(self, data: TensorDict) -> None:
        """Add another Tensordict to the buffer.

        Args:
            data (TensorDict): data to add.

        Raises:
            RuntimeError: the number of dimensions (the batch_size of the TensorDict) must be 2:
            one for the number of environments and one for the sequence length.
        """
        if len(data.shape) != 2:
            raise RuntimeError(
                "`data` must have 2 batch dimensions: [sequence_length, n_envs, d1, ..., dn]. "
                "`sequence_length` and `n_envs` should be 1. Shape is: {}".format(data.shape)
            )
        data_len = data.shape[0]
        next_pos = (self._pos + data_len) % self._buffer_size
        if self._pos == 0 and next_pos == 0:
            next_pos = data_len
        if next_pos < self._pos:
            idxes = torch.tensor(
                list(range(self._pos, self._buffer_size)) + list(range(0, next_pos)), device=self.device
            )
        else:
            idxes = torch.tensor(range(self._pos, next_pos), device=self.device)
        self._buf[idxes, :] = data
        if self._pos + data_len >= self._buffer_size:
            self._full = True
        self._pos = next_pos

    def sample(self, batch_size: int) -> TensorDict:
        """Sample elements from the replay buffer.

        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        Args:
            batch_size (int): batch_size (int): Number of element to sample

        Returns:
            TensorDict: the sampled TensorDict, cloned
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        if batch_size > self._buf.shape[0]:
            raise ValueError(f"Batch size {batch_size} is larger than the replay buffer size {self._buf.shape[0]}")
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self._full:
            batch_idxes = (
                torch.randint(1, self._buffer_size, size=(batch_size, self.n_envs), device=self.device) + self._pos
            ) % self._buffer_size
        else:
            batch_idxes = torch.randint(0, self._pos - 1, size=(batch_size, self.n_envs), device=self.device)
        return self._get_samples(batch_idxes)

    def _get_samples(self, batch_idxes: Tensor) -> TensorDict:
        buf = torch.gather(self._buf, dim=0, index=batch_idxes).clone()
        buf["next_obs"] = self._buf["observations"][
            (batch_idxes + 1) % self._buffer_size, torch.arange(self.n_envs, device=self.device)
        ].clone()
        return buf

    def __getitem__(self, key: str) -> torch.Tensor:
        if not isinstance(key, str):
            raise TypeError("`key` must be a string")
        return self._buf.get(key)

    def __setitem__(self, key: str, t: Tensor) -> None:
        self.buffer.set(key, t)
