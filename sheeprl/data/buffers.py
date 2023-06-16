import random
import typing
from typing import Optional, Union

import torch
from tensordict import LazyStackedTensorDict, MemmapTensor, TensorDict
from tensordict.tensordict import TensorDictBase
from torch import Size, Tensor, device


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        device: Union[device, str] = "cpu",
        memmap: bool = False,
    ):
        """A replay buffer which internally uses a TensorDict.

        Args:
            buffer_size (int): The buffer size.
            n_envs (int, optional): The number of environments. Defaults to 1.
            device (Union[torch.device, str], optional): The device where the buffer is created. Defaults to "cpu".
            memmap (bool, optional): Whether to memory-mapping the buffer
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
        if self._memmap:
            self._buffer = None
        else:
            self._buffer = TensorDict({}, batch_size=[buffer_size, n_envs], device=device)
        self._pos = 0
        self._full = False

    @property
    def buffer(self) -> Optional[TensorDictBase]:
        return self._buffer

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
            raise TypeError("`data` must be a TensorDictBase or a sheeprl.data.ReplayBuffer")
        if len(data.shape) != 2:
            raise RuntimeError(
                "`data` must have 2 batch dimensions: [sequence_length, n_envs]. "
                "`sequence_length` and `n_envs` should be 1. Shape is: {}".format(data.shape)
            )
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
        if self._memmap and self._buffer is None:
            self._buffer = TensorDict(
                {
                    k: MemmapTensor((self._buffer_size, self._n_envs, *v.shape[2:]), dtype=v.dtype, device=v.device)
                    for k, v in data_to_store.items()
                },
                batch_size=[self._buffer_size, self._n_envs],
                device=self.device,
            )
            self._buffer.memmap_()
        self._buffer[idxes, :] = data_to_store
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
        if self._buffer is None:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        buf = self._buffer[batch_idxes, env_idxes]
        if sample_next_obs:
            buf["next_observations"] = self._buffer["observations"][(batch_idxes + 1) % self._buffer_size, env_idxes]
        return buf

    def __getitem__(self, key: str) -> torch.Tensor:
        if not isinstance(key, str):
            raise TypeError("`key` must be a string")
        return self._buffer.get(key)

    def __setitem__(self, key: str, t: Tensor) -> None:
        self.buffer.set(key, t, inplace=True)


class Trajectory(TensorDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.shape[0]

    def sample(self, position: int, num_samples: int) -> Optional[TensorDictBase]:
        if len(self) < position + num_samples:
            return
        return self[position : position + num_samples]


class TrajectoryReplayBuffer:
    def __init__(self, max_num_trajectories: int):
        self._buffer = []
        self.max_num_trajectories = max_num_trajectories

    @property
    def buffer(self):
        return self._buffer

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index: int):
        return self._buffer[index]

    def add(self, trajectory: Optional[Trajectory]):
        if trajectory is None:
            return
        if not isinstance(trajectory, TensorDict):
            raise TypeError("Trajectory must be an instance of Trajectory")
        self._buffer.append(Trajectory(trajectory))  # convert to trajectory if tensordict
        if len(self) > self.max_num_trajectories:
            self._buffer.pop(0)

    def sample(self, batch_size: int, sequence_length: int) -> LazyStackedTensorDict:
        """Sample a batch of trajectories of length `sequence_length`.

        Args:
            batch_size (int): Number of trajectories to sample
            sequence_length (int): Length of the trajectories to sample

        Returns:
            LazyStackedTensorDict: the sampled trajectories with shape [batch_size, sequence_length, ...]
        """
        valid_trajectories = [t for t in self.buffer if len(t) >= sequence_length]
        if len(valid_trajectories) == 0:
            raise RuntimeError("No trajectories of length {} found".format(sequence_length))

        trajectories = random.choices(valid_trajectories, k=batch_size)
        positions = [random.randint(0, len(t) - sequence_length) for t in trajectories]
        return torch.stack([t.sample(p, sequence_length) for t, p in zip(trajectories, positions)], dim=1)
