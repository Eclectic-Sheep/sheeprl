from typing import Iterator

from tensordict.tensordict import TensorDictBase
from torch.utils.data import Sampler


class SequenceSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (TensorDictBase): TensorDict to sample sequences from.
            Sequences are sampled given the `episodes_end`.
    """
    _data_source: TensorDictBase

    def __init__(self, data_source: TensorDictBase) -> None:
        if "episodes_end" not in data_source.keys():
            raise KeyError(
                "Key `episodes_end` is not found! Please set `episodes_end` directly on the `data_source` "
                "or if the `data_source` comes from a fabricrl.data.buffers.ReplayBuffer, then call "
                "`rb.set_episodes_end()`."
            )
        self._data_source = data_source
        episodes_ends = self._data_source["episodes_end"].view(-1).nonzero().flatten().tolist()
        if episodes_ends[0] != 0:
            episodes_ends.insert(0, 0)
        self._sequences_ranges = [range(episodes_ends[i], episodes_ends[i + 1]) for i in range(len(episodes_ends) - 1)]

    def __iter__(self) -> Iterator[int]:
        for sequence in self._sequences_ranges:
            yield list(sequence)

    def __len__(self) -> int:
        return len(self._sequences_ranges)
