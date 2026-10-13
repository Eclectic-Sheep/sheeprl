import torch
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler

from sheeprl.data.samplers import EpochSampler


def test_the_minibatches_of_the_epochs_are_the_ones_of_a_random_sampler():
    # Shuffled at every epoch with the global generator of PyTorch, the last minibatch smaller
    torch.manual_seed(0)
    expected = [idxes for _ in range(3) for idxes in BatchSampler(RandomSampler(range(10)), 4, drop_last=False)]
    torch.manual_seed(0)
    minibatches = list(EpochSampler(4).epochs(10, 3))
    assert [idxes for idxes, _ in minibatches] == expected
    assert [size for _, size in minibatches] == [4, 4, 2] * 3


def test_every_process_takes_its_share_of_the_rollouts_of_all_of_them():
    # `buffer.share_data`: as `DistributedSampler`, with the epochs from `first_epoch`
    for rank in (0, 1):
        sampler = DistributedSampler(range(10), num_replicas=2, rank=rank, shuffle=True, seed=7)
        expected = []
        for epoch in (2, 3):
            sampler.set_epoch(epoch)
            expected += list(BatchSampler(sampler, 2, drop_last=False))
        minibatches = EpochSampler(2, num_replicas=2, rank=rank, seed=7, distributed=True).epochs(10, 2, first_epoch=2)
        assert [idxes for idxes, _ in minibatches] == expected


def test_the_minibatches_are_padded_with_copies_of_their_first_element():
    torch.manual_seed(0)
    minibatches = list(EpochSampler(4).epochs(10, 1, pad_to=8))
    assert all(len(idxes) == 8 and idxes[size:] == idxes[:1] * (8 - size) for idxes, size in minibatches)
    assert sorted(i for idxes, size in minibatches for i in idxes[:size]) == list(range(10))
