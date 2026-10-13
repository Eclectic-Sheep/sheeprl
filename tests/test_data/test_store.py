import pickle

import numpy as np
import pytest
import torch

from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import TransitionSampler
from sheeprl.data.store import ReplayStore


def store(prefetch, device="cpu"):
    return ReplayStore(ReplayBuffer(32, n_envs=2), TransitionSampler(seed=0), device=device, prefetch=prefetch)


def add(store, steps, first):
    store.add({"observations": np.arange(first, first + 2 * steps, dtype=np.float32).reshape(steps, 2, 1)})


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"))],
)
def test_the_prefetched_batches_are_sampled_at_the_end_of_the_previous_training(device):
    # The batches of the next iteration are sampled when the ones of an iteration have been used, before the steps of
    # the next iteration are written: as a store that samples them at that point
    eager, prefetching = store(False, device), store(True, device)
    for s in (eager, prefetching):
        add(s, 10, 0)
    first = [list(s.batches(3, 4)) for s in (eager, prefetching)]
    for a, b in zip(*first):
        assert torch.equal(a["observations"], b["observations"])
    expected = eager.sample(4, 3)["observations"]
    # Written after the prefetching: not in its batches
    for s in (eager, prefetching):
        add(s, 5, 100)
    batches = list(prefetching.batches(3, 4))
    assert len(batches) == 3
    for i, batch in enumerate(batches):
        assert batch["observations"].device.type == device
        assert torch.equal(batch["observations"], expected[i].float())
    assert all((batch["observations"] < 100).all() for batch in batches)
    # The next iteration has its batches prefetched too
    assert prefetching._prefetched is not None


def test_prefetched_batches_of_another_size_are_dropped():
    prefetching = store(True)
    add(prefetching, 10, 0)
    list(prefetching.batches(3, 4))
    # Two gradient steps instead of three: they are sampled again
    batches = list(prefetching.batches(2, 4))
    assert len(batches) == 2 and all(batch["observations"].shape == (4, 1) for batch in batches)


def test_a_checkpoint_waits_for_the_prefetched_batches():
    # The generator of the sampler is saved after the prefetched batches are drawn, without them
    eager, prefetching = store(False), store(True)
    for s in (eager, prefetching):
        add(s, 10, 0)
        list(s.batches(3, 4))
    eager.sample(4, 3)
    loaded = pickle.loads(pickle.dumps(prefetching))
    assert loaded._prefetched is None and loaded.prefetch
    assert loaded.sampler.rng.bit_generator.state == eager.sampler.rng.bit_generator.state
    # The store keeps its prefetched batches
    assert len(list(prefetching.batches(3, 4))) == 3


def test_an_error_of_the_prefetching_is_raised_by_the_training():
    class FailingSampler(TransitionSampler):
        calls = 0

        def sample(self, *args, **kwargs):
            FailingSampler.calls += 1
            if FailingSampler.calls > 1:
                raise RuntimeError("sampling failed")
            return super().sample(*args, **kwargs)

    prefetching = ReplayStore(ReplayBuffer(32, n_envs=2), FailingSampler(seed=0), prefetch=True)
    add(prefetching, 10, 0)
    list(prefetching.batches(1, 4))
    with pytest.raises(RuntimeError, match="sampling failed"):
        list(prefetching.batches(1, 4))
