import pickle

import numpy as np
import pytest
import torch

from sheeprl.data.buffers import EnvIndependentReplayBuffer, ReplayBuffer, SequentialReplayBuffer
from sheeprl.data.samplers import SequenceSampler, TransitionSampler
from sheeprl.data.store import ReplayStore

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="A buffer in the memory of a GPU")


def steps(t, n_envs, rng):
    return {
        "observations": rng.normal(size=(t, n_envs, 3)).astype(np.float32),
        "rgb": rng.integers(0, 255, size=(t, n_envs, 2, 2), dtype=np.uint8),
        "terminated": (rng.random((t, n_envs, 1)) < 0.2).astype(np.float32),
        "truncated": np.zeros((t, n_envs, 1), dtype=np.float32),
    }


def buffers(kind, device):
    kw = dict(obs_keys=("observations", "rgb"), device=device)
    if kind == "transitions":
        return ReplayBuffer(8, 2, **kw), lambda: TransitionSampler(True, True, seed=1)
    if kind == "sequences":
        return SequentialReplayBuffer(8, 2, **kw), lambda: SequenceSampler(3, True, True, seed=1)
    cls = ReplayBuffer if kind == "env_transitions" else SequentialReplayBuffer
    sampler = (
        (lambda: TransitionSampler(True, True, seed=1))
        if kind == "env_transitions"
        else (lambda: SequenceSampler(3, True, True, seed=1))
    )
    return EnvIndependentReplayBuffer(8, 2, buffer_cls=cls, **kw), sampler


@cuda
@pytest.mark.parametrize("kind", ["transitions", "sequences", "env_transitions", "env_sequences"])
def test_a_buffer_on_the_gpu_samples_what_a_buffer_on_the_cpu_samples(kind):
    # The same steps drawn by the same samplers: gathered on the GPU, with no copy from the CPU
    samples = {}
    for device in (None, "cuda"):
        rng = np.random.default_rng(0)
        storage, sampler = buffers(kind, device)
        store = ReplayStore(storage, sampler(), device="cuda")
        samples[device] = []
        for t in (5, 3, 9):
            store.add(steps(t, 2, rng))
            samples[device].append(store.sample(2, 3))
        # Through a checkpoint
        store = pickle.loads(pickle.dumps(store))
        store.add(steps(2, 2, rng))
        samples[device].append(store.sample(2, 3))
    for cpu, gpu in zip(samples[None], samples["cuda"]):
        assert cpu.keys() == gpu.keys()
        for k in cpu:
            assert gpu[k].device.type == "cuda" and gpu[k].dtype == cpu[k].dtype
            assert torch.equal(gpu[k], cpu[k]), k


@cuda
def test_a_buffer_on_the_gpu_must_fit_in_its_free_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (100, 10**10))
    rb = ReplayBuffer(8, 2, device="cuda")
    with pytest.raises(RuntimeError, match="buffer.on_device=False"):
        rb.add(steps(1, 2, np.random.default_rng(0)))


def test_a_buffer_on_a_device_is_not_memory_mapped(tmp_path):
    with pytest.raises(ValueError, match="memory-mapped"):
        ReplayBuffer(8, 2, memmap=True, memmap_dir=tmp_path, device="cpu")


def test_a_buffer_on_a_device_takes_columns_and_converts_to_tensors():
    # `__setitem__` (e.g. the `is_first` completed by DreamerV1) and `to_tensor` with tensors
    rb = ReplayBuffer(4, 2, device="cpu")
    rb.add(steps(4, 2, np.random.default_rng(0)))
    rb["is_first"] = np.ones((4, 2, 1), dtype=np.float32)
    assert torch.is_tensor(rb["is_first"]) and rb["is_first"].sum() == 8
    tensors = rb.to_tensor(dtype=torch.float64)
    assert tensors["rgb"].dtype == torch.float64 and tensors["rgb"].shape == (4, 2, 2, 2)


@cuda
@pytest.mark.parametrize("memmap", [False, True])
def test_a_buffer_moves_between_the_cpu_and_the_gpu(memmap, tmp_path):
    # A loaded buffer goes where the run keeps it (`ReplayStore.load`): a memory-mapped one is no longer memory-mapped
    rb = ReplayBuffer(8, 2, memmap=memmap, memmap_dir=tmp_path if memmap else None)
    # Full: the rows never written hold garbage, which can be NaN
    rb.add(steps(10, 2, np.random.default_rng(0)))
    expected = {k: np.array(v) for k, v in rb.buffer.items()}
    rb.to("cuda")
    assert rb.device.type == "cuda" and not rb.is_memmap
    assert all(torch.equal(v, torch.as_tensor(expected[k], device="cuda")) for k, v in rb.buffer.items())
    rb.to(None)
    assert rb.device is None and all(np.array_equal(v, expected[k]) for k, v in rb.buffer.items())
    assert all(isinstance(v, np.ndarray) for v in rb.buffer.values())
