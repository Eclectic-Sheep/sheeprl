import pathlib
import pickle
import re

import numpy as np
import pytest

import sheeprl
from sheeprl.data.buffers import EnvIndependentReplayBuffer, ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback


def test_every_called_hook_is_a_method_of_the_checkpoint_callback():
    # `fabric.call(hook, ...)` does nothing when no callback has the method `hook`: a misspelled or renamed hook would
    # silently stop saving the checkpoints
    hooks = set()
    for path in pathlib.Path(sheeprl.__file__).parent.rglob("*.py"):
        hooks |= set(re.findall(r'fabric\.call\(\s*"(\w+)"', path.read_text(encoding="utf-8")))
    assert len(hooks) > 0
    missing = {hook for hook in hooks if not callable(getattr(CheckpointCallback, hook, None))}
    assert len(missing) == 0, missing


@pytest.mark.parametrize("env_independent", [False, True])
def test_a_memory_mapped_checkpoint_keeps_the_truncation_of_its_last_step(tmp_path, env_independent):
    # A memory-mapped buffer is saved by reference to its files, where the truncation of the last step made for the
    # checkpoint is undone right after it: the loaded buffer writes it again when it is used, not when it is loaded
    cls = EnvIndependentReplayBuffer if env_independent else ReplayBuffer
    rb = cls(4, 2, memmap=True, memmap_dir=tmp_path)
    rb.add({"obs": np.arange(6).reshape(3, 2, 1), "truncated": np.zeros((3, 2, 1))})
    buffers = rb.buffer if env_independent else [rb]
    callback = CheckpointCallback()
    state = callback._ckpt_rb(rb)
    checkpoint = pickle.dumps(rb)
    callback._experiment_consistent_rb(rb, state)
    assert all(b["truncated"][2, :, 0].tolist() == [0] * b.n_envs for b in buffers)
    loaded = pickle.loads(checkpoint)
    # Loading writes nothing (the run that saved the checkpoint can still be running)
    assert all(np.asarray(b.buffer["truncated"])[2, :, 0].tolist() == [0] * b.n_envs for b in buffers)
    loaded_buffers = loaded.buffer if env_independent else [loaded]
    assert all(b["truncated"][2, :, 0].tolist() == [1] * b.n_envs for b in loaded_buffers)
