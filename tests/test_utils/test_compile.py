import pytest
import torch
from lightning import Fabric

import sheeprl.utils.compile as compile_utils
from sheeprl.utils.utils import dotdict


def compile_cfg(enabled=True, player=True, mode=None):
    return dotdict(
        {"algo": {"compile": {"enabled": enabled, "mode": mode, "player": player}}, "fabric": {"precision": "32-true"}}
    )


def counting_compile(monkeypatch, output=None):
    """`torch.compile` replaced by a function that counts the calls of what it compiles (and returns `output`, if
    given, as a CUDA graph returns the same memory at every replay)."""
    calls = []

    def compile(fn, mode=None):
        def compiled(*args, **kwargs):
            calls.append(mode)
            return fn(*args, **kwargs) if output is None else output

        return compiled

    monkeypatch.setattr(compile_utils.torch, "compile", compile)
    return calls


@pytest.mark.parametrize("enabled,player", [(False, True), (True, False)])
def test_a_player_is_compiled_only_when_enabled(monkeypatch, enabled, player):
    counting_compile(monkeypatch)
    fn = lambda x: x + 1  # noqa: E731
    assert compile_utils.compiled_player(fn, Fabric(accelerator="cpu"), compile_cfg(enabled, player)) is fn


def test_a_player_is_compiled_for_the_inputs_of_its_first_call(monkeypatch):
    # The final observations of some of the environments, or the greedy actions of the test, run uncompiled instead of
    # compiling the player again
    calls = counting_compile(monkeypatch)
    fn = compile_utils.compiled_player(lambda obs, greedy=False: obs["x"] * 2, Fabric(accelerator="cpu"), compile_cfg())
    for _ in range(3):
        assert torch.equal(fn({"x": torch.ones(4, 3)}), torch.full((4, 3), 2.0))
    assert torch.equal(fn({"x": torch.ones(2, 3)}), torch.full((2, 3), 2.0))
    fn({"x": torch.ones(4, 3)}, greedy=True)
    fn({"x": torch.ones(4, 3, dtype=torch.float64)})
    assert len(calls) == 3


@pytest.mark.parametrize("cuda_graphs", [True, False])
def test_the_outputs_of_a_player_replayed_by_cuda_graphs_are_copies(monkeypatch, cuda_graphs):
    # The next replay of a CUDA graph overwrites its outputs: e.g. the states of a recurrent player, given back at the
    # next step. Without CUDA graphs the player is compiled with the default mode
    static = torch.zeros(3)
    calls = counting_compile(monkeypatch, output=(static, {"states": static}))
    fabric = Fabric(accelerator="cpu")
    fn = compile_utils.compiled_player(lambda x: x, fabric, compile_cfg(mode="reduce-overhead"), cuda_graphs)
    actions, extra = fn(torch.zeros(3))
    static += 1
    assert calls == ["reduce-overhead" if cuda_graphs else None]
    assert torch.equal(actions, torch.zeros(3) if cuda_graphs else static)
    assert torch.equal(extra["states"], torch.zeros(3) if cuda_graphs else static)
