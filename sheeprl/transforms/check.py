"""Checks of a transform, of a network and of an input: what a transform or a network written for SheepRL must pass."""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from sheeprl.transforms.base import Transform


def _assert_in(space: gym.spaces.Box, x: np.ndarray, what: str) -> None:
    assert x.shape == space.shape, f"{what}: shape {x.shape}, the space promises {space.shape}"
    assert x.dtype == space.dtype, f"{what}: dtype {x.dtype}, the space promises {space.dtype}"
    if np.issubdtype(x.dtype, np.number):
        within = np.all(x >= space.low) and np.all(x <= space.high)
        assert within, f"{what}: values outside the bounds of the space [{space.low.min()}, {space.high.max()}]"


def _samples(space: gym.Space, n: int, seed: int = 0) -> list:
    space.seed(seed)
    return [space.sample() for _ in range(n)]


def check_transform(transform: Transform, space: gym.Space, n: int = 8, compile: bool = False) -> None:
    """Draw `n` samples from `space`, run `transform` at its placement and assert that its outputs have the shape, the
    dtype and the bounds its `space` promises. With `compile`, assert that its `batch` compiles in one graph
    (`torch._dynamo.explain`)."""
    out_space = transform.space(space)
    samples = _samples(space, n)
    if transform.where == "frame":
        transform.reset()
        info = {"terminated": False, "truncated": False}
        for i, sample in enumerate(samples):
            out = transform.frame(sample, info) if transform.needs_info else transform.frame(sample)
            _assert_in(out_space, np.asarray(out), f"{type(transform).__name__}.frame, sample {i}")
    elif transform.where == "envs":
        rows = np.stack(samples)
        for update in (True, False):
            out = transform.envs(rows, update=update, dones=np.zeros(n, dtype=bool))
            assert len(out) == n, f"{type(transform).__name__}.envs returned {len(out)} rows for {n}"
            for i, row in enumerate(out):
                _assert_in(out_space, np.asarray(row), f"{type(transform).__name__}.envs, row {i}")
    else:
        x = torch.as_tensor(np.stack(samples))
        out = transform.batch(x)
        expected_dtype = torch.from_numpy(np.zeros((), dtype=out_space.dtype)).dtype
        assert out.dtype == expected_dtype, f"{type(transform).__name__}.batch: dtype {out.dtype}"
        for i, row in enumerate(out.detach().cpu().numpy()):
            _assert_in(out_space, row, f"{type(transform).__name__}.batch, row {i}")
        if compile:
            _assert_one_graph(transform.batch, (x,), f"{type(transform).__name__}.batch")


def _assert_one_graph(fn: Any, args: Tuple[Any, ...], what: str) -> None:
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(fn)(*args)
    assert explanation.graph_count == 1 and explanation.graph_break_count == 0, (
        f"{what} compiles in {explanation.graph_count} graphs with {explanation.graph_break_count} breaks: "
        f"{[str(b.reason) for b in explanation.break_reasons]}"
    )


def check_net(net: nn.Module, input_shape: Tuple[int, ...]) -> None:
    """Assert that the encoder network `net` (built for samples of shape `input_shape`) exposes `output_dim` and
    returns it for inputs with one and two leading dimensions."""
    assert isinstance(getattr(net, "output_dim", None), int), f"{type(net).__name__} must expose `output_dim: int`"
    for leading in ((2,), (3, 2)):
        x = torch.rand(*leading, *input_shape)
        out = net(x)
        assert out.shape == (*leading, net.output_dim), (
            f"{type(net).__name__}: output shape {tuple(out.shape)} for an input of shape {tuple(x.shape)}, "
            f"expected {(*leading, net.output_dim)}"
        )


def check_input(cfg_input: Any, store_space: gym.spaces.Dict, n: int = 4, compile: bool = False) -> None:
    """Build the input configured by `cfg_input` (`sheeprl.models.io.Input`) from the spaces of the store and run it on
    `n` samples, in both phases: its features must have the shape `(n, output_dim)`. With `compile`, assert that its
    forward compiles in one graph."""
    from sheeprl.models.io import Input

    model = Input.from_config(cfg_input, store_space)
    obs = {k: torch.as_tensor(np.stack(_samples(store_space[k], n))) for k in model.keys}
    for phase in ("act", "train"):
        features = model(obs, phase=phase)
        assert features.shape == (n, model.output_dim), (
            f"The input of the keys {list(model.keys)} returned features of shape {tuple(features.shape)}, "
            f"expected {(n, model.output_dim)} ({phase})"
        )
    if compile:
        _assert_one_graph(model, (obs,), f"The input of the keys {list(model.keys)}")
