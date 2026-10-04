"""The losses of the algorithms compiled with `torch.compile` (`algo.compile`)."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from lightning.fabric import Fabric

# The precisions in which the losses are compiled with CUDA graphs (`algo.compile.mode=reduce-overhead`)
CUDA_GRAPHS_PRECISIONS = ("32-true", "32", 32, "bf16-mixed")


def compile_mode(cfg: Dict[str, Any]) -> Optional[str]:
    """The mode of `torch.compile` for the losses (`algo.compile.mode`), `None` for the default one. CUDA graphs
    (`reduce-overhead`) are used only in the precisions where they have been tested (`CUDA_GRAPHS_PRECISIONS`)."""
    mode = (cfg.algo.get("compile") or {}).get("mode", None)
    if mode == "reduce-overhead" and cfg.fabric.precision not in CUDA_GRAPHS_PRECISIONS:
        if not _WARNED.get("reduce-overhead"):
            warnings.warn(
                f"`algo.compile.mode=reduce-overhead` (CUDA graphs) is not used with `fabric.precision="
                f"{cfg.fabric.precision}`: the losses are compiled with the default mode"
            )
            _WARNED["reduce-overhead"] = True
        return None
    return mode


def compile_enabled(fabric: Fabric, cfg: Dict[str, Any]) -> bool:
    """Whether the losses are compiled (`algo.compile.enabled`), also with several processes: the modules are not
    wrapped by `DistributedDataParallel` (`sheeprl.utils.fabric.setup_module`), whose forward `torch.compile` doesn't
    trace, and the gradients are averaged after the backward pass (`sheeprl.utils.fabric.update`)."""
    return bool((cfg.algo.get("compile") or {}).get("enabled", False))


def compiled(fn: Callable, fabric: Fabric, cfg: Dict[str, Any], cuda_graphs: bool = True) -> Callable:
    """`fn` compiled with `torch.compile` when `algo.compile.enabled` is set (compiled once, at the first call).

    Without `cuda_graphs`, `reduce-overhead` falls back to the default mode: the gradients of a loss accumulated over
    several backward passes (e.g. A2C) live in the memory of the CUDA graphs, which the next replay overwrites."""
    if not compile_enabled(fabric, cfg):
        return fn
    mode = compile_mode(cfg)
    if not cuda_graphs and mode == "reduce-overhead":
        mode = None
    if (fn, mode) not in _COMPILED:
        _COMPILED[fn, mode] = torch.compile(fn, mode=mode)
    return _COMPILED[fn, mode]


def mark_gradient_step(fabric: Fabric, cfg: Dict[str, Any]) -> None:
    """A new gradient step: with CUDA graphs, the outputs of the ones of the previous step, already used, can be
    overwritten."""
    if compile_enabled(fabric, cfg) and compile_mode(cfg) == "reduce-overhead":
        torch.compiler.cudagraph_mark_step_begin()


_COMPILED: Dict[Tuple[Callable, Optional[str]], Callable] = {}
_WARNED: Dict[str, bool] = {}
