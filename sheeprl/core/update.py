"""Device, precision and optimizer steps: the parts of training that are the same for every algorithm.

Modules are not wrapped in `DistributedDataParallel`: with several processes, `update` averages the gradients
itself, with one all-reduce per optimizer step and only for the weights that step updates.
"""

from __future__ import annotations

import itertools
from contextlib import AbstractContextManager
from typing import Any, Iterable, List, Optional

import torch
from lightning import Fabric
from lightning.fabric.plugins.precision.amp import MixedPrecision
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.optim import Optimizer

from sheeprl.utils.fabric import autocast_cache_scope, get_single_device_fabric


class _CompilableFabricModule(_FabricModule):
    """The module of `setup_module`: a `_FabricModule` that can be compiled with `torch.compile` in mixed precision.

    In mixed precision, `_FabricModule` registers a hook on the outputs that checks, during the backward pass, that it
    goes through `fabric.backward` (as `update` does): traced by `torch.compile`, the hook breaks the graph at every
    call of the module. While compiling, this module does the conversions of the mixed precision (the inputs to the
    precision, the outputs to the default type) without the hook. Otherwise it is a `_FabricModule`.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        precision = self._strategy.precision
        if not torch.compiler.is_compiling() or not isinstance(precision, MixedPrecision):
            return super().forward(*args, **kwargs)
        args, kwargs = precision.convert_input((args, kwargs))
        with precision.forward_context():
            output = self._forward_module(*args, **kwargs)
        return precision.convert_output(output)


def setup_module(fabric: Fabric, module: nn.Module) -> _FabricModule:
    """Move `module` to the device and set its precision, like `fabric.setup_module` but without the
    `DistributedDataParallel` wrapper. With several processes, the weights of rank 0 are copied to the other
    processes, as `DistributedDataParallel` does, so that every process starts from the same weights.

    The returned module runs every call in the precision of `fabric` (`fabric.precision`).
    """
    module = get_single_device_fabric(fabric).setup_module(module)
    # The same object, with the forward of `_CompilableFabricModule` (`_FabricModule.__setattr__` would also change the
    # class of the wrapped module)
    object.__setattr__(module, "__class__", _CompilableFabricModule)
    if fabric.world_size > 1:
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            torch.distributed.broadcast(tensor.data, src=0)
    return module


def autocast(fabric: Fabric) -> AbstractContextManager:
    """The precision scope of one update phase: the forward passes and the loss of one optimizer step.

    Close it before calling `update`. The modules prepared by `setup_module` run every call in the precision of
    `fabric`; this scope keeps autocast's low-precision copies of the weights for the whole phase, instead of
    casting the weights again at every call (see `sheeprl.utils.fabric.autocast_cache_scope`). The copies are not
    updated by the optimizer: that's why the scope must be closed before the update.
    """
    return autocast_cache_scope(fabric)


def update(
    fabric: Fabric,
    loss: Tensor,
    optimizer: Optimizer,
    max_grad_norm: float = 0.0,
    params: Optional[Iterable[Tensor]] = None,
    error_if_nonfinite: bool = True,
) -> Optional[Tensor]:
    """One optimizer step on `loss`.

    The gradients are computed only for `params` (default: the weights of `optimizer`), averaged over the
    processes, clipped to a total norm of `max_grad_norm` (when greater than 0) and applied by `optimizer`.
    The backward pass doesn't compute the gradients of the other weights that took part in the loss.

    Args:
        fabric: the fabric of the run, which handles the precision of the backward pass and of the step.
        loss: the scalar to minimize.
        optimizer: the optimizer of the weights to update, set up with `fabric.setup_optimizers`.
        max_grad_norm: the maximum norm of the gradients; 0 to not clip them.
        params: the weights to compute the gradients of; default: all the weights of `optimizer`.
        error_if_nonfinite: when clipping, raise if the norm of the gradients is not finite.

    Returns:
        The norm of the gradients before clipping, as a tensor; `None` if they are not clipped.
    """
    params = [p for group in optimizer.param_groups for p in group["params"]] if params is None else list(params)
    optimizer.zero_grad(set_to_none=True)
    fabric.backward(loss, inputs=params)
    all_reduce_gradients(fabric, params)
    grad_norm = None
    if max_grad_norm > 0.0:
        # The first argument (the module) is used only by FSDP, which SheepRL doesn't use
        grad_norm = fabric.clip_gradients(
            None, optimizer, max_norm=max_grad_norm, error_if_nonfinite=error_if_nonfinite
        )
    optimizer.step()
    return grad_norm


def all_reduce_gradients(fabric: Fabric, params: List[Tensor]) -> None:
    """Average the gradients of `params` over the processes, with one all-reduce of all of them.

    Every process must compute the gradients of the same weights, which holds when the computation doesn't depend on
    the data. Does nothing with one process.
    """
    if fabric.world_size == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return
    flat = fabric.all_reduce(torch.cat([g.reshape(-1) for g in grads]), reduce_op="mean")
    offset = 0
    for g in grads:
        g.copy_(flat[offset : offset + g.numel()].view_as(g))
        offset += g.numel()
