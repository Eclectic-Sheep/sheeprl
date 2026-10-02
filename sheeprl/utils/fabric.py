"""Device, precision and optimizer steps of the modules.

The modules are not wrapped in `DistributedDataParallel` (`setup_module`): with several processes, `update` averages
the gradients itself, with one all-reduce per optimizer step and only for the weights that step updates.
"""

import itertools
from typing import Any, Iterable, List, Optional
from unittest import mock

import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.plugins.precision.amp import MixedPrecision
from lightning.fabric.strategies import SingleDeviceStrategy, SingleDeviceXLAStrategy
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.optim import Optimizer


def get_single_device_fabric(fabric: Fabric) -> Fabric:
    """Get a single device fabric. The returned fabric will share the same accelerator,
    precision and device as the input fabric. This is useful when you want to create a new
    fabric with the same device as the input fabric, but with a strategy running on a single
    device.

    Args:
        fabric (Fabric): The fabric to use as a base.

    Returns:
        Fabric: A new fabric with the same device, precision and accelerator as the input fabric but with
        a single-device strategy.
    """
    strategy_cls = SingleDeviceXLAStrategy if isinstance(fabric.accelerator, XLAAccelerator) else SingleDeviceStrategy
    strategy = strategy_cls(
        device=fabric.device,
        accelerator=fabric.accelerator,
        checkpoint_io=None,
        precision=fabric._precision,
    )
    with mock.patch.dict("os.environ") as mocked_os_environ:
        mocked_os_environ.pop("LT_DEVICES", None)
        mocked_os_environ.pop("LT_STRATEGY", None)
        mocked_os_environ.pop("LT_NUM_NODES", None)
        mocked_os_environ.pop("LT_PRECISION", None)
        mocked_os_environ.pop("LT_ACCELERATOR", None)
        fabric = Fabric(strategy=strategy)
    return fabric


def autocast_cache_scope(fabric: Fabric) -> torch.autocast:
    """Keep autocast's cache of the low-precision copies of the weights alive inside the returned context.

    Every call of a module set up with Fabric opens and closes its own autocast context, and PyTorch drops the cache
    of the casted weights when the outermost autocast context exits: without an enclosing context, the weights of a
    module called in a loop (e.g., the RSSM unroll of Dreamer) are casted again at every call. The returned context
    is an enclosing, *disabled* autocast context: the weights are casted once per scope, while the code between the
    module calls keeps running in full precision. The weights must not be updated inside the scope (e.g., with
    `optimizer.step()`), since the cached copies would become stale.

    Args:
        fabric (Fabric): the fabric instance.

    Returns:
        The context manager.
    """
    return torch.autocast(device_type=fabric.device.type, enabled=False)


class _CompilableFabricModule(_FabricModule):
    """A `_FabricModule` that can be compiled with `torch.compile` in mixed precision.

    In mixed precision, `_FabricModule` registers a hook on the outputs that checks, during the backward pass, that it
    goes through `fabric.backward`: traced by `torch.compile`, the hook breaks the graph at every call of the module.
    While compiling, this module does the conversions of the mixed precision (the inputs to the precision, the outputs
    to the default type) without the hook: the algorithms that compile always back-propagate with `fabric.backward`.
    Otherwise it is a `_FabricModule`.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        precision = self._strategy.precision
        if not torch.compiler.is_compiling() or not isinstance(precision, MixedPrecision):
            return super().forward(*args, **kwargs)
        args, kwargs = precision.convert_input((args, kwargs))
        with precision.forward_context():
            output = self._forward_module(*args, **kwargs)
        return precision.convert_output(output)


def compilable(module: _FabricModule) -> _CompilableFabricModule:
    """The module set up by Fabric, which `torch.compile` can compile also in mixed precision
    (`_CompilableFabricModule`): the same object, with the forward of `_CompilableFabricModule`
    (`_FabricModule.__setattr__` would change the class of the wrapped module)."""
    object.__setattr__(module, "__class__", _CompilableFabricModule)
    return module


def setup_module(fabric: Fabric, module: nn.Module) -> _CompilableFabricModule:
    """Move `module` to the device and set its precision, like `fabric.setup_module` but without the
    `DistributedDataParallel` wrapper: `update` averages the gradients over the processes. With several processes, the
    weights of rank 0 are copied to the other processes, as `DistributedDataParallel` does, so that every process starts
    from the same weights.

    The returned module runs every call in the precision of `fabric` and can be compiled with `torch.compile`
    (`compilable`). It can be shared by the player: it is set up on the device of the process only.
    """
    module = compilable(get_single_device_fabric(fabric).setup_module(module))
    if fabric.world_size > 1:
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            torch.distributed.broadcast(tensor.data, src=0)
    return module


def update(
    fabric: Fabric,
    loss: Tensor,
    optimizer: Optimizer,
    max_grad_norm: Optional[float] = None,
    params: Optional[Iterable[Tensor]] = None,
    error_if_nonfinite: bool = True,
) -> Optional[Tensor]:
    """One optimizer step on `loss`.

    The gradients are computed only for `params` (default: the weights of `optimizer`), averaged over the processes,
    clipped to a total norm of `max_grad_norm` (when given and greater than 0) and applied by `optimizer`. The backward
    pass doesn't compute the gradients of the other weights that took part in the loss.

    Args:
        fabric: the fabric of the run, which handles the precision of the backward pass and of the step.
        loss: the scalar to minimize.
        optimizer: the optimizer of the weights to update, set up with `fabric.setup_optimizers`.
        max_grad_norm: the maximum norm of the gradients; `None` or 0 to not clip them.
        params: the weights to compute the gradients of; default: all the weights of `optimizer`.
        error_if_nonfinite: when clipping, raise if the norm of the gradients is not finite.

    Returns:
        The norm of the gradients before clipping; `None` if they are not clipped.
    """
    params = [p for group in optimizer.param_groups for p in group["params"]] if params is None else list(params)
    optimizer.zero_grad(set_to_none=True)
    fabric.backward(loss, inputs=params)
    all_reduce_gradients(fabric, params)
    grad_norm = None
    if max_grad_norm is not None and max_grad_norm > 0:
        # The first argument (the module) is used only by FSDP
        grad_norm = fabric.clip_gradients(
            None, optimizer, max_norm=max_grad_norm, error_if_nonfinite=error_if_nonfinite
        )
    optimizer.step()
    return grad_norm


def all_reduce_gradients(fabric: Fabric, params: List[Tensor]) -> None:
    """Average the gradients of `params` over the processes, with one all-reduce of all of them.

    Every process must compute the gradients of the same weights, which holds when the computation doesn't depend on
    the data (as `DistributedDataParallel` requires). Does nothing with one process.
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
