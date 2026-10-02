from typing import Any
from unittest import mock

import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.plugins.precision.amp import MixedPrecision
from lightning.fabric.strategies import SingleDeviceStrategy, SingleDeviceXLAStrategy
from lightning.fabric.wrappers import _FabricModule


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
