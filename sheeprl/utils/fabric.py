from unittest import mock

import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.strategies import SingleDeviceStrategy, SingleDeviceXLAStrategy


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
