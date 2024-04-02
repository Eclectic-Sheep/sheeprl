from unittest import mock

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
