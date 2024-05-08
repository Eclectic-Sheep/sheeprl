from lightning import Fabric
from lightning.fabric.strategies import SingleDeviceStrategy

from sheeprl.utils.fabric import get_single_device_fabric


def test_get_single_device_fabric():
    fabric = Fabric(devices=2, accelerator="cpu", precision=16)
    single_device_fabric = get_single_device_fabric(fabric)
    assert single_device_fabric.device == fabric.device
    assert single_device_fabric._precision == fabric._precision
    assert single_device_fabric.accelerator == fabric.accelerator
    assert isinstance(single_device_fabric.strategy, SingleDeviceStrategy)
