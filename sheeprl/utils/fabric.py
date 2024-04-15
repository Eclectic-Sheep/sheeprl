import warnings
from datetime import timedelta
from typing import Any, Literal, Optional
from unittest import mock

import torch
import torch.distributed
from lightning.fabric import Fabric
from lightning.fabric.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator, XLAAccelerator
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import SingleDeviceStrategy, SingleDeviceXLAStrategy
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry


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


class SingleDeviceDDPStrategy(DDPStrategy):
    """Strategy for multi-process single-device training on one or multiple nodes."""

    def __init__(
        self,
        accelerator: Accelerator,
        parallel_devices: Optional[int] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
        **kwargs: Any,
    ) -> None:
        if parallel_devices is None:
            warnings.warn("The `parallel_devices` argument is not set. Defaulting to 1 device.")
            parallel_devices = 1
        elif not isinstance(parallel_devices, int):
            raise ValueError("`parallel_devices` must be an integer.")
        process_group_backend = "gloo"
        if isinstance(accelerator, CUDAAccelerator):
            parallel_devices = [torch.device("cuda", 0) for _ in range(parallel_devices)]
        elif isinstance(accelerator, XLAAccelerator):
            parallel_devices = [torch.device("xla", 0) for _ in range(parallel_devices)]
        elif isinstance(accelerator, CPUAccelerator):
            parallel_devices = [torch.device("cpu") for _ in range(parallel_devices)]
        elif isinstance(accelerator, MPSAccelerator):
            parallel_devices = [torch.device("mps", 0) for _ in range(parallel_devices)]
        else:
            raise ValueError("Unsupported accelerator: {}.".format(accelerator))
        super().__init__(
            accelerator=None,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method=start_method,
            **kwargs,
        )

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        entries = (
            ("single_ddp", "popen"),
            ("single_ddp_spawn", "spawn"),
            ("single_ddp_fork", "fork"),
            ("single_ddp_notebook", "fork"),
        )
        for name, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `start_method={start_method!r}` and a single device on GLOO.",
                start_method=start_method,
            )
