import torch
from lightning import Fabric
from lightning.fabric.strategies import SingleDeviceStrategy
from torch import nn

from sheeprl.utils.fabric import compilable, get_single_device_fabric


def test_get_single_device_fabric():
    fabric = Fabric(devices=2, accelerator="cpu", precision=16)
    single_device_fabric = get_single_device_fabric(fabric)
    assert single_device_fabric.device == fabric.device
    assert single_device_fabric._precision == fabric._precision
    assert single_device_fabric.accelerator == fabric.accelerator
    assert isinstance(single_device_fabric.strategy, SingleDeviceStrategy)


def test_a_compilable_module_is_compiled_in_one_graph_in_mixed_precision():
    # In mixed precision, the hook of `_FabricModule` on the outputs broke the compiled graph at every call of the
    # module. Traced without generating code (`aot_eager`): the same computations as the eager ones
    fabric = Fabric(accelerator="cpu", devices=1, precision="bf16-mixed")
    torch.manual_seed(0)
    module = compilable(fabric.setup_module(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))))
    x = torch.randn(3, 4)
    gradients = []
    for loss_fn in (
        lambda x: module(x).square().mean(),
        torch.compile(lambda x: module(x).square().mean(), fullgraph=True, backend="aot_eager"),
    ):
        module.zero_grad()
        loss = loss_fn(x)
        # The output of the module in the default type, as without compiling
        assert loss.dtype == torch.float32
        fabric.backward(loss)
        gradients.append([p.grad.clone() for p in module.parameters()])
    for compiled, eager in zip(*gradients):
        torch.testing.assert_close(compiled, eager)
