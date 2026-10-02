import torch
from lightning import Fabric
from lightning.fabric.strategies import SingleDeviceStrategy
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from sheeprl.utils.fabric import compilable, get_single_device_fabric
from sheeprl.utils.fabric import setup_module as fabric_setup_module  # `setup_module` would be a hook of pytest
from sheeprl.utils.fabric import update


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


def set_up_on_two_processes(fabric):
    # Different weights on every process
    torch.manual_seed(fabric.global_rank)
    module = fabric_setup_module(fabric, nn.Linear(3, 2))
    weights = fabric.all_gather(torch.cat([p.detach().flatten() for p in module.parameters()]))
    return not isinstance(module._forward_module, DistributedDataParallel) and torch.equal(weights[0], weights[1])


def test_the_modules_start_from_the_weights_of_rank_0_without_ddp():
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    assert fabric.launch(set_up_on_two_processes)


def one_update(fabric, xs, params_of=None, max_grad_norm=None):
    """One SGD step of a linear model, on the outputs of another module, scaled by a weight in no module (as the
    learnable initial state of Dreamer): the weights and the gradients after the step on the data of the process.
    `params_of(model, scale)` gives the weights to update (default: all the ones of the optimizer)."""
    torch.manual_seed(0)
    model = fabric_setup_module(fabric, nn.Linear(3, 1))
    other = fabric_setup_module(fabric, nn.Linear(3, 3))
    scale = nn.Parameter(torch.ones(1))
    optimizer = fabric.setup_optimizers(torch.optim.SGD([*model.parameters(), scale], lr=1.0))
    loss = sum((model(other(x)) * scale).sum() for x in xs) / len(xs)
    params = params_of(model, scale) if params_of is not None else None
    grad_norm = update(fabric, loss, optimizer, max_grad_norm=max_grad_norm, params=params)
    weights = torch.cat([model.weight.detach().flatten(), model.bias.detach(), scale.detach()])
    return weights, model, other, scale, grad_norm


def update_on_two_processes(fabric):
    xs = [torch.arange(3.0), torch.arange(3.0) * 2 - 1]
    # Every process steps on its own data, with the gradients averaged over the processes: the step on all the data
    weights, _, other, _, _ = one_update(fabric, [xs[fabric.global_rank]])
    expected, *_ = one_update(get_single_device_fabric(fabric), xs)
    gathered = fabric.all_gather(weights)
    return (
        torch.equal(gathered[0], gathered[1])
        and torch.allclose(weights, expected)
        and all(p.grad is None for p in other.parameters())
    )


def test_an_update_averages_the_gradients_of_its_weights_over_the_processes():
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    assert fabric.launch(update_on_two_processes)


def test_an_update_computes_the_gradients_of_its_weights_only():
    fabric = Fabric(accelerator="cpu", devices=1)
    x = [torch.arange(3.0)]
    _, model, other, scale, grad_norm = one_update(fabric, x, max_grad_norm=1e-3)
    # The weights that took part in the loss but that the optimizer doesn't update get no gradients
    assert all(p.grad is None for p in other.parameters())
    assert grad_norm is not None and grad_norm > 1e-3
    grads = torch.cat([model.weight.grad.flatten(), model.bias.grad, scale.grad])
    torch.testing.assert_close(torch.linalg.vector_norm(grads), torch.tensor(1e-3))
    # Only the given weights: the others of the optimizer keep their values
    _, model, _, scale, grad_norm = one_update(fabric, x, params_of=lambda model, scale: [scale])
    assert model.weight.grad is None and scale.grad is not None and grad_norm is None
