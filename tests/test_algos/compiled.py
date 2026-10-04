"""Helpers of the tests that compare the compiled losses of an algorithm (`algo.compile`) with the eager ones."""

import torch


def recording(module, monkeypatch, *modules):
    """An aggregator recording the losses and a list recording the gradients of every update of `module.train` (they
    stay in the weights after the step of the optimizer), also of the ones of the functions of `modules` it calls."""
    losses, grads = [], []
    update = module.update

    def recording_update(fabric, loss, optimizer, *args, params=None, **kwargs):
        params = None if params is None else list(params)
        out = update(fabric, loss, optimizer, *args, params=params, **kwargs)
        updated = [p for group in optimizer.param_groups for p in group["params"]] if params is None else params
        grads.append([None if p.grad is None else p.grad.detach().clone() for p in updated])
        return out

    class Aggregator:
        disabled = False

        def update(self, name, value):
            losses.append((name, value.detach().clone()))

    for m in (module, *modules):
        monkeypatch.setattr(m, "update", recording_update)
    return Aggregator(), losses, grads


def assert_same_step(eager, compiled):
    """The same losses and the same gradients: every tensor up to 0.1% of its norm, or 0.01% of the norm of all the
    gradients of its update (the gradients that nearly cancel out are dominated by the rounding errors)."""
    (eager_losses, eager_grads), (compiled_losses, compiled_grads) = eager, compiled
    assert [name for name, _ in compiled_losses] == [name for name, _ in eager_losses]
    for (name, e), (_, c) in zip(eager_losses, compiled_losses):
        torch.testing.assert_close(c, e, rtol=1e-4, atol=1e-6, msg=name)
    assert len(eager_grads) == len(compiled_grads) > 0
    for i, (eager_update, compiled_update) in enumerate(zip(eager_grads, compiled_grads)):
        assert [g is None for g in eager_update] == [g is None for g in compiled_update]
        total = torch.cat([g.double().flatten() for g in eager_update if g is not None]).norm()
        for j, (e, c) in enumerate(zip(eager_update, compiled_update)):
            if e is not None:
                gap = (c - e).double().norm()
                assert gap <= 1e-3 * e.double().norm() + 1e-4 * total, (i, j, tuple(e.shape), gap, e.norm(), total)


def same_random_numbers(monkeypatch):
    """The compiled code draws the random numbers of the eager code (`torch._inductor.config.fallback_random`): seeded
    alike, the two sample the same latent states and actions. The matrix products, the convolutions and the recurrent
    layers run without TF32 and with the algorithms that cuDNN picks without benchmarking them (whose rounding errors
    differ: the CLI turns the benchmarks on), and the distributions without the validation of their arguments, as in
    training."""
    import torch._inductor.config

    monkeypatch.setattr(torch._inductor.config, "fallback_random", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    monkeypatch.setattr(torch.backends.cudnn.conv, "fp32_precision", "ieee")
    monkeypatch.setattr(torch.backends.cudnn.rnn, "fp32_precision", "ieee")
    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)


def no_host_reads(monkeypatch):
    """`Tensor.item` raises: a loss that reads a tensor on the host synchronizes with the device at every step."""

    def item(self):
        raise AssertionError("The losses read a tensor on the host")

    monkeypatch.setattr(torch.Tensor, "item", item)
