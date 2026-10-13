"""The optimizer of DreamerV3 (Nature version): LaProp with adaptive gradient clipping and a linear warmup of the
learning rate, the chain of `Agent._make_opt` in https://github.com/danijar/dreamerv3/blob/main/dreamerv3/agent.py."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer


class LaProp(Optimizer):
    """LaProp (https://arxiv.org/abs/2002.04839): the gradients are divided by the square root of their running second
    moment first, and the momentum is applied to the result (Adam does the opposite). With:

    - adaptive gradient clipping (https://arxiv.org/abs/2102.06171), before everything else: the gradient of every
      tensor of weights is scaled down to a norm of at most `agc` times the norm of the weights (at least `agc_pmin`);
    - both moments corrected for their bias, as in Adam;
    - a learning rate that grows linearly from 0 to `lr` in the first `warmup` steps (0 at the first one);
    - a weight decay added to the update (`weight_decay`, not decoupled from the learning rate).

    The moments are kept in float32 at least, whatever the precision of the weights.

    Args:
        params: the weights to optimize, or their groups.
        lr: the learning rate. Default: 4e-5.
        betas: the decay rates of the momentum and of the second moment. Default: (0.9, 0.999).
        eps: added to the square root of the second moment. Default: 1e-20.
        agc: the largest ratio between the norms of the gradient and of the weights of every tensor; 0 to not clip.
            Default: 0.3.
        agc_pmin: the smallest norm of the weights used by the clipping. Default: 1e-3.
        warmup: the steps of the linear warmup of the learning rate; 0 to not warm up. Default: 1000.
        weight_decay: the coefficient of the weight decay. Default: 0.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 4e-5,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-20,
        agc: float = 0.3,
        agc_pmin: float = 1e-3,
        warmup: int = 1000,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if agc < 0 or warmup < 0 or weight_decay < 0:
            raise ValueError(
                f"`agc` ({agc}), `warmup` ({warmup}) and `weight_decay` ({weight_decay}) must be non-negative"
            )
        defaults = dict(
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            agc=agc,
            agc_pmin=agc_pmin,
            warmup=warmup,
            weight_decay=weight_decay,
            step=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if len(params) == 0:
                continue
            beta1, beta2 = group["betas"]
            # The moments and the computations in float32 at least
            dtypes = [torch.promote_types(p.dtype, torch.float32) for p in params]
            grads = [p.grad.to(dtype) for p, dtype in zip(params, dtypes)]
            for p, dtype in zip(params, dtypes):
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
            exp_avgs = [self.state[p]["exp_avg"] for p in params]
            exp_avg_sqs = [self.state[p]["exp_avg_sq"] for p in params]

            if group["agc"] > 0:
                grad_norms = torch.stack(torch._foreach_norm(grads))
                param_norms = torch.stack(torch._foreach_norm([p.to(d) for p, d in zip(params, dtypes)]))
                upper = group["agc"] * torch.clamp(param_norms, min=group["agc_pmin"])
                scales = 1 / torch.clamp(grad_norms / upper, min=1.0)
                grads = torch._foreach_mul(grads, list(scales.unbind()))

            # The learning rate of the warmup: 0 at the first step
            warmup = group["warmup"]
            lr = group["lr"] * min(1.0, group["step"] / warmup) if warmup > 0 else group["lr"]
            group["step"] += 1
            step = group["step"]

            # The gradients normalized by the square root of their second moment (corrected for its bias)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
            denom = torch._foreach_sqrt(torch._foreach_div(exp_avg_sqs, 1 - beta2**step))
            torch._foreach_add_(denom, group["eps"])
            updates = torch._foreach_div(grads, denom)

            # Their momentum (corrected for its bias)
            torch._foreach_lerp_(exp_avgs, updates, 1 - beta1)
            updates = torch._foreach_div(exp_avgs, 1 - beta1**step)
            if group["weight_decay"] > 0:
                torch._foreach_add_(updates, [p.to(d) for p, d in zip(params, dtypes)], alpha=group["weight_decay"])
            if lr == 0:
                continue
            if all(p.dtype == dtype for p, dtype in zip(params, dtypes)):
                torch._foreach_add_(params, updates, alpha=-lr)
            else:
                for p, update in zip(params, updates):
                    p.copy_(p.to(update.dtype) - lr * update)
        return loss
