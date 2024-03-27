from __future__ import annotations

from functools import partial
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import torch
from lightning.fabric.plugins.precision.amp import MixedPrecision, _optimizer_handles_unscaling
from lightning.fabric.utilities.types import Optimizable
from torch import Tensor, nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import LBFGS, Optimizer


class MixedPrecisionMultiScaler(MixedPrecision):
    def __init__(
        self,
        precision: Literal["16-mixed"] | Literal["bf16-mixed"],
        device: str,
        scaler: GradScaler | None = None,
        clip_interval: Tuple[float, float] | None = None,
    ) -> None:
        super().__init__(precision, device, scaler)
        self._scaler_kwargs = None
        if scaler is not None:
            self._scaler_kwargs = {
                "init_scale": scaler._init_scale,
                "growth_factor": scaler.get_growth_factor(),
                "backoff_factor": scaler.get_backoff_factor(),
                "growth_interval": scaler.get_growth_interval(),
                "enabled": scaler.is_enabled(),
            }
        self.scaler = None
        self._clip_fn = None
        if clip_interval is not None:
            self._clip_fn = partial(torch.clip, min=clip_interval[0], max=clip_interval[1])

    def setup_scalers(self, optimizer_ids: Sequence[str]):
        if self._scaler_kwargs is not None:
            self.scaler = {id: GradScaler(**self._scaler_kwargs) for id in optimizer_ids}

    def backward(
        self, tensor: Tensor, model: Optional[nn.Module], optim_id: str | None = None, *args: Any, **kwargs: Any
    ) -> None:
        if self.scaler is not None and optim_id is not None:
            tensor = self.scaler[optim_id].scale(tensor)
        super(MixedPrecision, self).backward(tensor, model, *args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        optim_id = getattr(optimizer, "optim_id", None)
        if self.scaler is None or optim_id is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super(MixedPrecision, self).optimizer_step(optimizer, **kwargs)
        if isinstance(optimizer, LBFGS):
            raise TypeError("AMP and the LBFGS optimizer are not compatible.")
        # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
        step_output = self.scaler[optim_id].step(optimizer, **kwargs)
        self.scaler[optim_id].update()
        self._clip_scale(optim_id)
        return step_output

    def state_dict(self) -> Dict[str, Any]:
        if self.scaler is not None:
            return {k: scaler.state_dict() for k, scaler in self.scaler.items()}
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.scaler is not None:
            [self.scaler[k].load_state_dict(state_dict[k]) for k in state_dict]

    def unscale_gradients(self, optimizer: Optimizer) -> None:
        optim_id = getattr(optimizer, "optim_id", None)
        if self.scaler is not None and optim_id is not None:
            scaler: GradScaler = self.scaler[optim_id]
            if _optimizer_handles_unscaling(optimizer):
                raise NotImplementedError("Gradient clipping is not implemented for optimizers handling the unscaling.")
            scaler.unscale_(optimizer)

    def _clip_scale(self, optim_id: str | None):
        if self.scaler is not None and optim_id is not None and self._clip_fn is not None:
            scale = torch.tensor(self.scaler[optim_id].get_scale())
            self.scaler[optim_id].update(self._clip_fn(scale).item())
