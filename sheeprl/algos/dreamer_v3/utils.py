from typing import Any

import torch
from lightning import Fabric
from torch import Tensor


class Moments(torch.nn.Module):
    def __init__(
        self,
        fabric: Fabric,
        decay: float = 0.99,
        max_: float = 1e8,
        perclo: float = 0.05,
        perchi: float = 0.95,
    ) -> None:
        super().__init__()
        self._fabric = fabric
        self._decay = decay
        self._max = torch.tensor(max_)
        self._perclo = perclo
        self._perchi = perchi
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor) -> Any:
        gathered_x = self._fabric.all_gather(x).detach()
        low = torch.quantile(gathered_x, self._perclo)
        high = torch.quantile(gathered_x, self._perchi)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()
