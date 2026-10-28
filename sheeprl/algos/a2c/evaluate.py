from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.a2c.a2c import A2C
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation
from sheeprl.utils.utils import dotdict


@register_evaluation(algorithms="a2c")
def evaluate_a2c(fabric: Fabric, cfg: dotdict, state: Dict[str, Any]):
    evaluate_trained(fabric, cfg, state, A2C(fabric, cfg))
