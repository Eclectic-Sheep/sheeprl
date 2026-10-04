from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.a2c.a2c import A2C
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="a2c")
def evaluate_a2c(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    evaluate_trained(fabric, cfg, state, A2C(fabric, cfg))
