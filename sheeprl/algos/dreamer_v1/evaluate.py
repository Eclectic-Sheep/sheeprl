from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.dreamer_v1.dreamer_v1 import DreamerV1
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="dreamer_v1")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    evaluate_trained(fabric, cfg, state, DreamerV1(fabric, cfg))
