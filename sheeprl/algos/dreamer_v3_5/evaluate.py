from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.dreamer_v3_5.dreamer_v3_5 import DreamerV3_5
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="dreamer_v3_5")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    evaluate_trained(fabric, cfg, state, DreamerV3_5(fabric, cfg))
