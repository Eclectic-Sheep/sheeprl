from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.dreamer_v2.dreamer_v2 import DreamerV2
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation
from sheeprl.utils.utils import dotdict


@register_evaluation(algorithms="dreamer_v2")
def evaluate(fabric: Fabric, cfg: dotdict, state: Dict[str, Any]):
    evaluate_trained(fabric, cfg, state, DreamerV2(fabric, cfg))
