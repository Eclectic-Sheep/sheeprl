from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.droq.droq import DroQ
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="droq")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    evaluate_trained(fabric, cfg, state, DroQ(fabric, cfg))
