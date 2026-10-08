from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.sac_ae.sac_ae import SACAE
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="sac_ae")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    evaluate_trained(fabric, cfg, state, SACAE(fabric, cfg))
