from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.p2e_dv2.p2e_dv2_exploration import P2EDV2Exploration
from sheeprl.algos.p2e_dv2.p2e_dv2_finetuning import P2EDV2Finetuning
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms=["p2e_dv2_exploration", "p2e_dv2_finetuning"])
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    # The task actor of an exploration or of a finetuning plays
    algorithm = P2EDV2Exploration if "exploration" in cfg.algo.name else P2EDV2Finetuning
    evaluate_trained(fabric, cfg, state, algorithm(fabric, cfg))
