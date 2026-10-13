from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.p2e_dv3.p2e_dv3_exploration import P2EDV3Exploration
from sheeprl.algos.p2e_dv3.p2e_dv3_finetuning import P2EDV3Finetuning
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms=["p2e_dv3_exploration", "p2e_dv3_finetuning"])
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    # The task actor of an exploration or of a finetuning plays
    algorithm = P2EDV3Exploration if "exploration" in cfg.algo.name else P2EDV3Finetuning
    evaluate_trained(fabric, cfg, state, algorithm(fabric, cfg))
