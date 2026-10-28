from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.ppo.ppo import PPO
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation
from sheeprl.utils.utils import dotdict


@register_evaluation(algorithms="ppo")
def evaluate_ppo(fabric: Fabric, cfg: dotdict, state: Dict[str, Any]):
    evaluate_trained(fabric, cfg, state, PPO(fabric, cfg))
