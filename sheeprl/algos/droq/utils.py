from __future__ import annotations

from typing import Any, Dict, Sequence

import gymnasium as gym
import mlflow
from lightning import Fabric
from mlflow.models.model import ModelInfo

from sheeprl.algos.droq.agent import build_agent
from sheeprl.algos.sac.utils import AGGREGATOR_KEYS as sac_aggregator_keys
from sheeprl.algos.sac.utils import MODELS_TO_REGISTER as sac_models_to_register
from sheeprl.utils.utils import unwrap_fabric

AGGREGATOR_KEYS = sac_aggregator_keys
MODELS_TO_REGISTER = sac_models_to_register


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence[ModelInfo]:
    # Create the models
    agent = build_agent(fabric, cfg, env.observation_space, env.action_space, state["agent"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["agent"] = mlflow.pytorch.log_model(unwrap_fabric(agent), artifact_path="agent")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
