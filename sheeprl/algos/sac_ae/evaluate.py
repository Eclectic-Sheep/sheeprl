from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.sac_ae.agent import SACAEAgent, build_agent
from sheeprl.algos.sac_ae.utils import test_sac_ae
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="sac_ae")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)

    env = make_env(
        cfg,
        cfg.seed,
        0,
        log_dir,
        "test",
        vector_env_idx=0,
    )()

    observation_space = env.observation_space
    action_space = env.action_space
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
    fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    agent: SACAEAgent
    agent, _, _ = build_agent(
        fabric, cfg, observation_space, action_space, state["agent"], state["encoder"], state["decoder"]
    )
    test_sac_ae(agent.actor, fabric, cfg, log_dir)
