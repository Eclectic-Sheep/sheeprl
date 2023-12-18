from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.droq.agent import build_agent
from sheeprl.algos.sac.utils import test
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="droq")
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
    action_space = env.action_space
    observation_space = env.observation_space
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the DroQ agent")
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.algo.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
    for k in cfg.algo.mlp_keys.encoder:
        if len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the DroQ agent. "
                f"Provided environment: {cfg.env.id}"
            )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    agent = build_agent(fabric, cfg, observation_space, action_space, state["agent"])
    test(agent.actor, fabric, cfg, log_dir)
