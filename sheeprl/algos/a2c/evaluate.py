from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.a2c.agent import build_agent
from sheeprl.algos.a2c.utils import test
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="a2c")
def evaluate_a2c(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    env = make_env(
        cfg,
        cfg.seed,
        0,
        log_dir,
        "test",
        vector_env_idx=0,
    )()
    observation_space = env.observation_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.algo.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `algo.mlp_keys.encoder=[state]`")
    for k in cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder:
        if k in observation_space.keys() and len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the A2C agent. "
                f"The observation with key '{k}' has shape {observation_space[k].shape}. "
                f"Provided environment: {cfg.env.id}"
            )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    # Create the actor and critic models
    _, agent = build_agent(fabric, actions_dim, is_continuous, cfg, observation_space, state["agent"])
    del _
    test(agent, fabric, cfg, log_dir)
