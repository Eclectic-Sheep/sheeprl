from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import hydra
from lightning import Fabric

from sheeprl.algos.mopo.agent import build_agent
from sheeprl.algos.mopo.utils import test
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation
from sheeprl.utils.utils import dotdict


@register_evaluation(algorithms="mopo")
def evaluate(fabric: Fabric, cfg: dotdict[str, Any], state: dict[str, Any]):
    """Evaluate the agent from a checkpoint.

    Args:
        fabric: The fabric instance.
        cfg: The configs used for training.
        state: The state of the models.
    """
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    env: gym.Wrapper = hydra.utils.instantiate(cfg.env.wrapper)
    if cfg.env.capture_video and fabric.global_rank == 0 and log_dir is not None:
        env = gym.experimental.wrappers.RecordVideoV0(env, os.path.join(log_dir, "eval_videos"), disable_logger=True)
        env.metadata["render_fps"] = getattr(env, "frames_per_sec", 30)

    observation_space = env.observation_space
    action_space = env.action_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
    fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    # Create the actor and critic models
    _, _, player = build_agent(
        fabric,
        cfg,
        observation_space,
        action_space,
        state,
    )
    del _
    test(player, fabric, env, cfg, 0)
    env.close()
