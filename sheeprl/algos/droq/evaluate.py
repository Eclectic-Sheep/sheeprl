from __future__ import annotations

from math import prod
from typing import Any, Dict

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.droq.agent import DROQAgent, DROQCritic
from sheeprl.algos.sac.agent import SACActor
from sheeprl.algos.sac.utils import test
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger, get_log_dir
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="droq")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = create_tensorboard_logger(fabric, cfg)
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
    if len(cfg.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
    for k in cfg.mlp_keys.encoder:
        if len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the DroQ agent. "
                f"Provided environment: {cfg.env.id}"
            )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder MLP keys:", cfg.mlp_keys.encoder)

    act_dim = prod(action_space.shape)
    obs_dim = sum([prod(observation_space[k].shape) for k in cfg.mlp_keys.encoder])
    actor = SACActor(
        observation_dim=obs_dim,
        action_dim=act_dim,
        distribution_cfg=cfg.distribution,
        hidden_size=cfg.algo.actor.hidden_size,
        action_low=action_space.low,
        action_high=action_space.high,
    )
    critics = [
        DROQCritic(
            observation_dim=obs_dim + act_dim,
            hidden_size=cfg.algo.critic.hidden_size,
            num_critics=1,
            dropout=cfg.algo.critic.dropout,
        )
        for _ in range(cfg.algo.critic.n)
    ]
    target_entropy = -act_dim
    agent = DROQAgent(
        actor, critics, target_entropy, alpha=cfg.algo.alpha.alpha, tau=cfg.algo.tau, device=fabric.device
    )
    agent.load_state_dict(state["agent"])
    agent = fabric.setup_module(agent)
    test(agent.actor, fabric, cfg, log_dir)
