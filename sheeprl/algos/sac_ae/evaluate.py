from __future__ import annotations

import copy
from math import prod
from typing import Any, Dict

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.sac_ae.agent import (
    CNNEncoder,
    MLPEncoder,
    SACAEAgent,
    SACAEContinuousActor,
    SACAECritic,
    SACAEQFunction,
)
from sheeprl.algos.sac_ae.utils import test_sac_ae
from sheeprl.models.models import MultiEncoder
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger, get_log_dir
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="sac_ae")
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

    observation_space = env.observation_space
    action_space = env.action_space
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if cfg.cnn_keys.encoder == [] and cfg.mlp_keys.encoder == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: "
            "`cnn_keys.encoder=[rgb]` or `mlp_keys.encoder=[state]`"
        )
    fabric.print("Encoder CNN keys:", cfg.cnn_keys.encoder)
    fabric.print("Encoder MLP keys:", cfg.mlp_keys.encoder)

    act_dim = prod(action_space.shape)
    target_entropy = -act_dim

    # Define the encoder and decoder and setup them with fabric.
    # Then we will set the critic encoder and actor decoder as the unwrapped encoder module:
    # we do not need it wrapped with the strategy inside actor and critic
    cnn_channels = [prod(observation_space[k].shape[:-2]) for k in cfg.cnn_keys.encoder]
    mlp_dims = [observation_space[k].shape[0] for k in cfg.mlp_keys.encoder]
    cnn_encoder = (
        CNNEncoder(
            in_channels=sum(cnn_channels),
            features_dim=cfg.algo.encoder.features_dim,
            keys=cfg.cnn_keys.encoder,
            screen_size=cfg.env.screen_size,
            cnn_channels_multiplier=cfg.algo.encoder.cnn_channels_multiplier,
        )
        if cfg.cnn_keys.encoder is not None and len(cfg.cnn_keys.encoder) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(
            sum(mlp_dims),
            cfg.mlp_keys.encoder,
            cfg.algo.encoder.dense_units,
            cfg.algo.encoder.mlp_layers,
            eval(cfg.algo.encoder.dense_act),
            cfg.algo.encoder.layer_norm,
        )
        if cfg.mlp_keys.encoder is not None and len(cfg.mlp_keys.encoder) > 0
        else None
    )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder)
    encoder.load_state_dict(state["encoder"])

    # Setup actor and critic. Those will initialize with orthogonal weights
    # both the actor and critic
    actor = SACAEContinuousActor(
        encoder=copy.deepcopy(encoder),
        action_dim=act_dim,
        distribution_cfg=cfg.distribution,
        hidden_size=cfg.algo.actor.hidden_size,
        action_low=action_space.low,
        action_high=action_space.high,
    )
    qfs = [
        SACAEQFunction(
            input_dim=encoder.output_dim, action_dim=act_dim, hidden_size=cfg.algo.critic.hidden_size, output_dim=1
        )
        for _ in range(cfg.algo.critic.n)
    ]
    critic = SACAECritic(encoder=encoder, qfs=qfs)

    # The agent will tied convolutional and linear weights between the encoder actor and critic
    agent = SACAEAgent(
        actor,
        critic,
        target_entropy,
        alpha=cfg.algo.alpha.alpha,
        tau=cfg.algo.tau,
        encoder_tau=cfg.algo.encoder.tau,
        device=fabric.device,
    )
    agent.load_state_dict(state["agent"])
    agent.actor = fabric.setup_module(agent.actor)
    test_sac_ae(agent.actor, fabric, cfg, log_dir)
