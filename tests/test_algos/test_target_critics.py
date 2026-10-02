"""SAC, DroQ and SAC-AE build the agent of a checkpoint with the target critics of the checkpoint."""

import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl.utils.utils import dotdict


def config(overrides):
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        return dotdict(OmegaConf.to_container(compose(config_name="config", overrides=overrides), resolve=True))


@pytest.mark.parametrize("algo", ["sac", "droq", "sac_ae"])
def test_the_agent_of_a_checkpoint_keeps_its_target_critics(algo):
    # Setting the critics of the agent made the target critics copies of them: a resumed run, an evaluation and a
    # registration replaced the target critics of the checkpoint with the critics
    build_agent = importlib.import_module(f"sheeprl.algos.{algo}.agent").build_agent
    fabric = Fabric(accelerator="cpu", devices=1)
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    if algo == "sac_ae":
        cfg = config(
            [
                "exp=sac_ae",
                "algo.cnn_keys.encoder=[rgb]",
                "algo.mlp_keys.encoder=[state]",
                "algo.hidden_size=8",
                "algo.dense_units=8",
                "algo.cnn_channels_multiplier=1",
                "algo.encoder.features_dim=8",
            ]
        )
        obs_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, (9, 64, 64), np.uint8),
                "state": gym.spaces.Box(-1, 1, (3,), np.float32),
            }
        )
        agent, encoder, decoder, _ = build_agent(fabric, cfg, obs_space, action_space)
        states = (encoder.state_dict(), decoder.state_dict())
    else:
        cfg = config([f"exp={algo}", "algo.mlp_keys.encoder=[state]"])
        obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (3,), np.float32)})
        agent, _ = build_agent(fabric, cfg, obs_space, action_space)
        states = ()

    # A checkpoint whose target critics differ from the critics, as after some training
    saved = {k: v + 1 if "target" in k else v.clone() for k, v in agent.state_dict().items()}
    restored, *_ = build_agent(fabric, cfg, obs_space, action_space, saved, *states)
    targets = {k: v for k, v in restored.state_dict().items() if "target" in k}
    assert len(targets) > 0
    for k, v in targets.items():
        assert torch.equal(v, saved[k]), k
