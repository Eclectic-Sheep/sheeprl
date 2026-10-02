"""DreamerV3 (and P2E-DV3): the initialization of the weights, the normalization layers of the decoder."""

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from torch import nn

from sheeprl.algos.dreamer_v3.agent import build_models
from sheeprl.algos.dreamer_v3.utils import init_weights
from sheeprl.utils.utils import dotdict


@pytest.mark.parametrize(
    "layer,fan_in,fan_out",
    [
        (nn.Linear(256, 512), 256, 512),
        (nn.Conv2d(32, 64, kernel_size=4), 32 * 16, 64 * 16),
        (nn.ConvTranspose2d(64, 32, kernel_size=4), 64 * 16, 32 * 16),
    ],
)
def test_the_weights_have_the_variance_of_the_average_fan(layer, fan_in, fan_out):
    # The weights of the convolutions were truncated at -2 and 2 instead of 2 standard deviations: the division by the
    # standard deviation of the truncated normal (0.8796) made them 14% larger than the ones of DreamerV3
    torch.manual_seed(0)
    layer.apply(init_weights)
    std = math.sqrt(2 / (fan_in + fan_out))
    weight = layer.weight.detach()
    assert weight.std().item() == pytest.approx(std, rel=0.03)
    assert weight.abs().max().item() <= 2 * std / 0.87962566103423978 + 1e-6


def test_the_cnn_decoder_is_normalized_as_configured():
    # Its normalization layers took the arguments of the ones of the MLP decoder (`observation_model.mlp_layer_norm`)
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v3",
                "env=dummy",
                "algo.cnn_keys.encoder=[rgb]",
                "algo.mlp_keys.encoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.encoder.cnn_channels_multiplier=2",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
                "algo.cnn_layer_norm.kw.eps=0.123",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict(
        {
            "rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
            "state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32),
        }
    )
    world_model, _, _ = build_models(torch.device("cpu"), [3], False, cfg, obs_space)
    cnn_decoder, mlp_decoder = world_model.observation_model.cnn_decoder, world_model.observation_model.mlp_decoder
    assert {m.eps for m in cnn_decoder.modules() if isinstance(m, nn.LayerNorm)} == {0.123}
    assert {m.eps for m in mlp_decoder.modules() if isinstance(m, nn.LayerNorm)} == {1e-3}
