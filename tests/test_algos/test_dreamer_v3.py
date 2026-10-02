"""DreamerV3 (and P2E-DV3): the initialization of the weights."""

import math

import pytest
import torch
from torch import nn

from sheeprl.algos.dreamer_v3.utils import init_weights


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
