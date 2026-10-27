"""The inputs and outputs of the agents (`sheeprl.models.io`) and the networks of their contract."""

from typing import List

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from torch import nn
from torch.distributions import Independent, Normal

from sheeprl.models import ConvEncoder, IdentityEncoder, MLPEncoder, NatureCNN
from sheeprl.models.io import Input, MultiDecoder, MultiEncoder, build_decoder, build_encoder
from sheeprl.transforms import check_input, check_net
from sheeprl.utils.utils import dotdict

STORE_SPACE = gym.spaces.Dict(
    {
        "rgb": gym.spaces.Box(0, 255, (3, 64, 64), np.uint8),
        "depth": gym.spaces.Box(0, 255, (1, 64, 64), np.uint8),
        "state": gym.spaces.Box(-1, 1, (4,), np.float32),
        "speed": gym.spaces.Box(-1, 1, (2,), np.float32),
        "fov": gym.spaces.Box(0, 1, (5, 5, 5), np.float32),
    }
)
SCALE = {"_target_": "sheeprl.transforms.Scale", "divisor": 255, "shift": 0.5}


@pytest.mark.parametrize(
    "net,input_shape",
    [
        (lambda: NatureCNN((4, 64, 48), features_dim=16), (4, 64, 48)),
        (lambda: NatureCNN((3, 36, 36), features_dim=None), (3, 36, 36)),
        (lambda: MLPEncoder((6,), features_dim=8, dense_units=8), (6,)),
        (lambda: MLPEncoder((6,), mlp_layers=0), (6,)),
        (lambda: ConvEncoder((5, 5, 5), channels=[8, 8]), (5, 5, 5)),
        (lambda: ConvEncoder((5, 5, 5), channels=[8], features_dim=4), (5, 5, 5)),
        (lambda: IdentityEncoder((5, 5)), (5, 5)),
    ],
)
def test_the_networks_follow_the_contract_of_the_inputs(net, input_shape):
    check_net(net(), input_shape)


def test_nature_cnn_is_sized_from_the_images():
    # 64 x 48: the flattened feature maps of the convolutions are 4 x 2 x 64
    assert NatureCNN((3, 64, 48), features_dim=None).output_dim == 64 * 4 * 2


def test_an_input_scales_and_concatenates_its_images_along_the_channels():
    inp = Input.from_config(
        {
            "keys": ["rgb", "depth"],
            "transforms": [SCALE],
            "net": {"_target_": "sheeprl.models.NatureCNN", "features_dim": 8},
        },
        STORE_SPACE,
    )
    assert inp.cat_dim == -3 and inp.net.input_shape == (4, 64, 64) and inp.output_dim == 8
    obs = {
        "rgb": torch.full((2, 3, 64, 64), 255, dtype=torch.uint8),
        "depth": torch.zeros(2, 1, 64, 64, dtype=torch.uint8),
    }
    rgb, depth = inp.transformed(obs)
    assert torch.equal(rgb, torch.full((2, 3, 64, 64), 0.5)) and torch.equal(depth, torch.full((2, 1, 64, 64), -0.5))


def test_an_input_of_images_and_vectors_must_say_how_to_concatenate_them():
    mlp = {"_target_": "sheeprl.models.MLPEncoder", "features_dim": 4}
    with pytest.raises(ValueError, match="Flatten"):
        Input.from_config({"keys": ["rgb", "state"], "net": mlp}, STORE_SPACE)
    # Flattened, they are vectors
    flatten = {"_target_": "sheeprl.transforms.Flatten", "where": "batch"}
    inp = Input.from_config({"keys": ["fov", "state"], "transforms": [flatten], "net": mlp}, STORE_SPACE)
    assert inp.net.model._model[0].in_features == 125 + 4


def test_check_input_compiles_an_input_of_images_in_one_graph():
    torch._dynamo.reset()
    cfg = {"keys": ["rgb"], "transforms": [SCALE], "net": {"_target_": "sheeprl.models.NatureCNN", "features_dim": 8}}
    check_input(cfg, STORE_SPACE, compile=True)


def experiment_inputs(exp: str, *overrides: str) -> dotdict:
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"exp={exp}", "algo.inputs.image.keys=[rgb]", "algo.inputs.vector.keys=[state]", *overrides],
        )
    return dotdict(OmegaConf.to_container(cfg, resolve=True)).algo.inputs


@pytest.mark.parametrize("exp", ["ppo", "a2c", "ppo_recurrent"])
def test_the_inputs_of_the_on_policy_algorithms_compile_in_one_graph(exp):
    for spec in experiment_inputs(exp).values():
        check_input(spec, STORE_SPACE, compile=True)


@pytest.mark.parametrize("exp", ["dreamer_v1", "dreamer_v2", "dreamer_v3", "dreamer_v3_5"])
def test_the_inputs_of_the_dreamers_compile_in_one_graph(exp):
    # Shrunk: the shipped sizes compile the same graph, only slower
    inputs = experiment_inputs(exp, "algo.inputs.image.net.channels_multiplier=4", "algo.dense_units=8")
    assert set(inputs) == {"image", "vector"}
    for spec in inputs.values():
        check_input(spec, STORE_SPACE, compile=True)


def test_a_multi_encoder_concatenates_the_features_of_its_inputs_in_order_and_skips_the_empty_ones():
    inputs = experiment_inputs("ppo")
    inputs.image["keys"] = []
    inputs["extra"] = {"keys": ["speed", "state"], "net": {"_target_": "sheeprl.models.IdentityEncoder"}}
    encoder = MultiEncoder.from_config(inputs, STORE_SPACE)
    assert list(encoder.inputs) == ["vector", "extra"]
    # Every key once, in the order of the inputs
    assert encoder.keys == ("state", "speed")
    obs = {"state": torch.randn(3, 4), "speed": torch.randn(3, 2)}
    features = encoder(obs)
    assert features.shape == (3, encoder.inputs["vector"].output_dim + 6)
    torch.testing.assert_close(features[:, -6:], torch.cat([obs["speed"], obs["state"]], -1))
    with pytest.raises(ValueError, match="at least one input"):
        MultiEncoder.from_config({"image": {"keys": []}}, STORE_SPACE)


class LinearDecoder(nn.Module):
    """A decoder network of the contract: a linear layer reshaped to `output_shape`."""

    def __init__(self, latent_size: int, output_shape: List[int]) -> None:
        super().__init__()
        self.output_shape = tuple(output_shape)
        self.linear = nn.Linear(latent_size, int(np.prod(output_shape)))

    def forward(self, latent):
        return self.linear(latent).reshape(*latent.shape[:-1], *self.output_shape)


def unit_normal(reconstruction, dims):
    return Independent(Normal(reconstruction, 1), dims)


def test_a_decoder_reconstructs_its_keys_in_the_domain_of_the_encoder_and_scores_them():
    inputs = experiment_inputs("ppo")
    encoder = MultiEncoder.from_config(inputs, STORE_SPACE)
    decoder_net = {"_target_": "tests.test_models.test_io.LinearDecoder"}
    likelihood = {"_target_": "tests.test_models.test_io.unit_normal", "_partial_": True}
    decoder = MultiDecoder.from_config(
        {
            "image": {"keys": ["rgb"], "net": decoder_net, "likelihood": likelihood},
            "vector": {"keys": ["state"], "net": decoder_net, "likelihood": likelihood},
        },
        encoder,
        latent_size=6,
    )
    reconstructions = decoder(torch.zeros(2, 6))
    assert reconstructions["rgb"].shape == (2, 3, 64, 64) and reconstructions["state"].shape == (2, 4)
    obs = {"rgb": torch.full((2, 3, 64, 64), 255, dtype=torch.uint8), "state": torch.zeros(2, 4)}
    targets = encoder.targets(obs, decoder.keys)
    # The targets of the images are scaled as the encoder reads them
    assert torch.equal(targets["rgb"], torch.full((2, 3, 64, 64), 0.5))
    log_prob = decoder.log_prob(reconstructions, targets)
    expected = unit_normal(reconstructions["rgb"], 3).log_prob(targets["rgb"]) + unit_normal(
        reconstructions["state"], 1
    ).log_prob(targets["state"])
    torch.testing.assert_close(log_prob, expected)


def test_a_grid_is_reconstructed_by_an_mlp_and_scored_by_a_bernoulli_likelihood():
    # Issue #237: the 125 values of the decoder reshaped to the grid are the logits of its cells, which are 0 or 1
    conv = {"_target_": "sheeprl.models.ConvEncoder", "channels": [4], "features_dim": 4}
    encoder = MultiEncoder.from_config({"grid": {"keys": ["fov"], "net": conv}}, STORE_SPACE)
    mlp = {"_target_": "sheeprl.algos.dreamer_v3.agent.MLPDecoder", "mlp_layers": 1, "dense_units": 8}
    bernoulli = {"_target_": "sheeprl.utils.distribution.BernoulliLogits", "_partial_": True}
    decoder = MultiDecoder.from_config(
        {"grid": {"keys": ["fov"], "net": mlp, "likelihood": bernoulli}}, encoder, latent_size=6
    )
    logits = decoder(torch.randn(3, 2, 6))["fov"]
    assert logits.shape == (3, 2, 5, 5, 5)
    targets = encoder.targets({"fov": torch.randint(0, 2, (3, 2, 5, 5, 5)).float()}, decoder.keys)
    expected = -nn.functional.binary_cross_entropy_with_logits(logits, targets["fov"], reduction="none")
    torch.testing.assert_close(decoder.log_prob({"fov": logits}, targets), expected.sum(dim=(-3, -2, -1)))


class RecordingEncoder(MultiEncoder):
    """An encoder class set by `algo.encoder._target_`."""


class RecordingDecoder(MultiDecoder):
    """A decoder class set by `algo.decoder._target_`."""


def hook_cfg(**sections) -> dotdict:
    net = {"_target_": "sheeprl.models.ConvEncoder", "channels": [4], "features_dim": 4}
    decoder_net = {"_target_": "tests.test_models.test_io.LinearDecoder"}
    algo = {
        "inputs": {"grid": {"keys": ["fov"], "net": net}},
        "outputs": {"grid": {"keys": ["fov"], "net": decoder_net}},
    }
    return dotdict({"algo": {**algo, **sections}})


def test_the_encoder_and_the_decoder_are_of_the_configured_classes():
    # Without the sections, the default classes
    encoder = build_encoder(hook_cfg(), STORE_SPACE)
    assert type(encoder) is MultiEncoder and type(build_decoder(hook_cfg(), encoder, 6)) is MultiDecoder
    cfg = hook_cfg(
        encoder={"_target_": "tests.test_models.test_io.RecordingEncoder"},
        decoder={"_target_": "tests.test_models.test_io.RecordingDecoder"},
    )
    encoder = build_encoder(cfg, STORE_SPACE)
    decoder = build_decoder(cfg, encoder, latent_size=6)
    assert type(encoder) is RecordingEncoder and type(decoder) is RecordingDecoder
    # The names, the keys and the networks of the inputs and of the outputs, through the contracts
    assert encoder.named_keys() == {"grid": ("fov",)} and decoder.named_keys() == {"grid": ("fov",)}
    assert isinstance(encoder.named_nets()["grid"], ConvEncoder)
    assert isinstance(decoder.named_nets()["grid"], LinearDecoder)


def test_the_encoder_and_the_decoder_classes_must_follow_the_contracts():
    with pytest.raises(TypeError, match="sheeprl.models.io.Encoder"):
        build_encoder(hook_cfg(encoder={"_target_": "torch.nn.Linear"}), STORE_SPACE)
    encoder = build_encoder(hook_cfg(), STORE_SPACE)
    with pytest.raises(TypeError, match="sheeprl.models.io.Decoder"):
        build_decoder(hook_cfg(decoder={"_target_": "torch.nn.Linear"}), encoder, 6)


def test_the_images_of_sac_ae_are_reconstructed_as_their_pixels_with_5_bits():
    # The output `image` of SAC-AE declares its target transform (`Dequantize`); its encoder reads the images in [0, 1]
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=sac_ae",
                "algo.inputs.image.keys=[rgb]",
                "algo.inputs.vector.keys=[]",
                "algo.outputs.vector.keys=[]",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    encoder = build_encoder(cfg, STORE_SPACE)
    decoder = build_decoder(cfg, encoder, latent_size=encoder.output_dim)
    pixels = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    target = decoder.targets({"rgb": pixels}, encoder)["rgb"]
    assert torch.equal(torch.floor((target + 0.5) * 32), torch.floor(pixels / 8))
