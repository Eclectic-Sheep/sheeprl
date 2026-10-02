"""DreamerV3 (and P2E-DV3): the initialization of the weights, the normalization layers of the decoder, the gradient
steps (the representation model, the KL loss, the compiled losses)."""

import math
import tempfile
import warnings

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf
from torch import nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence

from sheeprl.algos.dreamer_v3 import agent as dv3_agent
from sheeprl.algos.dreamer_v3 import dreamer_v3
from sheeprl.algos.dreamer_v3.agent import RepresentationModel, build_models
from sheeprl.algos.dreamer_v3.loss import categorical_kl
from sheeprl.algos.dreamer_v3.utils import init_weights
from sheeprl.core import TrainSchedule
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


def test_the_player_resets_the_environments_one_by_one(recwarn):
    # After a full reset, the recurrent state was the initial one expanded to the environments, whose rows share the
    # memory: resetting one environment (e.g. at the end of an episode during the random actions, when the player
    # computes no new state) wrote in the expanded tensor (a deprecated `index_put_`, with a warning)
    from sheeprl.algos.dreamer_v3.agent import PlayerDV3

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v3",
                "env=dummy",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    world_model, actor, _ = build_models(torch.device("cpu"), [3], False, cfg, obs_space)
    world_model_cfg = cfg.algo.world_model
    player = PlayerDV3(
        world_model.encoder,
        world_model.rssm,
        actor,
        [3],
        3,
        world_model_cfg.stochastic_size,
        world_model_cfg.recurrent_model.recurrent_state_size,
        torch.device("cpu"),
        discrete_size=world_model_cfg.discrete_size,
    )
    player.init_states()
    initial = player.recurrent_state.clone()
    player.recurrent_state[:, 1] += 1.0
    torch.testing.assert_close(player.recurrent_state[:, [0, 2]], initial[:, [0, 2]])
    player.init_states([1])
    torch.testing.assert_close(player.recurrent_state, initial)
    assert not [w for w in recwarn if "expanded tensors" in str(w.message)]


def test_the_representation_model_computes_the_part_of_the_observations_once():
    # The first layer of the representation model takes the recurrent state and the embedded observation: the part of
    # the observations can be computed for a whole sequence at once
    torch.manual_seed(0)
    model = RepresentationModel(
        4,
        input_dims=4 + 6,
        output_dim=5,
        hidden_sizes=[8],
        activation=nn.SiLU,
        layer_args={"bias": False},
        flatten_dim=None,
        norm_layer=[nn.LayerNorm],
        norm_args=[{"normalized_shape": 8}],
    ).double()
    recurrent_states, observations = torch.randn(3, 2, 4).double(), torch.randn(3, 2, 6).double()
    projection = model(observations=observations)
    torch.testing.assert_close(
        model(recurrent_state=recurrent_states, observation_projection=projection),
        model(torch.cat((recurrent_states, observations), -1)),
    )


def test_the_kl_of_the_latents_is_the_one_of_pytorch():
    torch.manual_seed(0)
    p, q = torch.randn(5, 4, 8, 16), torch.randn(5, 4, 8, 16)
    expected = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=p), 1),
        Independent(OneHotCategoricalStraightThrough(logits=q), 1),
    )
    torch.testing.assert_close(categorical_kl(p, q), expected, rtol=0, atol=0)


def small_dreamer_v3(overrides, accelerator="cpu", precision="32-true"):
    """A small DreamerV3 on images and vectors, with 3 discrete actions, and a batch for it."""
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
                "algo.horizon=3",
                "algo.per_rank_batch_size=2",
                "algo.per_rank_sequence_length=4",
                "metric.log_level=0",
                *overrides,
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    fabric = Fabric(accelerator=accelerator, devices=1, precision=precision)
    algo = dreamer_v3.DreamerV3(fabric, cfg)
    obs_space = gym.spaces.Dict(
        {
            "rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
            "state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32),
        }
    )
    schedule = TrainSchedule(cfg, 1, algo.steps_per_iteration, off_policy=True)
    torch.manual_seed(0)
    state, _ = algo.build(obs_space, gym.spaces.Discrete(3), schedule, tempfile.mkdtemp())
    T, B = cfg.algo.per_rank_sequence_length, cfg.algo.per_rank_batch_size
    generator = torch.Generator().manual_seed(1)
    batch = {
        "rgb": torch.randint(0, 256, (T, B, 3, 64, 64), generator=generator).float(),
        "state": torch.randn(T, B, 5, generator=generator),
        "actions": nn.functional.one_hot(torch.randint(0, 3, (T, B), generator=generator), 3).float(),
        "rewards": torch.randn(T, B, 1, generator=generator),
        "terminated": torch.zeros(T, B, 1),
        "truncated": torch.zeros(T, B, 1),
        "is_first": (torch.rand(T, B, 1, generator=generator) < 0.3).float(),
    }
    return cfg, algo, state, {k: v.to(fabric.device) for k, v in batch.items()}


@pytest.mark.parametrize("decoupled_rssm", [False, True])
def test_a_gradient_step_of_dreamer_v3(decoupled_rssm):
    # Both RSSMs: the priors are computed after the unroll, the initial states once per sequence
    _, algo, state, batch = small_dreamer_v3([f"algo.world_model.decoupled_rssm={decoupled_rssm}"])
    before = [p.detach().clone() for p in state.world_model.parameters()]
    metrics = algo.train_step(state, batch, 0)
    assert all(torch.isfinite(v).all() for v in metrics.values())
    assert {"Loss/world_model_loss", "Loss/policy_loss", "Loss/value_loss", "State/kl"} <= set(metrics)
    assert any(not torch.equal(b, a) for b, a in zip(before, state.world_model.parameters()))


def test_cuda_graphs_are_used_only_in_the_tested_precisions():
    cfg = dotdict(
        {"algo": {"compile": {"enabled": True, "mode": "reduce-overhead"}}, "fabric": {"precision": "32-true"}}
    )
    assert dreamer_v3.compile_mode(cfg) == "reduce-overhead"
    cfg.fabric.precision = "bf16-mixed"
    assert dreamer_v3.compile_mode(cfg) == "reduce-overhead"
    cfg.fabric.precision = "16-mixed"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert dreamer_v3.compile_mode(cfg) is None
    # Not compiled unless enabled
    cfg.algo.compile.enabled = False
    assert dreamer_v3.compiled(dreamer_v3.world_model_loss, cfg) is dreamer_v3.world_model_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("precision,tolerance", [("32-true", 1e-4), ("bf16-mixed", 2e-2)])
def test_the_compiled_losses_are_the_ones_of_the_eager_losses(monkeypatch, precision, tolerance):
    # The same weights, the same batch and the latents taken as the most likely ones (the compiled code draws other
    # random numbers): the same losses, with and without `torch.compile` (and its CUDA graphs). In mixed precision the
    # modules are compiled without the hook of Lightning on their outputs (`sheeprl.core.update.setup_module`)
    original = dv3_agent.compute_stochastic_state
    monkeypatch.setattr(
        dv3_agent,
        "compute_stochastic_state",
        lambda logits, discrete=32, sample=True: original(logits, discrete=discrete, sample=False),
    )
    monkeypatch.setattr(dreamer_v3, "_COMPILED", {})
    losses = []
    for enabled in (False, True):
        _, algo, state, batch = small_dreamer_v3(
            [f"algo.compile.enabled={enabled}", f"fabric.precision={precision}"],
            accelerator="cuda",
            precision=precision,
        )
        losses.append(algo.train_step(state, batch, 0))
    for name in ("Loss/world_model_loss", "Loss/observation_loss", "Loss/state_loss", "Loss/value_loss"):
        torch.testing.assert_close(losses[1][name], losses[0][name], rtol=tolerance, atol=tolerance)
