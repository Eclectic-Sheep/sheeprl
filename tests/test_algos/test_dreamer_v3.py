"""DreamerV3 (and P2E-DV3): the initialization of the weights, the normalization layers of the decoder, the gradient
steps (the representation model, the KL loss, the compiled losses)."""

import math
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
from sheeprl.algos.dreamer_v3.agent import RepresentationModel, build_agent
from sheeprl.algos.dreamer_v3.loss import categorical_kl
from sheeprl.algos.dreamer_v3.utils import init_weights
from sheeprl.utils import compile as compile_utils
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
    world_model, *_ = build_agent(Fabric(accelerator="cpu", devices=1), [3], False, cfg, obs_space)
    cnn_decoder, mlp_decoder = world_model.observation_model.cnn_decoder, world_model.observation_model.mlp_decoder
    assert {m.eps for m in cnn_decoder.modules() if isinstance(m, nn.LayerNorm)} == {0.123}
    assert {m.eps for m in mlp_decoder.modules() if isinstance(m, nn.LayerNorm)} == {1e-3}


def test_the_cnn_decoder_projects_the_latent_state_with_the_official_initialization():
    # The official decoder initializes the projection of the latent state to the first feature maps with the default
    # initializer of its linear layers (uniform with the variance of the average fan), not with the one of the
    # configuration: it was a truncated normal
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v3",
                "env=dummy",
                "algo.cnn_keys.encoder=[rgb]",
                "algo.mlp_keys.encoder=[]",
                "algo.dense_units=8",
                "algo.world_model.encoder.cnn_channels_multiplier=2",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8)})
    torch.manual_seed(0)
    world_model, *_ = build_agent(Fabric(accelerator="cpu", devices=1), [3], False, cfg, obs_space)
    layer = world_model.observation_model.cnn_decoder.model[0]
    weight = layer.weight.detach()
    limit = math.sqrt(3 / ((layer.in_features + layer.out_features) / 2))
    assert weight.abs().max().item() <= limit
    assert weight.std().item() == pytest.approx(limit / math.sqrt(3), rel=0.03)
    assert (weight.abs() > 0.9 * limit).float().mean().item() == pytest.approx(0.1, abs=0.01)
    assert torch.all(layer.bias == 0)


def test_the_player_resets_the_environments_one_by_one(recwarn):
    # After a full reset, the recurrent state was the initial one expanded to the environments, whose rows share the
    # memory: resetting one environment (e.g. at the end of an episode during the random actions, when the player
    # computes no new state) wrote in the expanded tensor (a deprecated `index_put_`, with a warning)
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v3",
                "env=dummy",
                "env.num_envs=3",
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
    *_, player = build_agent(Fabric(accelerator="cpu", devices=1), [3], False, cfg, obs_space)
    player.init_states()
    initial = player.recurrent_state.clone()
    player.recurrent_state[:, 1] += 1.0
    torch.testing.assert_close(player.recurrent_state[:, [0, 2]], initial[:, [0, 2]])
    player.init_states([1])
    torch.testing.assert_close(player.recurrent_state, initial)
    assert not [w for w in recwarn if "expanded tensors" in str(w.message)]


def initial_states_after_a_gradient_step(fabric: Fabric) -> bool:
    """One gradient step of DreamerV3 in every process, each on its own batch: whether the processes then have the same
    learnable initial recurrent state."""
    from sheeprl.algos.dreamer_v3.dreamer_v3 import train
    from sheeprl.algos.dreamer_v3.utils import Moments

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v3",
                "env=dummy",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.mlp_keys.decoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
                "algo.horizon=3",
                "algo.per_rank_batch_size=2",
                "algo.per_rank_sequence_length=4",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    fabric.seed_everything(0)
    world_model, actor, critic, target_critic, _ = build_agent(fabric, [3], False, cfg, obs_space)
    optimizers = fabric.setup_optimizers(
        *[torch.optim.Adam(m.parameters(), lr=1e-2) for m in (world_model, actor, critic)]
    )
    moments = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    # A different batch in every process
    generator = torch.Generator().manual_seed(fabric.global_rank)
    T, B = 4, 2
    data = {
        "state": torch.randn(T, B, 5, generator=generator),
        "actions": torch.nn.functional.one_hot(torch.randint(3, (T, B), generator=generator), 3).float(),
        "rewards": torch.randn(T, B, 1, generator=generator),
        "terminated": torch.zeros(T, B, 1),
        "truncated": torch.zeros(T, B, 1),
        "is_first": torch.zeros(T, B, 1),
    }
    train(fabric, world_model, actor, critic, target_critic, *optimizers, data, None, cfg, False, [3], moments)
    initial_states = fabric.all_gather(world_model.rssm.initial_recurrent_state.detach())
    return torch.equal(initial_states[0], initial_states[1])


def test_the_processes_learn_the_same_initial_recurrent_state():
    # The learnable initial recurrent state is in no module wrapped by DDP: its gradient wasn't averaged over the
    # processes, and each one trained its own
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    assert fabric.launch(initial_states_after_a_gradient_step)


@pytest.mark.parametrize(
    "actor_cls", ["sheeprl.algos.dreamer_v3.agent.Actor", "sheeprl.algos.dreamer_v3.agent.MinedojoActor"]
)
def test_the_actors_of_dreamer_v3_and_p2e_dv3_follow_the_configuration_of_the_actor(actor_cls):
    # The actors ignored `algo.actor.max_std` and `algo.actor.unimix` (the one of the world model, `algo.unimix`, was
    # used), and the exploration actor of P2E-DV3 `algo.actor.action_clip`. The task actor is built as the actor of
    # DreamerV3
    from sheeprl.algos.p2e_dv3.agent import build_agent as build_p2e_agent

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=p2e_dv3_exploration",
                "env=dummy",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.mlp_keys.decoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
                f"algo.actor.cls={actor_cls}",
                "algo.actor.max_std=0.7",
                "algo.actor.unimix=0.2",
                "algo.actor.action_clip=0.3",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    _, _, actor_task, _, _, actor_exploration, *_ = build_p2e_agent(
        Fabric(accelerator="cpu", devices=1), [3], False, cfg, obs_space
    )
    for actor in (actor_task.module, actor_exploration.module):
        assert actor.max_std == 0.7
        assert actor._unimix == 0.2
        assert actor._action_clip == 0.3


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


class RecordingAggregator:
    """A stand-in for the metric aggregator, which keeps the values of the last gradient step."""

    disabled = False

    def __init__(self):
        self.values = {}

    def update(self, name, value):
        self.values[name] = value


def small_dreamer_v3(overrides, accelerator="cpu", precision="32-true"):
    """A small DreamerV3 on images and vectors, with 3 discrete actions, its player, a batch for it, and its gradient
    step."""
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
                *overrides,
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    fabric = Fabric(accelerator=accelerator, devices=1, precision=precision)
    obs_space = gym.spaces.Dict(
        {
            "rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
            "state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32),
        }
    )
    torch.manual_seed(0)
    world_model, actor, critic, target_critic, player = build_agent(fabric, [3], False, cfg, obs_space)
    optimizers = fabric.setup_optimizers(
        *[torch.optim.Adam(module.parameters(), lr=1e-4) for module in (world_model, actor, critic)]
    )
    moments = dreamer_v3.Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
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
    batch = {k: v.to(fabric.device) for k, v in batch.items()}

    def train_step():
        aggregator = RecordingAggregator()
        data = {k: v.clone() for k, v in batch.items()}
        dreamer_v3.train(
            fabric, world_model, actor, critic, target_critic, *optimizers, data, aggregator, cfg, False, [3], moments
        )
        return aggregator.values

    return cfg, world_model, player, train_step


@pytest.mark.parametrize("decoupled_rssm", [False, True])
def test_a_gradient_step_of_dreamer_v3(decoupled_rssm):
    # Both RSSMs: the priors are computed after the unroll, the initial states once per sequence
    _, world_model, _, train_step = small_dreamer_v3([f"algo.world_model.decoupled_rssm={decoupled_rssm}"])
    before = [p.detach().clone() for p in world_model.parameters()]
    metrics = train_step()
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
    assert {"Loss/world_model_loss", "Loss/policy_loss", "Loss/value_loss", "State/kl"} <= set(metrics)
    assert any(not torch.equal(b, a) for b, a in zip(before, world_model.parameters()))


def test_cuda_graphs_are_used_only_in_the_tested_precisions(monkeypatch):
    monkeypatch.setattr(compile_utils, "_WARNED", {})
    cfg = dotdict(
        {"algo": {"compile": {"enabled": True, "mode": "reduce-overhead"}}, "fabric": {"precision": "32-true"}}
    )
    assert compile_utils.compile_mode(cfg) == "reduce-overhead"
    cfg.fabric.precision = "bf16-mixed"
    assert compile_utils.compile_mode(cfg) == "reduce-overhead"
    cfg.fabric.precision = "16-mixed"
    with pytest.warns(UserWarning, match="reduce-overhead"):
        assert compile_utils.compile_mode(cfg) is None


@pytest.mark.parametrize("strategy", ["auto", "ddp"])
def test_the_losses_are_compiled_only_when_enabled(monkeypatch, strategy):
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    fabric = Fabric(accelerator="cpu", devices=1, strategy=strategy)
    cfg = dotdict({"algo": {"compile": {"enabled": False, "mode": None}}, "fabric": {"precision": "32-true"}})
    assert compile_utils.compiled(dreamer_v3.world_model_loss, fabric, cfg) is dreamer_v3.world_model_loss
    # Also with the strategy of several processes: the modules are not wrapped by DistributedDataParallel, whose forward
    # isn't traced by `torch.compile`
    cfg.algo.compile.enabled = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compile_utils.compiled(dreamer_v3.world_model_loss, fabric, cfg) is not dreamer_v3.world_model_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("precision,tolerance", [("32-true", 1e-4), ("bf16-mixed", 2e-2)])
def test_the_compiled_losses_are_the_ones_of_the_eager_losses(monkeypatch, precision, tolerance):
    # The same weights, the same batch and the latents taken as the most likely ones (the compiled code draws other
    # random numbers): the same losses, with and without `torch.compile` (and its CUDA graphs). The modules are
    # compiled without the hook of Lightning on their outputs (`sheeprl.utils.fabric.compilable`)
    original = dv3_agent.compute_stochastic_state
    monkeypatch.setattr(
        dv3_agent,
        "compute_stochastic_state",
        lambda logits, discrete=32, sample=True: original(logits, discrete=discrete, sample=False),
    )
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    losses = []
    for enabled in (False, True):
        _, world_model, _, train_step = small_dreamer_v3(
            [f"algo.compile.enabled={enabled}", f"fabric.precision={precision}"],
            accelerator="cuda",
            precision=precision,
        )
        # The imagined continues are the most likely ones: at the initialization the continue model predicts about
        # 0.5, and the rounding of bf16 changes some of them between the compiled and the eager code. A clear
        # prediction keeps them the same
        with torch.no_grad():
            world_model.continue_model.model[-1].bias.fill_(3.0)
        losses.append({k: torch.as_tensor(v).float().clone() for k, v in train_step().items()})
    for name in ("Loss/world_model_loss", "Loss/observation_loss", "Loss/state_loss", "Loss/value_loss"):
        torch.testing.assert_close(losses[1][name], losses[0][name], rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The weights are channels-last only on CUDA")
def test_the_player_shares_the_channels_last_weights_of_the_world_model():
    # The weights of the convolutions are stored channels-last, also the ones the player shares with the world model:
    # the player still plays with the weights of the last update
    _, world_model, player, train_step = small_dreamer_v3([], accelerator="cuda")
    convolutions = [
        m.weight
        for model in (world_model.encoder, world_model.observation_model)
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    ]
    assert convolutions and all(w.is_contiguous(memory_format=torch.channels_last) for w in convolutions)
    before = [p.detach().clone() for p in player.encoder.parameters()]
    train_step()
    for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
        assert p.data_ptr() == agent_p.data_ptr()
    assert any(not torch.equal(b, p) for b, p in zip(before, player.encoder.parameters()))
