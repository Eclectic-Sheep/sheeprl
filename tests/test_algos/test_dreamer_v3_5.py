"""DreamerV3 as published in Nature (`dreamer_v3_5`): its parts (the two-hot distribution, the lambda-returns, the
optimizer, the layers, the latent states kept in the replay buffer) behave as the official ones."""

from __future__ import annotations

import math
from typing import Any, Dict
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf
from torch import nn

from sheeprl.algos.dreamer_v3_5 import dreamer_v3_5
from sheeprl.algos.dreamer_v3_5.agent import (
    RSSM,
    Actor,
    BlockLinear,
    Conv2d,
    Linear,
    MaxPool,
    RMSNorm,
    Upsample,
    build_agent,
)
from sheeprl.algos.dreamer_v3_5.loss import TwoHot, lambda_return, symexp_bins
from sheeprl.algos.dreamer_v3_5.optim import LaProp
from sheeprl.algos.dreamer_v3_5.utils import Moments
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import SequenceSampler
from sheeprl.data.store import ReplayStore
from sheeprl.utils import compile as compile_utils
from sheeprl.utils.utils import dotdict, symexp

from .compiled import same_random_numbers


def most_likely_latents(monkeypatch: pytest.MonkeyPatch) -> None:
    """The latent states and the actions taken as the most likely ones instead of sampled (with the straight-through
    gradients of the latent states)."""

    def sample(logits):
        probs = logits.softmax(-1)
        one_hot = F.one_hot(probs.argmax(-1), probs.shape[-1]).to(probs.dtype)
        return (one_hot + probs - probs.detach()).flatten(-2)

    monkeypatch.setattr(RSSM, "sample", staticmethod(sample))
    forward = Actor.forward
    monkeypatch.setattr(
        Actor,
        "forward",
        lambda self, state, greedy=False, mask=None, **kwargs: forward(self, state, True, mask, **kwargs),
    )


def compose_cfg(overrides) -> Dict[str, Any]:
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["exp=dreamer_v3_5", "env=dummy", *overrides])
    return dotdict(OmegaConf.to_container(cfg, resolve=True))


def test_the_bins_are_symmetric_and_spaced_by_symexp():
    bins = symexp_bins(255)
    assert bins[127] == 0
    assert torch.equal(bins, -bins.flip(0))
    torch.testing.assert_close(bins, symexp(torch.linspace(-20, 20, 255)), rtol=1e-5, atol=1e-6)
    assert torch.equal(symexp_bins(6), -symexp_bins(6).flip(0))


def test_the_two_hot_of_uniform_logits_predicts_zero():
    # The heads initialized with zero weights predict exactly 0, summed symmetrically from the middle
    assert torch.equal(TwoHot(torch.zeros(4, 255)).mean, torch.zeros(4))


@pytest.mark.parametrize("target", [-12345.6, -3.7, 0.0, 0.5, 123.4, 1e12])
def test_the_two_hot_encoding_has_the_value_as_mean(target):
    # The encoding is linear in the space of the values (the bins are spaced by symexp): its mean is the value, but
    # beyond the last bin
    two_hot = TwoHot(torch.zeros(1, 255))
    encoded = two_hot.encode(torch.tensor([target]))
    assert encoded.sum() == pytest.approx(1.0)
    assert (encoded > 0).sum() <= 2
    expected = min(target, two_hot.bins[-1].item())
    assert (encoded * two_hot.bins.double()).sum().item() == pytest.approx(expected, rel=1e-5)


def official_lambda_return(last, term, rew, boot, disc, lam):
    """`lambda_return` of the official implementation, batch first (NumPy)."""
    rets = [boot[:, -1]]
    live = (1 - term)[:, 1:] * disc
    cont = (1 - last)[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)


def test_the_lambda_returns_are_the_official_ones():
    rng = np.random.default_rng(0)
    B, T = 3, 9
    last = (rng.random((B, T)) < 0.2).astype(np.float64)
    term = last * (rng.random((B, T)) < 0.5)
    rew, boot = rng.normal(size=(B, T)), rng.normal(size=(B, T))
    expected = official_lambda_return(last, term, rew, boot, 0.997, 0.95)
    as_time_first = lambda x: torch.as_tensor(x.T)
    returns = lambda_return(*map(as_time_first, (last, term, rew, boot)), 0.997, 0.95)
    np.testing.assert_allclose(returns.numpy().T, expected, rtol=1e-12)
    # At the end of an episode the return is the reward plus the bootstrap, unless the episode terminated
    returns = lambda_return(
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 2.0, 5.0]),
        torch.tensor([7.0, 3.0, 9.0]),
        0.5,
        0.9,
    )
    assert returns[0].item() == pytest.approx(2.0 + 0.5 * 3.0)


def official_optimizer_steps(params, grads, lr, warmup, agc=0.3, pmin=1e-3, beta1=0.9, beta2=0.999, eps=1e-20):
    """The weights after the steps of the optimizer of the official implementation (`Agent._make_opt`: adaptive
    gradient clipping, RMS scaling, momentum, warmup), in NumPy."""
    params = [p.astype(np.float64) for p in params]
    nu = [np.zeros_like(p) for p in params]
    mu = [np.zeros_like(p) for p in params]
    history = []
    for step, step_grads in enumerate(grads, 1):
        rate = lr * min(1.0, (step - 1) / warmup)
        for i, (p, g) in enumerate(zip(params, step_grads)):
            upper = agc * max(pmin, np.linalg.norm(p))
            g = g / max(1.0, np.linalg.norm(g) / upper)
            nu[i] = beta2 * nu[i] + (1 - beta2) * g * g
            u = g / (np.sqrt(nu[i] / (1 - beta2**step)) + eps)
            mu[i] = beta1 * mu[i] + (1 - beta1) * u
            params[i] = p - rate * mu[i] / (1 - beta1**step)
        history.append([p.copy() for p in params])
    return history


def test_laprop_is_the_optimizer_of_the_official_implementation():
    rng = np.random.default_rng(0)
    # Weights with a small norm (the clipping takes `pmin`), gradients large (clipped) and small (not clipped)
    params = [rng.normal(size=(4, 3)), np.full((5,), 1e-5), rng.normal(0, 0.1, size=(7,))]
    grads = [[rng.normal(size=p.shape) * s for p, s in zip(params, (10.0, 1.0, 1e-3))] for _ in range(6)]
    expected = official_optimizer_steps(params, grads, lr=1e-2, warmup=3)
    weights = [nn.Parameter(torch.tensor(p, dtype=torch.float64)) for p in params]
    optimizer = LaProp(weights, lr=1e-2, warmup=3)
    for step, (step_grads, step_expected) in enumerate(zip(grads, expected)):
        for w, g in zip(weights, step_grads):
            w.grad = torch.tensor(g, dtype=torch.float64)
        optimizer.step()
        for w, e in zip(weights, step_expected):
            np.testing.assert_allclose(w.detach().numpy(), e, rtol=1e-5, atol=1e-12, err_msg=f"step {step}")
    # The first step has a learning rate of 0
    assert np.array_equal(expected[0][0], params[0])
    # The step is in the state of the optimizer
    restored = LaProp([nn.Parameter(w.detach().clone()) for w in weights], lr=1e-2, warmup=3)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.param_groups[0]["step"] == 6


def test_the_block_linear_layer_has_block_diagonal_weights():
    torch.manual_seed(0)
    layer = BlockLinear(12, 8, blocks=4)
    x = torch.randn(5, 12)
    dense = torch.block_diag(*layer.weight.detach())
    torch.testing.assert_close(layer(x), x @ dense + layer.bias)


@pytest.mark.parametrize(
    "layer,fan_in",
    [
        (Linear(512, 256), 512),
        (Conv2d(32, 64, 5), 32 * 25),
        # The fan-in of the whole input, as in the official implementation
        (BlockLinear(1024, 512, blocks=8), 1024),
    ],
)
def test_the_weights_have_the_variance_of_the_fan_in(layer, fan_in):
    std = 1 / math.sqrt(fan_in)
    weight = layer.weight.detach()
    assert weight.std().item() == pytest.approx(std, rel=0.03)
    assert weight.abs().max().item() <= 2 * std / 0.87962566103423978 + 1e-6
    assert torch.all(layer.bias == 0)


def test_the_heads_with_zero_scale_have_zero_weights():
    assert torch.all(Linear(16, 255, outscale=0.0).weight == 0)
    assert Linear(512, 8, outscale=0.01).weight.std().item() == pytest.approx(0.01 / math.sqrt(512), rel=0.2)


def test_the_pooling_and_the_upsampling_of_the_images():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 6)
    # The reshape of the compiled code and the max pooling
    with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
        reshaped = MaxPool()(x)
    torch.testing.assert_close(reshaped, MaxPool()(x), rtol=0, atol=0)
    torch.testing.assert_close(MaxPool()(x), F.max_pool2d(x, 2), rtol=0, atol=0)
    torch.testing.assert_close(Upsample()(x), F.interpolate(x, scale_factor=2, mode="nearest"), rtol=0, atol=0)


def test_the_rms_norm_of_the_channels_normalizes_every_pixel():
    torch.manual_seed(0)
    norm = RMSNorm(3, eps=1e-4, channels_first=True)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
    x = torch.randn(2, 3, 4, 5)
    expected = x * torch.rsqrt(x.square().mean(1, keepdim=True) + 1e-4) * norm.weight.view(1, 3, 1, 1)
    torch.testing.assert_close(norm(x), expected)
    x = x.contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(norm(x), expected)


def test_the_sampled_latents_follow_their_probabilities():
    torch.manual_seed(0)
    logits = torch.tensor([[[2.0, 0.0, -1.0, 0.5]]]).log_softmax(-1).expand(20000, 1, 4).requires_grad_()
    samples = RSSM.sample(logits)
    torch.testing.assert_close(samples.detach().mean(0), logits.detach()[0, 0].exp(), atol=0.015, rtol=0)
    # One-hots, up to the rounding of the straight-through gradients
    torch.testing.assert_close(samples.detach().sum(-1), torch.ones(20000))
    assert torch.all(samples.detach().amax(-1) > 0.999)
    # The straight-through gradients are the ones of the probabilities
    weights = torch.randn(4)
    (samples * weights).sum().backward()
    expected = torch.autograd.grad((logits.softmax(-1).flatten(-2) * weights).sum(), logits)[0]
    torch.testing.assert_close(logits.grad, expected)


def small_dreamer_v3_5(overrides=(), accelerator="cpu", precision="32-true", actions="discrete", peaked=False):
    """A small DreamerV3 (Nature version) on images and vectors, a batch of 4 steps (and their context) for it and its
    gradient step. With `peaked`, the two-hot heads (reward, critic, slow critic) predict distributions peaked on the
    middle bins, as trained ones, instead of uniform ones: the mean of a uniform distribution is 0 only up to the
    rounding of the sum of bins as large as 4.85e8, which depends on the order of the sum (e.g. compiled or not)."""
    cfg = compose_cfg(
        [
            "algo.cnn_keys.encoder=[rgb]",
            "algo.mlp_keys.encoder=[state]",
            "algo.dense_units=8",
            "algo.cnn_channels_multiplier=2",
            "algo.world_model.recurrent_model.recurrent_state_size=16",
            "algo.world_model.recurrent_model.hidden_size=8",
            "algo.world_model.recurrent_model.blocks=4",
            "algo.world_model.observation_model.block_space=2",
            "algo.world_model.stochastic_size=4",
            "algo.world_model.discrete_size=5",
            "algo.horizon=3",
            "algo.per_rank_batch_size=2",
            "algo.per_rank_sequence_length=4",
            f"fabric.precision={precision}",
            *overrides,
        ]
    )
    fabric = Fabric(accelerator=accelerator, devices=1, precision=precision)
    obs_space = gym.spaces.Dict(
        {
            "rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
            "state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32),
        }
    )
    continuous = actions == "continuous"
    actions_dim = [2] if continuous else [3]
    torch.manual_seed(0)
    world_model, actor, critic, target_critic, player = build_agent(fabric, actions_dim, continuous, cfg, obs_space)
    if peaked:
        with torch.no_grad():
            for head in (world_model.reward_model, critic, target_critic):
                bias = head.module[-1].bias
                bias.copy_(-0.3 * (torch.arange(len(bias), device=bias.device) - len(bias) // 2).abs())
    optimizer = fabric.setup_optimizers(
        LaProp([*world_model.parameters(), *actor.parameters(), *critic.parameters()], lr=1e-3, warmup=0)
    )
    moments = Moments(0.99, 1.0, 0.05, 0.95)
    T, B = cfg.algo.per_rank_sequence_length + cfg.algo.replay_context, cfg.algo.per_rank_batch_size
    generator = torch.Generator().manual_seed(1)
    batch = {
        "rgb": torch.randint(0, 256, (T, B, 3, 64, 64), generator=generator, dtype=torch.uint8),
        "state": torch.randn(T, B, 5, generator=generator),
        "rewards": torch.randn(T, B, 1, generator=generator),
        "terminated": torch.zeros(T, B, 1),
        "truncated": torch.zeros(T, B, 1),
        "is_first": (torch.rand(T, B, 1, generator=generator) < 0.3).float(),
        "deter": torch.rand(T, B, 16, generator=generator).half(),
        "stoch": torch.randint(0, 5, (T, B, 4), generator=generator, dtype=torch.uint8),
    }
    if continuous:
        batch["actions"] = torch.rand(T, B, 2, generator=generator) * 2 - 1
    else:
        batch["actions"] = F.one_hot(torch.randint(0, 3, (T, B), generator=generator), 3).float()
    batch = {k: v.to(fabric.device) for k, v in batch.items()}

    def train_step():
        return dreamer_v3_5.train(
            fabric,
            cfg,
            world_model,
            actor,
            critic,
            target_critic,
            optimizer,
            moments,
            {k: v.clone() for k, v in batch.items()},
            actions_dim,
        )

    return cfg, (world_model, actor, critic, target_critic), player, train_step


@pytest.mark.parametrize("actions", ["discrete", "continuous"])
@pytest.mark.parametrize("replay_context", [0, 1])
def test_a_gradient_step_of_dreamer_v3_5(actions, replay_context):
    cfg, models, _, train_step = small_dreamer_v3_5([f"algo.replay_context={replay_context}"], actions=actions)
    world_model, actor, critic, target_critic = models
    before = [[p.detach().clone() for p in m.parameters()] for m in models]
    metrics, latents = train_step()
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
    expected = {
        "Loss/world_model_loss",
        "Loss/policy_loss",
        "Loss/value_loss",
        "Loss/replay_value_loss",
        "State/kl",
        "Grads/agent",
    }
    assert expected <= set(metrics)
    # The world model, the actor and the critic are updated, not the slow critic
    for b, m in zip(before[:3], models[:3]):
        assert any(not torch.equal(x, p) for x, p in zip(b, m.parameters()))
    assert all(torch.equal(x, p) for x, p in zip(before[3], target_critic.parameters()))
    # The latent states of the trained steps, to write back in the replay buffer
    if replay_context == 0:
        assert latents is None
    else:
        deter, stoch = latents
        assert deter.shape == (4, 2, 16) and deter.dtype == torch.float16
        assert stoch.shape == (4, 2, 4) and stoch.dtype == torch.uint8 and stoch.max() < 5


def test_the_episodes_start_from_zeros(monkeypatch):
    # At the first step of an episode the previous latent state and actions are zeroed: the latent states from there
    # are the ones of a sequence starting there
    most_likely_latents(monkeypatch)
    cfg, (world_model, *_), _, _ = small_dreamer_v3_5()
    T, B = 6, 2
    generator = torch.Generator().manual_seed(0)
    obs = {"rgb": torch.randint(0, 256, (T, B, 3, 64, 64), generator=generator), "state": torch.randn(T, B, 5)}
    actions = F.one_hot(torch.randint(0, 3, (T, B), generator=generator), 3).float()
    is_first = torch.zeros(T, B, 1)
    is_first[3] = 1
    kwargs = dreamer_v3_5.world_model_loss_kwargs(cfg)

    def recurrent_states(start, length, carry):
        return dreamer_v3_5.world_model_loss(
            world_model,
            {k: v[start : start + length] for k, v in obs.items()},
            actions[start : start + length],
            is_first[start : start + length],
            torch.zeros(length, B, 1),
            torch.zeros(length, B, 1),
            *carry,
            **kwargs,
        )[3]

    random_carry = (torch.rand(1, B, 16) * 2 - 1, F.one_hot(torch.randint(0, 5, (1, B, 4)), 5).flatten(-2).float())
    zeros = (torch.zeros(1, B, 16), torch.zeros(1, B, 20))
    with torch.no_grad():
        torch.testing.assert_close(recurrent_states(0, T, random_carry)[3:], recurrent_states(3, 3, zeros))


def test_the_player_starts_the_environments_from_zeros():
    _, _, player, _ = small_dreamer_v3_5()
    player.num_envs = 3
    player.init_states()
    obs = {"rgb": torch.rand(1, 3, 3, 64, 64) - 0.5, "state": torch.randn(1, 3, 5)}
    with torch.no_grad():
        actions = player.get_actions(obs)
        # The first recurrent state is 0 (at the initialization the biases are 0): the second one depends on the
        # first stochastic state
        assert torch.all(player.recurrent_state == 0)
        actions = player.get_actions(obs)
    assert len(actions) == 1 and actions[0].shape == (1, 3, 3)
    assert torch.all(actions[0].sum(-1) == 1)
    assert torch.all(player.recurrent_state.abs().sum(-1) > 0)
    player.init_states([1])
    assert torch.all(player.recurrent_state[:, 1] == 0) and torch.all(player.stochastic_state[:, 1] == 0)
    assert torch.all(player.actions[:, 1] == 0)
    assert player.recurrent_state[:, [0, 2]].abs().sum() > 0


def test_the_continuous_actions_of_the_player_are_clipped():
    _, _, player, _ = small_dreamer_v3_5(actions="continuous")
    player.num_envs = 64
    player.init_states()
    obs = {"rgb": torch.rand(1, 64, 3, 64, 64) - 0.5, "state": torch.randn(1, 64, 5)}
    with torch.no_grad():
        (actions,) = player.get_actions(obs)
    assert actions.abs().max() <= 1 and torch.equal(player.actions, actions)


def add_steps(rb: ReplayBuffer, counters: np.ndarray, n: int) -> None:
    """Add `n` steps of every environment, with their identifiers (`dreamer_v3_5.STEP_ID_KEY`)."""
    for _ in range(n):
        rb.add(
            {
                "deter": np.zeros((1, rb.n_envs, 3), np.float16),
                "stoch": np.zeros((1, rb.n_envs, 2), np.uint8),
                dreamer_v3_5.STEP_ID_KEY: np.stack((np.arange(rb.n_envs), counters), -1)[np.newaxis],
            }
        )
        counters += 1


def test_the_latent_states_are_written_back_at_their_steps():
    # The buffer of 5 steps goes around: the steps of the sampled sequences are found from their identifiers
    rb = ReplayBuffer(5, n_envs=2)
    counters = np.zeros(2, np.int64)
    add_steps(rb, counters, 8)
    store = ReplayStore(rb, SequenceSampler(3, seed=0))
    sample, step_ids = dreamer_v3_5.sample_sequences(store, batch_size=4, n_samples=1)
    step_ids = step_ids[0]
    # The written latent states: their step identifiers, as floats
    deter = torch.as_tensor(np.repeat(step_ids[..., 1:].astype(np.float16), 3, -1))
    stoch = torch.as_tensor(np.repeat(step_ids[..., :1].astype(np.uint8), 2, -1))
    dreamer_v3_5.write_latent_states(rb, [(step_ids, deter, stoch)], rb.buffer_size)
    written = 0
    for env in range(rb.n_envs):
        ids = rb[dreamer_v3_5.STEP_ID_KEY][:, env]
        rows = np.isin(ids[:, 1], step_ids[..., 1][step_ids[..., 0] == env])
        assert np.all(rb["deter"][rows, env] == ids[rows, 1:].astype(np.float16))
        assert np.all(rb["stoch"][rows, env] == env)
        assert np.all(rb["deter"][~rows, env] == 0)
        written += rows.sum()
    assert written > 0
    # The steps added to every buffer, from the identifiers it holds
    assert np.array_equal(dreamer_v3_5.step_counters(rb), counters)
    assert np.array_equal(dreamer_v3_5.step_counters(ReplayBuffer(5, n_envs=2)), [0, 0])


def test_the_slow_critic_moves_towards_the_critic():
    critic, slow = nn.Linear(3, 2), nn.Linear(3, 2)
    expected = [0.98 * s.detach() + 0.02 * c.detach() for c, s in zip(critic.parameters(), slow.parameters())]
    dreamer_v3_5.update_target_critic(critic, slow, 0.02)
    for e, s in zip(expected, slow.parameters()):
        torch.testing.assert_close(s.detach(), e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("precision,tolerance", [("32-true", 1e-4), ("bf16-mixed", 2e-2)])
def test_the_compiled_losses_are_the_ones_of_the_eager_losses(monkeypatch, precision, tolerance):
    # The same weights, the same batch and the same random numbers: the same losses, with and without `torch.compile`
    # (and its CUDA graphs)
    same_random_numbers(monkeypatch)
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    losses = []
    for enabled in (False, True):
        _, _, _, train_step = small_dreamer_v3_5(
            [f"algo.compile.enabled={enabled}"], accelerator="cuda", precision=precision, peaked=True
        )
        torch.manual_seed(1)
        metrics, _ = train_step()
        losses.append({k: torch.as_tensor(v).float().clone() for k, v in metrics.items()})
    for name in (
        "Loss/world_model_loss",
        "Loss/observation_loss",
        "Loss/state_loss",
        "Loss/policy_loss",
        "Loss/value_loss",
        "Loss/replay_value_loss",
    ):
        torch.testing.assert_close(losses[1][name], losses[0][name], rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The weights are channels-last only on CUDA")
def test_the_player_shares_the_channels_last_weights_of_the_world_model():
    _, (world_model, *_), player, train_step = small_dreamer_v3_5(accelerator="cuda")
    convolutions = [
        m.weight
        for model in (world_model.encoder, world_model.observation_model)
        for m in model.modules()
        if isinstance(m, nn.Conv2d)
    ]
    assert convolutions and all(w.is_contiguous(memory_format=torch.channels_last) for w in convolutions)
    before = [p.detach().clone() for p in player.encoder.parameters()]
    train_step()
    for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
        assert p.data_ptr() == agent_p.data_ptr()
    assert any(not torch.equal(b, p) for b, p in zip(before, player.encoder.parameters()))
