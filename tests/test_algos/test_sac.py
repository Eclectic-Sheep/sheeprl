"""SAC, DroQ and SAC-AE: their losses compile into single graphs, which with CUDA graphs give the losses of the eager
ones, and the target critics move towards the critics with the formula of the loop over their weights."""

import copy
from types import SimpleNamespace

import gymnasium as gym
import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl.algos.droq import droq
from sheeprl.algos.droq.agent import build_agent as droq_build_agent
from sheeprl.algos.sac import sac
from sheeprl.algos.sac.agent import build_agent as sac_build_agent
from sheeprl.algos.sac_ae import sac_ae
from sheeprl.algos.sac_ae.agent import build_agent as sac_ae_build_agent
from sheeprl.utils import compile as compile_utils
from sheeprl.utils.utils import dotdict

from .compiled import assert_same_step, no_host_reads, recording, same_random_numbers

B = 16


def optimizer(cfg_optimizer, params):
    return hydra.utils.instantiate(cfg_optimizer, params=params, _convert_="all")


def small_agent(algo, overrides=(), accelerator="cpu", precision="32-true"):
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"exp={algo}",
                "algo.hidden_size=8",
                f"algo.per_rank_batch_size={B}",
                f"fabric.precision={precision}",
                *overrides,
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    fabric = Fabric(accelerator=accelerator, devices=1, precision=precision)
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)})
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    build_agent = sac_build_agent if algo == "sac" else droq_build_agent
    torch.manual_seed(0)
    agent, _ = build_agent(fabric, cfg, obs_space, action_space)
    optimizers = fabric.setup_optimizers(
        *(
            optimizer(c.optimizer, params)
            for c, params in (
                (cfg.algo.actor, agent.actor.parameters()),
                (cfg.algo.critic, agent.qfs.parameters()),
                (cfg.algo.alpha, [agent.log_alpha]),
            )
        )
    )
    g = torch.Generator().manual_seed(1)
    data = {
        "observations": torch.randn(B, 5, generator=g),
        "next_observations": torch.randn(B, 5, generator=g),
        "actions": torch.rand(B, 2, generator=g) * 2 - 1,
        "rewards": torch.randn(B, 1, generator=g),
        "terminated": (torch.rand(B, 1, generator=g) < 0.3).float(),
    }
    return cfg, fabric, agent, optimizers, {k: v.to(fabric.device) for k, v in data.items()}


@pytest.mark.parametrize("algo", ["sac", "droq"])
def test_the_losses_compile_into_single_graphs_without_synchronizations(monkeypatch, algo):
    # The temperature was read with `.item()`: a synchronization with the device at every loss, also inside the
    # compiled ones (23% slower with CUDA graphs)
    _, _, agent, _, data = small_agent(algo)
    no_host_reads(monkeypatch)
    # The distributions without the validation of their arguments, as in training (the CLI turns it off): the default
    # of PyTorch validates them, with checks that `torch.compile` doesn't trace
    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    obs, actions, rewards = data["observations"], data["actions"], data["rewards"]
    next_obs, terminated = data["next_observations"], data["terminated"]
    graph = lambda fn: torch.compile(fn, backend="eager", fullgraph=True)  # noqa: E731
    if algo == "sac":
        graph(sac.critic_loss_fn)(agent, obs, actions, rewards, next_obs, terminated, 0.99)
        graph(sac.actor_loss_fn)(agent, obs)
    else:
        targets = graph(droq.next_target_fn)(agent, next_obs, rewards, terminated, 0.99)
        graph(droq.critic_loss_fn)(agent, obs, actions, targets, 0)
        graph(droq.actor_loss_fn)(agent, obs)


@pytest.mark.parametrize("algo", ["sac", "droq"])
def test_the_target_critics_move_as_with_the_loop_over_their_weights(algo):
    # All the weights at once, with the roundings of `tau * critic + (1 - tau) * target`
    _, _, agent, _, _ = small_agent(algo)
    with torch.no_grad():
        for p in agent.qfs_unwrapped.parameters():
            p.add_(torch.randn_like(p))
    expected = [
        agent._tau * p + (1 - agent._tau) * t
        for p, t in zip(agent.qfs_unwrapped.parameters(), agent.qfs_target.parameters())
    ]
    if algo == "sac":
        agent.qfs_target_ema()
    else:
        for i in range(agent.num_critics):
            agent.qfs_target_ema(critic_idx=i)
    for e, t in zip(expected, agent.qfs_target.parameters()):
        assert torch.equal(t, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("algo", ["sac", "droq"])
def test_the_compiled_losses_are_the_ones_of_the_eager_losses(monkeypatch, algo):
    # The same weights, the same batch and the same random numbers (of the actions and of the dropout of DroQ): the same
    # losses and gradients of a gradient step (two of the critics for DroQ), with and without `torch.compile` (and its
    # CUDA graphs)
    same_random_numbers(monkeypatch)
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    module = sac if algo == "sac" else droq
    results = []
    for enabled in (False, True):
        cfg, fabric, agent, optimizers, data = small_agent(
            algo, [f"algo.compile.enabled={enabled}"], accelerator="cuda"
        )
        with monkeypatch.context() as patch:
            aggregator, losses, grads = recording(module, patch)
            torch.manual_seed(1)
            if algo == "sac":
                for name, value in sac.train(fabric, agent, *optimizers, copy.copy(data), 0, cfg, 1).items():
                    aggregator.update(name, value)
            else:

                class Store:
                    # Every batch is `data` (`ReplayStore.sample`); the actor's sampler shares the generator
                    sampler = SimpleNamespace(rng=None, queue=None)

                    def sample(self, batch_size, n_samples=1, **kwargs):
                        return {k: v[None].repeat(n_samples, *([1] * v.dim())) for k, v in data.items()}

                    def batches(self, n_steps, batch_size, max_sampled=None):
                        for _ in range(n_steps):
                            yield {k: v.float() for k, v in data.items()}

                # Two gradient steps of DroQ: two updates of the critics, then one of the actor
                droq_algo = droq.DroQ(fabric, cfg)
                actor_optimizer, qf_optimizer, alpha_optimizer = optimizers
                state = sac.SACState(agent, qf_optimizer, actor_optimizer, alpha_optimizer)
                for step, batch in enumerate(droq_algo.batches(state, Store(), 2, 1)):
                    for name, value in droq_algo.train_step(state, batch, step).items():
                        aggregator.update(name, value)
        results.append((losses, grads))
    assert_same_step(*results)


def small_sac_ae(overrides=(), accelerator="cpu"):
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=sac_ae",
                "env.screen_size=64",
                "algo.cnn_keys.encoder=[rgb]",
                "algo.mlp_keys.encoder=[]",
                "algo.mlp_keys.decoder=[]",
                "algo.hidden_size=8",
                "algo.cnn_channels_multiplier=1",
                "algo.encoder.features_dim=8",
                f"algo.per_rank_batch_size={B}",
                "algo.actor.per_rank_update_freq=1",
                *overrides,
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    fabric = Fabric(accelerator=accelerator, devices=1)
    obs_space = gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, (9, 64, 64), np.uint8)})
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    torch.manual_seed(0)
    agent, encoder, decoder, _ = sac_ae_build_agent(fabric, cfg, obs_space, action_space)
    optimizers = fabric.setup_optimizers(
        *(
            optimizer(cfg_optimizer, params)
            for cfg_optimizer, params in (
                (cfg.algo.actor.optimizer, agent.actor.parameters()),
                (cfg.algo.critic.optimizer, agent.critic.parameters()),
                (cfg.algo.alpha.optimizer, [agent.log_alpha]),
                (cfg.algo.encoder.optimizer, encoder.parameters()),
                (cfg.algo.decoder.optimizer, decoder.parameters()),
            )
        )
    )
    g = torch.Generator().manual_seed(1)
    data = {
        "rgb": torch.randint(0, 256, (B, 9, 64, 64), generator=g).float(),
        "next_rgb": torch.randint(0, 256, (B, 9, 64, 64), generator=g).float(),
        "actions": torch.rand(B, 2, generator=g) * 2 - 1,
        "rewards": torch.randn(B, 1, generator=g),
        "terminated": (torch.rand(B, 1, generator=g) < 0.3).float(),
    }
    return cfg, fabric, (agent, encoder, decoder), optimizers, {k: v.to(fabric.device) for k, v in data.items()}


def test_the_losses_of_sac_ae_compile_into_single_graphs_without_synchronizations(monkeypatch):
    _, _, (agent, encoder, decoder), _, data = small_sac_ae()
    no_host_reads(monkeypatch)
    # Without the validation of the arguments of the distributions, as in training
    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    obs, next_obs = {"rgb": data["rgb"] / 255}, {"rgb": data["next_rgb"] / 255}
    graph = lambda fn: torch.compile(fn, backend="eager", fullgraph=True)  # noqa: E731
    graph(sac_ae.critic_loss_fn)(agent, obs, next_obs, data["actions"], data["rewards"], data["terminated"], 0.99)
    graph(sac_ae.actor_loss_fn)(agent, obs)
    graph(sac_ae.reconstruction_loss_fn)(encoder, decoder, obs, data, ("rgb",), (), 1e-6)


def test_the_target_critic_of_sac_ae_moves_as_with_the_loop_over_its_weights():
    _, _, (agent, _, _), _, _ = small_sac_ae()
    critic, target = agent.critic_unwrapped, agent.critic_target
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(torch.randn_like(p))
    expected_qfs = [
        agent._tau * p + (1 - agent._tau) * t for p, t in zip(critic.qfs.parameters(), target.qfs.parameters())
    ]
    expected_encoder = [
        agent._encoder_tau * p + (1 - agent._encoder_tau) * t
        for p, t in zip(critic.encoder.parameters(), target.encoder.parameters())
    ]
    agent.critic_target_ema()
    agent.critic_encoder_target_ema()
    for e, t in zip([*expected_qfs, *expected_encoder], [*target.qfs.parameters(), *target.encoder.parameters()]):
        assert torch.equal(t, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_the_compiled_losses_of_sac_ae_are_the_ones_of_the_eager_losses(monkeypatch):
    # The same weights, the same batch and the same random numbers (of the actions and of the dequantization of the
    # targets of the decoder): the same losses and gradients of a gradient step, with and without `torch.compile` (and
    # its CUDA graphs)
    same_random_numbers(monkeypatch)
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    results = []
    for enabled in (False, True):
        cfg, fabric, models, optimizers, data = small_sac_ae([f"algo.compile.enabled={enabled}"], accelerator="cuda")
        with monkeypatch.context() as patch:
            aggregator, losses, grads = recording(sac_ae, patch)
            torch.manual_seed(1)
            for name, value in sac_ae.train(fabric, *models, *optimizers, copy.copy(data), 0, cfg).items():
                aggregator.update(name, value)
        results.append((losses, grads))
    assert_same_step(*results)
