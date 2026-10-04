"""DreamerV1 (and P2E-DV1): the continue loss, the exploration noise, the player, the episode starts, the actor and
the initialization."""

import copy
import math
import os
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Dict
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import get_class, instantiate
from lightning import Fabric
from omegaconf import OmegaConf
from torch import nn
from torch.distributions import Bernoulli, Independent, Normal, TransformedDistribution

from sheeprl import ROOT_DIR
from sheeprl.algos.dreamer_v1 import agent, dreamer_v1
from sheeprl.algos.dreamer_v1.agent import RSSM, PlayerDV1, RecurrentModel, build_agent, gru_step
from sheeprl.algos.dreamer_v1.loss import reconstruction_loss, state_kl
from sheeprl.algos.dreamer_v1.utils import add_is_first
from sheeprl.algos.dreamer_v2.agent import Actor
from sheeprl.utils import compile as compile_utils
from sheeprl.utils.distribution import SafeTanhTransform
from sheeprl.utils.utils import dotdict

from .compiled import assert_same_step, no_host_reads, recording, same_random_numbers

DREAMER_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "dry_run=True",
    "env=dummy",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=0",
    "checkpoint.save_last=False",
    "algo.run_test=False",
    "algo.cnn_keys.encoder=[]",
    "algo.cnn_keys.decoder=[]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
    "algo.dense_units=8",
    "algo.world_model.recurrent_model.recurrent_state_size=8",
    "algo.world_model.representation_model.hidden_size=8",
    "algo.world_model.transition_model.hidden_size=8",
    "algo.horizon=4",
    "algo.per_rank_batch_size=1",
    "algo.per_rank_sequence_length=2",
    "algo.learning_starts=0",
    "algo.replay_ratio=1",
    "buffer.size=10",
]


def run_dreamer_v1(args, root_dir):
    from sheeprl.cli import run

    argv = [os.path.join(ROOT_DIR, "__main__.py"), *DREAMER_ARGS, *args, f"root_dir={root_dir}"]
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(sys, "argv", argv):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)


def test_the_continue_loss_is_the_mean_negative_log_likelihood():
    # It was the log-likelihood itself, one per step: the loss wasn't a scalar (the training crashed) and its sign
    # trained the continue model away from the targets
    torch.manual_seed(0)
    T, B = 4, 3
    # The targets are the discount, not 0 or 1: the training doesn't validate the arguments of the distributions
    qc = Independent(Bernoulli(logits=torch.randn(T, B, 1), validate_args=False), 1)
    targets = (torch.rand(T, B, 1) > 0.3).float() * 0.99
    qo = {"state": Independent(Normal(torch.zeros(T, B, 2), 1), 1)}
    qr = Independent(Normal(torch.zeros(T, B, 1), 1), 1)
    states = Independent(Normal(torch.zeros(T, B, 5), 1), 1)
    rec_loss, *_, continue_loss = reconstruction_loss(
        qo, {"state": torch.randn(T, B, 2)}, qr, torch.zeros(T, B, 1), states, states, 3.0, 1.0, qc, targets, 2.0
    )
    assert rec_loss.shape == continue_loss.shape == ()
    torch.testing.assert_close(continue_loss, -2.0 * qc.log_prob(targets).mean())
    assert continue_loss > 0


@pytest.mark.parametrize("exp", ["dreamer_v1", "p2e_dv1_exploration"])
def test_the_continue_model_is_trained(exp):
    run_dreamer_v1(
        [f"exp={exp}", "env.id=discrete_dummy", "algo.world_model.use_continues=True"], f"pytest_{exp}_continues"
    )


def test_the_exploration_noise_halves_every_decay_steps():
    # The amount was `0.5 ** step / decay`: `expl_min` from the first steps
    actor = SimpleNamespace(_expl_amount=0.4, _expl_decay=1000, _expl_min=0.05)
    for step, amount in ((0, 0.4), (1000, 0.2), (2000, 0.1), (4000, 0.05)):
        assert Actor._get_expl_amount(actor, step) == pytest.approx(amount)


def test_each_environment_explores_on_its_own():
    # One random draw decided for all the environments whether they played a random action
    torch.manual_seed(0)
    num_envs = 1000
    actions = torch.nn.functional.one_hot(torch.zeros(1, num_envs, dtype=torch.long), 4).float()
    actor = SimpleNamespace(is_continuous=False, _get_expl_amount=lambda step: 0.5)
    (explored,) = Actor.add_exploration_noise(actor, [actions])
    assert explored.shape == actions.shape
    changed = (explored != actions).any(-1).sum().item()
    # About half of the envs explore, and 3/4 of the random actions differ from the played one
    assert 300 < changed < 450, changed


def test_the_player_samples_the_posterior_with_the_minimum_std_of_the_world_model(monkeypatch):
    # The player used the default minimum std (0.1) whatever `algo.world_model.min_std`
    min_stds = []

    def compute_stochastic_state(state_information, event_shape=1, min_std=0.1):
        min_stds.append(min_std)
        return (None, None), torch.zeros(*state_information.shape[:-1], 4)

    monkeypatch.setattr(agent, "compute_stochastic_state", compute_stochastic_state)
    player = PlayerDV1(
        encoder=lambda obs: torch.zeros(1, 2, 5),
        recurrent_model=RecurrentModel(4 + 3, 8),
        representation_model=nn.Linear(8 + 5, 8),
        actor=lambda latent_state, greedy, mask: ([torch.zeros(1, 2, 3)], None),
        actions_dim=[3],
        num_envs=2,
        stochastic_size=4,
        recurrent_state_size=8,
        device="cpu",
        min_std=0.5,
    )
    player.init_states()
    player.get_actions({})
    assert min_stds == [0.5]


@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"))],
)
def test_the_player_follows_the_updates_of_the_agent(accelerator):
    # The player was a copy of the agent with the weights tied: on CUDA its GRU moved them into a new buffer at its
    # first forward (`flatten_parameters`), and the player played with the initial recurrent model for the whole
    # training. It also takes the minimum std of the world model
    from hydra import compose, initialize_config_module
    from lightning import Fabric
    from omegaconf import OmegaConf

    from sheeprl.algos.dreamer_v1.agent import build_agent
    from sheeprl.utils.utils import dotdict

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v1",
                "env.num_envs=2",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.mlp_keys.decoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.min_std=0.3",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    fabric = Fabric(accelerator=accelerator, devices=1)
    world_model, actor, _, player = build_agent(fabric, [3], False, cfg, obs_space)
    assert player.min_std == 0.3
    player.init_states()
    with torch.no_grad():
        player.get_actions({"state": torch.randn(1, 2, 5, device=fabric.device)})
        for p in [*world_model.parameters(), *actor.parameters()]:
            p.add_(1.0)
    for agent_module, player_module in (
        (world_model.encoder, player.encoder),
        (world_model.rssm.recurrent_model, player.recurrent_model),
        (world_model.rssm.representation_model, player.representation_model),
        (actor, player.actor),
    ):
        agent_params, player_params = list(agent_module.parameters()), list(player_module.parameters())
        assert len(agent_params) == len(player_params) > 0
        for agent_p, player_p in zip(agent_params, player_params):
            assert torch.equal(agent_p, player_p)


def test_the_rssm_starts_the_episodes_from_the_zero_state():
    # The sequences cross the episodes: a step marked `is_first` starts from the zero state and the zero action, as the
    # player does at the start of an episode; the other steps go on from the previous ones
    torch.manual_seed(0)
    rssm = RSSM(RecurrentModel(4 + 2, 8), nn.Linear(8 + 5, 8), nn.Linear(8, 8), {}, min_std=0.1)
    posterior, recurrent_state, action, embedded_obs = (torch.randn(1, 3, n) for n in (4, 8, 2, 5))
    is_first = torch.tensor([1.0, 0.0, 1.0]).view(1, 3, 1)

    def dynamic(*args):
        torch.manual_seed(1)
        recurrent_state, posterior, _, posterior_mean_std, _ = rssm.dynamic(*args)
        return recurrent_state, posterior, posterior_mean_std[0]

    marked = dynamic(posterior, recurrent_state, action, embedded_obs, is_first)
    restarted = dynamic(
        torch.zeros_like(posterior), torch.zeros_like(recurrent_state), torch.zeros_like(action), embedded_obs, is_first
    )
    continued = dynamic(posterior, recurrent_state, action, embedded_obs, torch.zeros_like(is_first))
    first = is_first.view(-1).bool()
    for out, start, cont in zip(marked, restarted, continued):
        torch.testing.assert_close(out[:, first], start[:, first])
        torch.testing.assert_close(out[:, ~first], cont[:, ~first])
        assert not torch.allclose(out[:, first], cont[:, first])


@pytest.mark.parametrize("exp", ["dreamer_v1", "p2e_dv1_exploration"])
def test_the_rows_mark_the_first_observations_of_the_episodes(exp):
    # DreamerV1 stored no `is_first`: its sequences crossed the episodes with the recurrent state of the previous one
    from sheeprl.data.buffers import EnvIndependentReplayBuffer

    rows = []
    buffer_add = EnvIndependentReplayBuffer.add

    def recording_add(self, data, indices=None, *args, **kwargs):
        rows.append((copy.deepcopy({k: np.asarray(v) for k, v in data.items()}), indices))
        return buffer_add(self, data, indices, *args, **kwargs)

    with mock.patch.object(EnvIndependentReplayBuffer, "add", recording_add):
        # Episodes of 3 steps (truncated) in 7 iterations
        run_dreamer_v1(
            [
                f"exp={exp}",
                "env.id=discrete_dummy",
                "dry_run=False",
                "env.max_episode_steps=3",
                "algo.total_steps=14",
                "algo.replay_ratio=0",
            ],
            f"pytest_{exp}_is_first",
        )
    # The rows of every environment: the first one, then the one of every step, and after the step that ends an episode
    # the row with the first observation of the new one
    first_row, *other_rows = rows
    assert first_row[0]["is_first"].tolist() == [[[1.0], [1.0]]]
    step_rows = [row for row, indices in other_rows if indices is None]
    reset_rows = [(row, indices) for row, indices in other_rows if indices is not None]
    assert len(step_rows) == 7 and all(not row["is_first"].any() for row in step_rows)
    assert len(reset_rows) == 2 and all(row["is_first"].all() and indices == [0, 1] for row, indices in reset_rows)


@pytest.mark.parametrize("memmap", [False, True])
def test_a_buffer_saved_without_is_first_is_completed(memmap, tmp_path):
    # The buffers of the runs before `is_first` was stored: resumed (or loaded by a finetuning), adding a row with it
    # failed
    from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer

    rb = EnvIndependentReplayBuffer(
        6, n_envs=2, memmap=memmap, memmap_dir=tmp_path, buffer_cls=SequentialReplayBuffer, obs_keys=["state"]
    )
    # Environment 0 ends an episode at its second row, environment 1 at its third (truncated)
    terminated = np.array([[0, 0], [1, 0], [0, 0], [0, 0]], dtype=np.float32).reshape(4, 1, 2, 1)
    truncated = np.array([[0, 0], [0, 0], [0, 1], [0, 0]], dtype=np.float32).reshape(4, 1, 2, 1)
    for t in range(4):
        rb.add(
            {
                "state": np.zeros((1, 2, 3), np.float32),
                "terminated": terminated[t],
                "truncated": truncated[t],
                "actions": np.zeros((1, 2, 1), np.float32),
                "rewards": np.zeros((1, 2, 1), np.float32),
            }
        )
    add_is_first(rb)
    is_first = [np.asarray(buffer["is_first"])[:4, 0, 0].tolist() for buffer in rb.buffer]
    assert is_first == [[1, 0, 1, 0], [1, 0, 0, 1]]
    # Rows with it can be added
    rb.add(
        {
            "state": np.zeros((1, 2, 3), np.float32),
            "terminated": np.zeros((1, 2, 1), np.float32),
            "truncated": np.zeros((1, 2, 1), np.float32),
            "actions": np.zeros((1, 2, 1), np.float32),
            "rewards": np.zeros((1, 2, 1), np.float32),
            "is_first": np.ones((1, 2, 1), np.float32),
        }
    )


def dreamer_v1_cfg(exp: str, overrides=()) -> Dict[str, Any]:
    """The configuration of a small DreamerV1 (or P2E-DV1) on images."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"exp={exp}",
                "env=dummy",
                "algo.cnn_keys.encoder=[rgb]",
                "algo.mlp_keys.encoder=[]",
                "algo.dense_units=8",
                "algo.world_model.encoder.cnn_channels_multiplier=2",
                "algo.world_model.recurrent_model.recurrent_state_size=16",
                "algo.world_model.representation_model.hidden_size=16",
                "algo.world_model.transition_model.hidden_size=16",
                "algo.world_model.stochastic_size=4",
                *overrides,
            ],
        )
    return dotdict(OmegaConf.to_container(cfg, resolve=True))


IMAGES = gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8)})


@pytest.mark.parametrize("exp", ["dreamer_v1", "p2e_dv1_exploration"])
def test_the_continuous_actions_come_from_a_tanh_normal_of_the_initial_std(exp):
    # They came from a truncated normal, whose std couldn't go below 0.1: in the official implementation from a
    # tanh-transformed normal whose std is `init_std` when the network outputs zero, at least 1e-4. The tanh is the one
    # of its `TanhBijector`, whose inverse clips to the largest float32 below 1
    cfg = dreamer_v1_cfg(exp)
    actor_cfg = cfg.algo.actor
    assert (actor_cfg.init_std, actor_cfg.min_std, cfg.distribution.type) == (5.0, 1e-4, "auto")
    actor = get_class(actor_cfg.cls)(
        latent_state_size=4,
        actions_dim=[2],
        is_continuous=True,
        distribution_cfg=cfg.distribution,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        dense_units=8,
        mlp_layers=1,
    )
    nn.init.zeros_(actor.mlp_heads[0].weight)
    nn.init.zeros_(actor.mlp_heads[0].bias)
    _, (dist,) = actor(torch.randn(3, 4))
    assert isinstance(dist.base_dist, TransformedDistribution)
    assert [type(t) for t in dist.base_dist.transforms] == [SafeTanhTransform]
    torch.testing.assert_close(dist.base_dist.base_dist.loc, torch.zeros(3, 2))
    torch.testing.assert_close(dist.base_dist.base_dist.scale, torch.full((3, 2), 5.0 + 1e-4))


def test_the_gru_is_initialized_as_the_one_of_keras():
    # It had the initialization of PyTorch (uniform weights and biases): Keras initializes the weights of the inputs
    # with the uniform Glorot initializer, the ones of the recurrent state with an orthogonal matrix and the biases to
    # zero
    torch.manual_seed(0)
    # The GRU takes the output of a dense layer of its size
    gru = RecurrentModel(64, 128).rnn
    limit = math.sqrt(6 / (128 + 3 * 128))
    assert gru.weight_ih_l0.abs().max().item() <= limit
    assert (gru.weight_ih_l0.abs() > 0.9 * limit).float().mean().item() == pytest.approx(0.1, abs=0.02)
    torch.testing.assert_close(gru.weight_hh_l0.T @ gru.weight_hh_l0, torch.eye(128), atol=1e-5, rtol=0)
    assert torch.all(gru.bias_ih_l0 == 0) and torch.all(gru.bias_hh_l0 == 0)


@pytest.mark.parametrize("exp", ["dreamer_v1", "p2e_dv1_exploration"])
def test_the_weights_are_initialized_as_the_layers_of_keras(exp):
    # Keras initializes the kernels with the uniform Glorot initializer and the biases to zero: the kernels had the
    # uniform Kaiming initialization, with larger weights
    from sheeprl.algos.p2e_dv1.agent import build_agent as p2e_dv1_build_agent

    torch.manual_seed(0)
    cfg = dreamer_v1_cfg(exp, ["algo.dense_units=512"])
    fabric = Fabric(accelerator="cpu", devices=1)
    if exp == "dreamer_v1":
        world_model, actor, critic, _ = build_agent(fabric, [3], False, cfg, IMAGES)
        actors_and_critics = [actor, critic]
    else:
        world_model, _, *actors_and_critics, _ = p2e_dv1_build_agent(fabric, [3], False, cfg, IMAGES)
    layers = [world_model.reward_model.module.model[2], world_model.encoder.module.cnn_encoder.model[0].model[2]]
    for model in actors_and_critics:
        mlp = model.module.model
        layers.append((mlp if isinstance(mlp, nn.Sequential) else mlp.model)[2])
    for layer in layers:
        weight = layer.weight.detach()
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        limit = math.sqrt(6 / (fan_in + fan_out))
        assert weight.abs().max().item() <= limit
        assert (weight.abs() > 0.9 * limit).float().mean().item() == pytest.approx(0.1, abs=0.02)
        assert torch.all(layer.bias == 0)


@pytest.mark.parametrize("use_continues", [False, True])
def test_the_imagination_starts_from_the_steps_that_are_not_terminal(monkeypatch, use_continues):
    # It started from every step: with the continues, the last step of the sequences could be terminal, and the
    # official implementation starts from the other ones (`Dreamer._imagine_ahead`)
    # The continue targets are the discount, not 0 or 1, as in the training
    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    T, B = 4, 2
    cfg = dreamer_v1_cfg(
        "dreamer_v1",
        [
            f"algo.world_model.use_continues={use_continues}",
            f"algo.per_rank_sequence_length={T}",
            f"algo.per_rank_batch_size={B}",
            "algo.horizon=2",
        ],
    )
    fabric = Fabric(accelerator="cpu", devices=1)
    world_model, actor, critic, _ = build_agent(fabric, [3], False, cfg, IMAGES)
    rssm = world_model.rssm
    posteriors, starts = [], []

    def dynamic(*args, **kwargs):
        outputs = type(rssm).dynamic(rssm, *args, **kwargs)
        posteriors.append(outputs[1].detach())
        return outputs

    def imagination(prior, *args, **kwargs):
        starts.append(prior.detach())
        return type(rssm).imagination(rssm, prior, *args, **kwargs)

    rssm.dynamic, rssm.imagination = dynamic, imagination
    generator = torch.Generator().manual_seed(0)
    batch = {
        "rgb": torch.randint(0, 256, (T, B, 3, 64, 64), generator=generator).float(),
        "actions": nn.functional.one_hot(torch.randint(0, 3, (T, B), generator=generator), 3).float(),
        "rewards": torch.randn(T, B, 1, generator=generator),
        "terminated": torch.zeros(T, B, 1),
        "truncated": torch.zeros(T, B, 1),
        "is_first": torch.zeros(T, B, 1),
    }
    batch["terminated"][-1, 0] = 1
    optimizers = [torch.optim.Adam(m.parameters()) for m in (world_model, actor, critic)]
    dreamer_v1.train(fabric, world_model, actor, critic, *optimizers, batch, cfg)
    steps = posteriors[:-1] if use_continues else posteriors
    torch.testing.assert_close(starts[0], torch.cat(steps).reshape(1, -1, posteriors[0].shape[-1]))


def compilable_dreamer_v1(continuous, overrides=(), accelerator="cpu"):
    """A small DreamerV1 with continues, its optimizers and a batch for a gradient step."""
    cfg = dreamer_v1_cfg(
        "dreamer_v1",
        [
            "algo.world_model.use_continues=True",
            "algo.per_rank_batch_size=2",
            "algo.per_rank_sequence_length=4",
            "algo.horizon=3",
            *overrides,
        ],
    )
    fabric = Fabric(accelerator=accelerator, devices=1)
    torch.manual_seed(0)
    world_model, actor, critic, _ = build_agent(fabric, [2] if continuous else [3], continuous, cfg, IMAGES)
    optimizers = fabric.setup_optimizers(
        *(
            instantiate(optimizer, params=model.parameters(), _convert_="all")
            for optimizer, model in (
                (cfg.algo.world_model.optimizer, world_model),
                (cfg.algo.actor.optimizer, actor),
                (cfg.algo.critic.optimizer, critic),
            )
        )
    )
    T, B = 4, 2
    g = torch.Generator().manual_seed(1)
    actions = (
        torch.rand(T, B, 2, generator=g) * 2 - 1
        if continuous
        else nn.functional.one_hot(torch.randint(0, 3, (T, B), generator=g), 3).float()
    )
    data = {
        "rgb": torch.randint(0, 256, (T, B, 3, 64, 64), generator=g).float(),
        "actions": actions,
        "rewards": torch.randn(T, B, 1, generator=g),
        "terminated": torch.zeros(T, B, 1),
        "truncated": torch.zeros(T, B, 1),
        "is_first": torch.zeros(T, B, 1),
    }
    data["terminated"][-1, 0] = 1
    return cfg, fabric, (world_model, actor, critic), optimizers, {k: v.to(fabric.device) for k, v in data.items()}


@pytest.mark.parametrize("continuous", [False, True])
def test_the_losses_compile_into_single_graphs_without_host_reads(monkeypatch, continuous):
    # The losses are compiled with `algo.compile.enabled`: `nn.GRU` and the dispatch of `kl_divergence` broke the
    # graph of the world model, at every step of the unroll for the GRU
    cfg, _, (world_model, actor, critic), _, data = compilable_dreamer_v1(continuous)
    world_model_cfg = cfg.algo.world_model
    no_host_reads(monkeypatch)
    graph = lambda fn: torch.compile(fn, backend="eager", fullgraph=True)  # noqa: E731
    _, posteriors, recurrent_states, _ = graph(dreamer_v1.world_model_loss)(
        world_model,
        data,
        cnn_keys=("rgb",),
        mlp_keys=(),
        stochastic_size=world_model_cfg.stochastic_size,
        recurrent_state_size=world_model_cfg.recurrent_model.recurrent_state_size,
        use_continues=True,
        gamma=cfg.algo.gamma,
        kl_free_nats=world_model_cfg.kl_free_nats,
        kl_regularizer=world_model_cfg.kl_regularizer,
        continue_scale_factor=world_model_cfg.continue_scale_factor,
    )
    trajectories, lambda_values, discount = graph(dreamer_v1.imagine)(
        world_model,
        actor,
        critic,
        posteriors.detach(),
        recurrent_states.detach(),
        horizon=cfg.algo.horizon,
        gamma=cfg.algo.gamma,
        lmbda=cfg.algo.lmbda,
        use_continues=True,
    )
    graph(dreamer_v1.value_loss_fn)(critic, trajectories, lambda_values, discount)


def test_the_compiled_gru_step_and_kl_are_the_ones_of_pytorch():
    torch.manual_seed(0)
    p = Independent(Normal(torch.randn(5, 3, 30), torch.rand(5, 3, 30) + 0.1), 1)
    q = Independent(Normal(torch.randn(5, 3, 30), torch.rand(5, 3, 30) + 0.1), 1)
    assert torch.equal(state_kl(p, q), torch.distributions.kl_divergence(p, q))
    rnn = nn.GRU(16, 16)
    x, h = torch.randn(1, 4, 16), torch.randn(1, 4, 16)
    step = gru_step(x, h, rnn.weight_ih_l0, rnn.bias_ih_l0, rnn.weight_hh_l0, rnn.bias_hh_l0)
    torch.testing.assert_close(step, rnn(x, h)[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("continuous", [False, True])
def test_the_compiled_losses_are_the_ones_of_the_eager_losses(monkeypatch, continuous):
    # The same weights, the same batch and the same random numbers: the same losses and gradients, with and without
    # `torch.compile` (and its CUDA graphs)
    same_random_numbers(monkeypatch)
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    results = []
    for enabled in (False, True):
        cfg, fabric, models, optimizers, data = compilable_dreamer_v1(
            continuous, [f"algo.compile.enabled={enabled}"], accelerator="cuda"
        )
        with monkeypatch.context() as patch:
            aggregator, losses, grads = recording(dreamer_v1, patch)
            torch.manual_seed(1)
            for name, value in dreamer_v1.train(fabric, *models, *optimizers, data, cfg).items():
                aggregator.update(name, value)
        results.append((losses, grads))
    assert_same_step(*results)
