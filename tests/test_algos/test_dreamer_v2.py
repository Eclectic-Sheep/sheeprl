"""DreamerV2 (and P2E-DV2): the KL loss, the decoder, the replay buffer, the objective of the actor, the RSSM, the
initialization and the weight decay."""

import math
import os
import shutil
import sys
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf
from torch import nn
from torch.distributions import Independent, Normal

from sheeprl import ROOT_DIR
from sheeprl.algos.dreamer_v2.agent import CNNDecoder, build_agent
from sheeprl.algos.dreamer_v2.loss import reconstruction_loss
from sheeprl.algos.dreamer_v2.utils import (
    actor_objective,
    build_buffer,
    build_optimizer,
    env_buffer_size,
    sample_batches,
)
from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer
from sheeprl.utils.utils import dotdict


def kl_loss_of(posteriors_logits, priors_logits, kl_free_nats, kl_free_avg):
    T, B = posteriors_logits.shape[:2]
    po = {"state": Independent(Normal(torch.zeros(T, B, 2), 1), 1)}
    pr = Independent(Normal(torch.zeros(T, B, 1), 1), 1)
    _, kl, kl_loss, *_ = reconstruction_loss(
        po,
        {"state": torch.zeros(T, B, 2)},
        pr,
        torch.zeros(T, B, 1),
        priors_logits,
        posteriors_logits,
        kl_balancing_alpha=0.8,
        kl_free_nats=kl_free_nats,
        kl_free_avg=kl_free_avg,
    )
    return kl, kl_loss


def test_the_free_nats_bound_the_kl_of_every_step_without_free_avg():
    # With `kl_free_avg=False` the loss crashed (`torch.maximum` of a tensor and a float)
    torch.manual_seed(0)
    T, B, S, D = 4, 3, 2, 5
    posteriors_logits = torch.randn(T, B, S, D)
    # Half of the steps have the posterior as prior: KL 0, under the free nats
    priors_logits = torch.where(torch.rand(T, B, 1, 1) < 0.5, posteriors_logits, torch.randn(T, B, S, D))
    kl, kl_loss = kl_loss_of(posteriors_logits, priors_logits, kl_free_nats=1.0, kl_free_avg=False)
    assert kl.shape == (T, B) and (kl < 1.0).any() and (kl > 1.0).any()
    # The two terms of the KL balancing have the same value: the loss is the mean of the bounded KLs
    torch.testing.assert_close(kl_loss, kl.clamp(min=1.0).mean())
    # With `kl_free_avg=True` the bound is on the mean
    _, kl_loss_avg = kl_loss_of(posteriors_logits, priors_logits, kl_free_nats=1.0, kl_free_avg=True)
    torch.testing.assert_close(kl_loss_avg, kl.mean().clamp(min=1.0))


@pytest.mark.parametrize("buffer_type", ["sequential", "episode"])
def test_the_buffer_holds_buffer_size_steps_of_the_process(buffer_type, tmp_path):
    # The episode buffer, shared by the environments of the process, held `buffer.size` divided by their number
    cfg = dotdict(
        {
            "dry_run": False,
            "seed": 0,
            "buffer": {"size": 1000, "type": buffer_type, "memmap": False, "prioritize_ends": False},
            "env": {"num_envs": 4},
            "algo": {"per_rank_sequence_length": 5, "cnn_keys": {"encoder": []}, "mlp_keys": {"encoder": ["state"]}},
        }
    )
    buffer = build_buffer(SimpleNamespace(world_size=2, global_rank=0), cfg, str(tmp_path), dry_run_size=2)
    if buffer_type == "episode":
        assert isinstance(buffer, EpisodeBuffer) and buffer.buffer_size == 500
    else:
        assert isinstance(buffer, EnvIndependentReplayBuffer) and buffer.buffer_size == 125


@pytest.mark.parametrize("output_channels", [[1], [3], [3, 3]])
def test_the_cnn_decoder_normalizes_its_three_hidden_layers(output_channels):
    # The LayerNorms were one per output channel: only 3 channels (one RGB image) built a decoder
    decoder = CNNDecoder(
        keys=[f"rgb{i}" for i in range(len(output_channels))],
        output_channels=output_channels,
        channels_multiplier=2,
        latent_state_size=10,
        cnn_encoder_output_dim=16,
        image_size=(64, 64),
        layer_norm=True,
    )
    norms = [m for m in decoder.modules() if isinstance(m, nn.LayerNorm)]
    assert [n.normalized_shape for n in norms] == [(8,), (4,), (2,)]
    reconstructed = decoder(torch.randn(2, 3, 10))
    assert [reconstructed[f"rgb{i}"].shape for i in range(len(output_channels))] == [
        (2, 3, c, 64, 64) for c in output_channels
    ]


@pytest.mark.parametrize(
    "objective_mix,is_continuous,expected",
    [(None, True, "dynamics"), (None, False, "reinforce"), (0.0, False, "dynamics"), (1.0, True, "reinforce")],
)
def test_the_actor_objective_is_the_dynamics_or_reinforce(objective_mix, is_continuous, expected):
    # By default (`null`) DreamerV2 learns the continuous actions by backpropagating the lambda-values through the
    # dynamics and the discrete ones with REINFORCE (`actor_grad: auto`); the default was REINFORCE for both
    dynamics, reinforce = torch.tensor(1.0), torch.tensor(2.0)
    objective = actor_objective(objective_mix, is_continuous, dynamics, lambda: reinforce)
    assert objective is (dynamics if expected == "dynamics" else reinforce)
    torch.testing.assert_close(actor_objective(0.25, is_continuous, dynamics, lambda: reinforce), torch.tensor(1.25))


def test_the_default_objective_depends_on_the_actions():
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["exp=dreamer_v2"])
    assert cfg.algo.actor.objective_mix is None


DREAMER_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "dry_run=True",
    "env=dummy",
    "env.id=continuous_dummy",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=0",
    "checkpoint.save_last=True",
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


@pytest.mark.parametrize(
    "module,args,expected",
    [
        # DreamerV2 on continuous actions, with the default objective
        ("sheeprl.algos.dreamer_v2.dreamer_v2", ["exp=dreamer_v2"], [(None, True)] * 2),
        # The exploration of Plan2Explore reads `algo.actor.objective_mix` (it was hard-coded to the default), for its
        # exploration actor and for its task actor
        (
            "sheeprl.algos.p2e_dv2.p2e_dv2_exploration",
            ["exp=p2e_dv2_exploration", "algo.actor.objective_mix=0.5"],
            [(0.5, True), (0.5, True)] * 2,
        ),
    ],
)
def test_the_actors_are_trained_with_the_configured_objective(module, args, expected):
    import importlib

    from sheeprl.cli import run

    algo_module = importlib.import_module(module)
    objectives = []

    def recording_objective(objective_mix, is_continuous, dynamics, reinforce):
        objectives.append((objective_mix, is_continuous))
        return actor_objective(objective_mix, is_continuous, dynamics, reinforce)

    root_dir = "pytest_dreamer_v2_objective"
    argv = [os.path.join(ROOT_DIR, "__main__.py"), *DREAMER_ARGS, *args, f"root_dir={root_dir}"]
    try:
        with (
            mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(algo_module, "actor_objective", recording_objective),
        ):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert objectives == expected


@pytest.mark.parametrize(
    "exp,module",
    [
        ("dreamer_v2", "sheeprl.algos.dreamer_v2.dreamer_v2"),
        ("p2e_dv2_exploration", "sheeprl.algos.p2e_dv2.p2e_dv2_exploration"),
        ("dreamer_v3", "sheeprl.algos.dreamer_v3.dreamer_v3"),
        ("p2e_dv3_exploration", "sheeprl.algos.p2e_dv3.p2e_dv3_exploration"),
    ],
)
def test_the_entropy_of_the_tanh_normal_actors_is_estimated(exp, module):
    # The tanh-normal distribution has no analytic entropy: the fallback of the actor losses had the wrong shape. The
    # losses of DreamerV2 crashed, the ones of DreamerV3 broadcast it: their entropy was always 0
    import importlib

    from sheeprl.cli import run
    from sheeprl.utils.distribution import entropy

    algo_module = importlib.import_module(module)
    entropies = []

    def recording_entropy(dist):
        entropies.append(entropy(dist))
        return entropies[-1]

    root_dir = f"pytest_{exp}_tanh_normal"
    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *DREAMER_ARGS,
        f"exp={exp}",
        "distribution.type=tanh_normal",
        "algo.actor.ent_coef=1e-4",
        f"root_dir={root_dir}",
    ]
    if "v3" in exp:
        # A dry run of DreamerV3 plays one step before training
        argv.append("algo.per_rank_sequence_length=1")
    try:
        with (
            mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(algo_module, "policy_entropy", recording_entropy),
            # The task actor of P2E-DV3 learns as the actor of DreamerV3
            mock.patch("sheeprl.algos.dreamer_v3.dreamer_v3.policy_entropy", recording_entropy),
        ):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    # Every actor loss (two for Plan2Explore, at every gradient step) estimates the entropy of its policies
    assert len(entropies) == (4 if "p2e" in exp else 2)
    assert all(torch.isfinite(e).all() for e in entropies)


def test_the_batches_of_an_iteration_are_sampled_16_at_a_time():
    # They were sampled, and moved to the device, all at once: the 100 gradient steps of the first training of
    # DreamerV2 (`algo.per_rank_pretrain_steps`) would take 100 batches on the device
    calls = []

    class Buffer:
        def sample_tensors(self, batch_size, sequence_length, n_samples, **kwargs):
            calls.append(n_samples)
            return {"rewards": torch.arange(n_samples).view(-1, 1, 1).expand(n_samples, sequence_length, batch_size)}

    cfg = dotdict({"algo": {"per_rank_batch_size": 3, "per_rank_sequence_length": 2}, "buffer": {"from_numpy": False}})
    batches = list(sample_batches(SimpleNamespace(device="cpu"), cfg, Buffer(), 40))
    assert calls == [16, 16, 8]
    assert len(batches) == 40
    assert all(batch["rewards"].shape == (2, 3) and batch["rewards"].dtype == torch.float32 for batch in batches)


@pytest.mark.parametrize("module,exp", [("dreamer_v2", "dreamer_v2"), ("dreamer_v1", "dreamer_v1")])
def test_the_first_training_pretrains(module, exp):
    # The ratio took the pretraining steps as the policy steps of the first training, capped to the ones of the
    # iteration: with the replay ratio of DreamerV2 (0.2) the first training did no gradient step instead of 100
    import importlib

    from sheeprl.cli import run

    algo_module = importlib.import_module(f"sheeprl.algos.{module}.{module}")
    steps = []

    def counting_train(*args, **kwargs):
        steps.append(1)
        return module_train(*args, **kwargs)

    module_train = algo_module.train
    root_dir = f"pytest_{exp}_pretrain"
    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *DREAMER_ARGS,
        f"exp={exp}",
        "env.id=discrete_dummy",
        "dry_run=False",
        # 4 iterations of 2 environments, random actions in the first 2, training from the second one
        "algo.total_steps=8",
        "algo.learning_starts=4",
        "algo.replay_ratio=0.25",
        "algo.per_rank_pretrain_steps=5",
        f"root_dir={root_dir}",
    ]
    try:
        with (
            mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(algo_module, "train", counting_train),
        ):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    # The replay ratio gives 0, 1, 0 gradient steps (2 policy steps per iteration); the first training also pretrains
    assert len(steps) == 5 + 1


def test_the_buffer_of_every_environment_holds_a_sequence():
    # A buffer smaller than a sequence crashed at the first training, after the random actions; a dry run had a buffer
    # of 2 steps, whatever the length of the sequences
    cfg = dotdict(
        {"dry_run": False, "buffer": {"size": 40}, "env": {"num_envs": 4}, "algo": {"per_rank_sequence_length": 5}}
    )
    fabric = SimpleNamespace(world_size=2)
    assert env_buffer_size(fabric, cfg, dry_run_size=2) == 5
    cfg.buffer.size = 39
    with pytest.raises(ValueError, match="increase `buffer.size`"):
        env_buffer_size(fabric, cfg, dry_run_size=2)
    cfg.dry_run = True
    assert env_buffer_size(fabric, cfg, dry_run_size=2) == 5
    assert env_buffer_size(fabric, cfg, dry_run_size=8) == 8


def small_dreamer_v2(overrides=()):
    """A small DreamerV2 on images and vectors."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v2",
                "env=dummy",
                "algo.cnn_keys.encoder=[rgb]",
                "algo.mlp_keys.encoder=[state]",
                "algo.dense_units=8",
                "algo.mlp_layers=2",
                "algo.world_model.encoder.cnn_channels_multiplier=2",
                "algo.world_model.recurrent_model.recurrent_state_size=16",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
                "algo.world_model.stochastic_size=4",
                "algo.world_model.discrete_size=5",
                *overrides,
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict(
        {
            "rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
            "state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32),
        }
    )
    world_model, actor, critic, target_critic, _ = build_agent(
        Fabric(accelerator="cpu", devices=1), (3,), False, cfg, obs_space
    )
    return cfg, (world_model, actor, critic, target_critic)


def test_the_rssm_is_the_one_of_the_official_implementation():
    # The layer before the GRU had the units of the other layers and a LayerNorm, and the LayerNorms the epsilon of
    # PyTorch: in the official RSSM it has the units of the hidden layers of the prior and of the posterior and the
    # normalization of the configuration (none by default), and the LayerNorms the epsilon of Keras
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["exp=dreamer_v2"])
    world_model_cfg = cfg.algo.world_model
    assert world_model_cfg.recurrent_model.dense_units == world_model_cfg.transition_model.hidden_size == 600
    assert world_model_cfg.recurrent_model.layer_norm is False
    _, (world_model, *_) = small_dreamer_v2()
    recurrent = world_model.rssm.recurrent_model.module
    assert not any(isinstance(m, nn.LayerNorm) for m in recurrent.mlp.modules())
    assert {m.eps for m in recurrent.modules() if isinstance(m, nn.LayerNorm)} == {1e-3}
    _, (world_model, *_) = small_dreamer_v2(["algo.layer_norm=True"])
    assert {m.eps for m in world_model.modules() if isinstance(m, nn.LayerNorm)} == {1e-3}


def test_the_weights_are_initialized_as_the_layers_of_keras():
    # Keras initializes the kernels with the uniform Glorot initializer: they were drawn from a normal distribution
    _, models = small_dreamer_v2(["algo.dense_units=512", "algo.mlp_layers=1"])
    layer = models[2].module.model[0]
    weight = layer.weight.detach()
    limit = math.sqrt(6 / (layer.in_features + layer.out_features))
    assert weight.abs().max().item() <= limit
    assert (weight.abs() > 0.9 * limit).float().mean().item() == pytest.approx(0.1, abs=0.02)
    assert torch.all(layer.bias == 0)


def test_the_weight_decay_multiplies_the_weights_before_every_step():
    # The weights are multiplied by `1 - weight_decay` before every step, as in the official implementation: the
    # weight decay of Adam was added to the gradients
    weight = nn.Parameter(torch.ones(3))
    optimizer = build_optimizer(
        {"_target_": "torch.optim.Adam", "lr": 1e-3, "eps": 1e-5, "weight_decay": 0.1}, [weight]
    )
    weight.grad = torch.zeros(3)
    optimizer.step()
    torch.testing.assert_close(weight.detach(), torch.full((3,), 0.9))
    # Without weight decay, the optimizer of the configuration
    assert type(build_optimizer({"_target_": "torch.optim.Adam", "lr": 1e-3}, [weight])) is torch.optim.Adam
