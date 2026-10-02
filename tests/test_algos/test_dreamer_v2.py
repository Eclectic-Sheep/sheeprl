"""DreamerV2 (and P2E-DV2): the KL loss, the decoder, the replay buffer, the objective of the actor."""

import os
import shutil
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from torch.distributions import Independent, Normal

from sheeprl import ROOT_DIR
from sheeprl.algos.dreamer_v2.agent import CNNDecoder
from sheeprl.algos.dreamer_v2.loss import reconstruction_loss
from sheeprl.algos.dreamer_v2.utils import actor_objective, build_buffer
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
