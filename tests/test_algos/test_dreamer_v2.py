"""DreamerV2 (and P2E-DV2): the KL loss, the decoder, the sampling of the batches and the objective of the actor."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf
from torch import nn
from torch.distributions import Independent, Normal

from sheeprl.algos.dreamer_v2.agent import CNNDecoder
from sheeprl.algos.dreamer_v2.dreamer_v2 import DreamerV2, behaviour_learning, env_buffer_size, sample_batches
from sheeprl.algos.dreamer_v2.loss import reconstruction_loss
from sheeprl.core import TrainSchedule
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


def test_the_batches_of_an_iteration_are_sampled_16_at_a_time():
    # They were sampled, and moved to the device, all at once: the 100 gradient steps of the first training of
    # DreamerV2 (`algo.per_rank_pretrain_steps`) took 100 batches on the device
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


def behaviour_on_continuous_actions(overrides):
    """The configuration of a small DreamerV2 on continuous actions and the output of its behaviour learning on random
    latent states (the models initialized with seed 0)."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v2",
                "env=dummy",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.mlp_keys.decoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
                "algo.horizon=4",
                "algo.per_rank_batch_size=2",
                "algo.per_rank_sequence_length=3",
                "metric.log_level=0",
            ]
            + overrides,
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    fabric = Fabric(accelerator="cpu", devices=1)
    algo = DreamerV2(fabric, cfg)
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    schedule = TrainSchedule(cfg, 1, algo.steps_per_iteration, off_policy=True)
    torch.manual_seed(0)
    state, _ = algo.build(obs_space, gym.spaces.Box(-1, 1, shape=(2,)), schedule, "unused")
    T, B = cfg.algo.per_rank_sequence_length, cfg.algo.per_rank_batch_size
    world_model_cfg = cfg.algo.world_model
    posteriors = torch.randn(T, B, world_model_cfg.stochastic_size, world_model_cfg.discrete_size)
    recurrent_states = torch.randn(T, B, world_model_cfg.recurrent_model.recurrent_state_size)
    out = behaviour_learning(
        fabric,
        cfg,
        state.world_model,
        state.actor,
        state.critic,
        state.target_critic,
        state.actor_optimizer,
        state.critic_optimizer,
        posteriors,
        recurrent_states,
        torch.zeros(T, B, 1),
        algo.is_continuous,
        algo.actions_dim,
        objective_mix=cfg.algo.actor.objective_mix,
    )
    return cfg, out


def test_the_actor_learns_continuous_actions_by_dynamics_backpropagation():
    # The default objective was REINFORCE for every action space: DreamerV2 backpropagates the lambda-values through the
    # dynamics for continuous actions (`actor_grad: auto`)
    cfg, out = behaviour_on_continuous_actions(["algo.actor.ent_coef=0"])
    assert cfg.algo.actor.objective_mix is None and not cfg.algo.world_model.use_continues
    # The objective is the lambda-values of the imagined states, discounted
    lambda_values = out["lambda_values"]
    discount = cfg.algo.gamma ** torch.arange(cfg.algo.horizon - 1).view(-1, 1, 1)
    torch.testing.assert_close(out["policy_loss"], -(discount * lambda_values[1:]).mean())


def test_the_entropy_of_the_tanh_normal_actor_is_in_its_loss():
    # The entropy of the tanh-normal distribution, which has no analytic one, was 0: `ent_coef` did nothing
    args = ["distribution.type=tanh_normal", "algo.actor.objective_mix=0"]
    _, without_entropy = behaviour_on_continuous_actions(args + ["algo.actor.ent_coef=0"])
    _, with_entropy = behaviour_on_continuous_actions(args + ["algo.actor.ent_coef=1"])
    torch.testing.assert_close(with_entropy["lambda_values"], without_entropy["lambda_values"])
    assert not torch.allclose(with_entropy["policy_loss"], without_entropy["policy_loss"])
