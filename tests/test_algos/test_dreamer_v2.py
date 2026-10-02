"""DreamerV2 (and P2E-DV2): the KL loss and the decoder."""

import pytest
import torch
from torch import nn
from torch.distributions import Independent, Normal

from sheeprl.algos.dreamer_v2.agent import CNNDecoder
from sheeprl.algos.dreamer_v2.loss import reconstruction_loss


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
