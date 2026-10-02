"""DreamerV2 (and P2E-DV2): the KL loss, the replay buffer."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Independent, Normal

from sheeprl.algos.dreamer_v2.loss import reconstruction_loss
from sheeprl.algos.dreamer_v2.utils import build_buffer
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
