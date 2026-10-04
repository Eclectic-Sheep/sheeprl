"""The distributions of the Dreamer actors: the truncated normal of DreamerV2, whose bounds are validated only with the
validation of the arguments, and the tanh of the tanh-normal ones, whose inverse is the one of the official
`TanhBijector`."""

import pytest
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from sheeprl.algos.dreamer_v2.agent import Actor as DV2Actor
from sheeprl.algos.dreamer_v3.agent import Actor as DV3Actor
from sheeprl.utils.distribution import SafeTanhTransform, TruncatedNormal


def test_the_truncated_normal_reads_no_tensor_on_the_host_without_validation(monkeypatch):
    # It checked its bounds on the host at every construction: a synchronization with the device at every step of the
    # imagination of DreamerV2, and a break of its compiled graph
    def host_read(self, *args, **kwargs):
        raise AssertionError("A tensor read on the host")

    loc, scale = torch.zeros(3, 2), torch.ones(3, 2)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", host_read)
        patch.setattr(torch.Tensor, "item", host_read)
        dist = TruncatedNormal(loc, scale, -1, 1, validate_args=False)
    assert dist.mean.shape == (3, 2)


def test_the_truncated_normal_validates_its_bounds_with_the_validation_of_the_arguments():
    with pytest.raises(ValueError, match="Incorrect truncation range"):
        TruncatedNormal(torch.zeros(2), torch.ones(2), 1, -1, validate_args=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_tanh_has_finite_log_probabilities_where_it_rounds_to_one(dtype):
    # The tanh of 9.5 rounds to +-1 (in bfloat16 already above ~3.47): the atanh of PyTorch gives an infinite pre-image
    # and a NaN log-probability; the inverse of the official `TanhBijector` clips to the largest float32 below 1
    base = Normal(torch.tensor([5.0, -5.0], dtype=dtype), torch.tensor([1.0, 1.0], dtype=dtype))
    saturated = torch.tanh(torch.tensor([9.5, -9.5], dtype=dtype))
    assert saturated.abs().eq(1).all()
    assert TransformedDistribution(base, TanhTransform()).log_prob(saturated).isnan().all()
    assert TransformedDistribution(base, SafeTanhTransform()).log_prob(saturated).isfinite().all()
    largest_below_one = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    expected = torch.stack((largest_below_one, -largest_below_one)).atanh()
    torch.testing.assert_close(SafeTanhTransform().inv(torch.tensor([1.0, -1.0])), expected, rtol=0, atol=0)


def test_the_tanh_is_the_one_of_pytorch_where_it_does_not_round_to_one():
    base = Normal(torch.zeros(5), 2 * torch.ones(5))
    actions = torch.tensor([-0.999, -0.5, 0.0, 0.5, 0.999])
    torch.testing.assert_close(
        TransformedDistribution(base, SafeTanhTransform()).log_prob(actions),
        TransformedDistribution(base, TanhTransform()).log_prob(actions),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("actor_cls", [DV2Actor, DV3Actor])
def test_the_tanh_normal_actors_have_finite_log_probabilities_at_the_bounds(actor_cls):
    # The log-probabilities of REINFORCE (`algo.actor.objective_mix`) and of the greedy actions
    actor = actor_cls(
        latent_state_size=4,
        actions_dim=[2],
        is_continuous=True,
        distribution_cfg={"type": "tanh_normal"},
        dense_units=8,
        mlp_layers=1,
    )
    dist = actor(torch.randn(3, 4))[1][0]
    assert dist.log_prob(torch.tensor([[1.0, -1.0]]).expand(3, 2)).isfinite().all()
    assert actor(torch.randn(3, 4), greedy=True)[0][0].isfinite().all()
