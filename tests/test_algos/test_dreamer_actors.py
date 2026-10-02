"""The actors of DreamerV2 and DreamerV3 (DreamerV1 and the Plan2Explore variants use them too): their distributions
over continuous actions."""

import pytest
import torch
from torch import nn

from sheeprl.algos.dreamer_v2.agent import Actor as DV2Actor
from sheeprl.algos.dreamer_v3.agent import Actor as DV3Actor

LATENT, UNITS = 6, 8


def make_actor(actor_cls, distribution, actions_dim=(2,), is_continuous=True, **kwargs):
    torch.manual_seed(0)
    return actor_cls(
        latent_state_size=LATENT,
        actions_dim=list(actions_dim),
        is_continuous=is_continuous,
        distribution_cfg={"type": distribution},
        init_std=0.0,
        min_std=0.1,
        dense_units=UNITS,
        mlp_layers=1,
        **kwargs,
    )


@pytest.mark.parametrize("actor_cls", [DV2Actor, DV3Actor])
def test_the_normal_distribution_has_a_positive_std(actor_cls):
    # The std was the output of the network as it was: negative for half of the inputs, with NaN log-probabilities
    actor = make_actor(actor_cls, "normal")
    head = actor.mlp_heads[0]
    with torch.no_grad():
        # Mean 0, std output -3 for every input
        head.weight.zero_()
        head.bias.copy_(torch.tensor([0.0, 0.0, -3.0, -3.0]))
    actions, (dist,) = actor(torch.randn(1, 5, LATENT))
    std = dist.base_dist.scale
    assert (std > 0.1).all()
    torch.testing.assert_close(std, torch.full_like(std, nn.functional.softplus(torch.tensor(-3.0)).item() + 0.1))
    assert torch.isfinite(dist.log_prob(actions[0])).all()
    assert torch.isfinite(dist.entropy()).all()
