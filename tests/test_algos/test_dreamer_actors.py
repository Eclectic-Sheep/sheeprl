"""The actors of DreamerV2 and DreamerV3 (DreamerV1 and the Plan2Explore variants use them too): their distributions
over continuous actions and their entropy, and the actors of MineDojo."""

import pytest
import torch
from torch import nn
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution

from sheeprl.algos.dreamer_v2.agent import Actor as DV2Actor
from sheeprl.algos.dreamer_v3.agent import Actor as DV3Actor
from sheeprl.algos.dreamer_v3.agent import MinedojoActor as DV3MinedojoActor
from sheeprl.utils.distribution import entropy

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


@pytest.mark.parametrize(
    "actor_cls,distribution",
    [
        (DV2Actor, "tanh_normal"),
        (DV2Actor, "normal"),
        (DV2Actor, "trunc_normal"),
        (DV3Actor, "tanh_normal"),
        (DV3Actor, "normal"),
        (DV3Actor, "scaled_normal"),
    ],
)
def test_greedy_continuous_actions_are_the_most_likely_sample_of_every_environment(actor_cls, distribution):
    # The samples were indexed with the best index of every environment at once: with N environments the actions had
    # N * N * A values, the most likely samples of every environment for every environment
    kwargs = {"action_clip": 0.0} if actor_cls is DV3Actor else {}
    actor = make_actor(actor_cls, distribution, actions_dim=(2,), **kwargs)
    num_envs = 3
    state = torch.randn(1, num_envs, LATENT)
    torch.manual_seed(1)
    (actions,), (dist,) = actor(state, greedy=True)
    assert actions.shape == (1, num_envs, 2)

    torch.manual_seed(1)
    sample = dist.sample((100,))
    log_prob = dist.log_prob(sample)
    for env in range(num_envs):
        torch.testing.assert_close(actions[0, env], sample[log_prob[:, 0, env].argmax(), 0, env])


def test_the_minedojo_actor_of_dreamer_v3_samples_its_actions():
    # It took the most likely actions unless asked otherwise: the imagination, which doesn't ask, learned from the
    # greedy actions only
    actor = make_actor(DV3MinedojoActor, "discrete", actions_dim=(4, 3, 3), is_continuous=False)
    state = torch.randn(1, 200, LATENT)
    actions, dists = actor(state)
    greedy_actions, _ = actor(state, greedy=True)
    for action, greedy_action, dist in zip(actions, greedy_actions, dists):
        torch.testing.assert_close(greedy_action, dist.mode)
        assert (action != greedy_action).any(-1).float().mean() > 0.2


@pytest.mark.parametrize("actor_cls", [DV2Actor, DV3Actor])
def test_the_entropy_of_the_tanh_normal_distribution_is_estimated_from_samples(actor_cls):
    # It has no analytic entropy: the actor losses used 0 instead, so `algo.actor.ent_coef` did nothing
    actor = make_actor(actor_cls, "tanh_normal")
    _, (dist,) = actor(torch.randn(1, 4, LATENT))
    torch.manual_seed(0)
    estimate = entropy(dist, n_samples=20000)
    assert estimate.shape == (1, 4)
    # The mean negative log-probability of many samples, in double precision
    base = dist.base_dist.base_dist
    reference_dist = Independent(
        TransformedDistribution(Normal(base.loc.double(), base.scale.double()), TanhTransform()), 1
    )
    samples = reference_dist.sample((200000,))
    reference = -reference_dist.log_prob(samples).mean(0)
    torch.testing.assert_close(estimate.detach().double(), reference.detach(), atol=0.03, rtol=0)
    # It trains the actor
    estimate.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.mlp_heads.parameters())


def test_the_entropy_estimate_is_finite_where_the_tanh_saturates():
    dist = Independent(TransformedDistribution(Normal(torch.full((3, 2), 5.0), 4.0), TanhTransform()), 1)
    torch.manual_seed(0)
    assert torch.isfinite(entropy(dist)).all()
    # The analytic entropy of the other distributions
    normal = Independent(Normal(torch.zeros(3, 2), 2.0), 1)
    torch.testing.assert_close(entropy(normal), normal.entropy())
