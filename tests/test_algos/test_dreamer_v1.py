"""DreamerV1 (and P2E-DV1): the continue loss, the episode starts in the sequences, the exploration noise and the
posterior of the player."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Bernoulli, Independent, Normal

from sheeprl.algos.dreamer_v1 import agent
from sheeprl.algos.dreamer_v1.agent import RSSM, PlayerDV1, RecurrentModel
from sheeprl.algos.dreamer_v1.loss import reconstruction_loss
from sheeprl.algos.dreamer_v2.agent import Actor


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
