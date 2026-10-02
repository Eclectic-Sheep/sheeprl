"""PPO and A2C: the rewards of the rollout, the losses, the advantages."""

import numpy as np
import pytest
import torch

from sheeprl.algos.ppo.loss import value_loss
from sheeprl.algos.ppo.utils import bootstrap_truncated
from sheeprl.utils.utils import normalize_tensor


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_value_loss_has_the_same_scale_with_and_without_clipping(reduction):
    # When the new values stay within `clip_coef` of the old ones, clipping changes nothing: the two losses are equal
    # (the clipped one was halved and always averaged)
    torch.manual_seed(0)
    old_values = torch.randn(16, 1)
    new_values = old_values + 0.1 * torch.rand(16, 1)
    returns = torch.randn(16, 1)
    unclipped = value_loss(new_values, old_values, returns, 0.2, clip_vloss=False, reduction=reduction)
    clipped = value_loss(new_values, old_values, returns, 0.2, clip_vloss=True, reduction=reduction)
    torch.testing.assert_close(clipped, unclipped)


def test_normalize_tensor_of_one_element():
    # The last minibatch of a rollout can hold a single advantage: it has no standard deviation (it was NaN)
    torch.testing.assert_close(normalize_tensor(torch.tensor([[2.5]])), torch.tensor([[2.5]]))
    mask = torch.tensor([[True], [False]])
    torch.testing.assert_close(normalize_tensor(torch.tensor([[2.5], [1.0]]), mask=mask), torch.tensor([2.5]))
    normalized = normalize_tensor(torch.tensor([[1.0], [3.0]]))
    torch.testing.assert_close(normalized.mean(), torch.tensor(0.0))


@pytest.mark.parametrize("clip_rewards", [False, True])
def test_only_the_truncated_episodes_are_bootstrapped(clip_rewards):
    # Three environments: truncated, truncated and terminated in the same step (gymnasium's `TimeLimit` truncates also
    # when the termination falls on the last allowed step), terminated. Only the first one is bootstrapped, with the
    # value of its final observation added after the reward is clipped (it was clipped with it)
    rewards = np.full(3, 3.0)
    if clip_rewards:
        rewards = np.tanh(rewards)
    asked = []

    def final_values(env_idxes):
        asked.append(env_idxes.tolist())
        return np.full((len(env_idxes), 1), 2.0)

    terminated = np.array([False, True, True])
    truncated = np.array([True, True, False])
    bootstrapped = bootstrap_truncated(rewards, terminated, truncated, final_values, gamma=0.9)
    reward = np.tanh(3.0) if clip_rewards else 3.0
    np.testing.assert_allclose(bootstrapped, [reward + 0.9 * 2.0, reward, reward])
    assert asked == [[0]]
    # The rewards given are left as they are
    np.testing.assert_allclose(rewards, np.full(3, reward))
