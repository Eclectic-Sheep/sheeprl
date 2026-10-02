"""PPO and A2C: the losses, the advantages."""

import pytest
import torch

from sheeprl.algos.ppo.loss import value_loss
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
