from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor
from torch.distributions import OneHotCategoricalStraightThrough

if TYPE_CHECKING:
    pass


def compute_stochastic_state(
    logits: Tensor,
    discrete: int = 32,
) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = OneHotCategoricalStraightThrough(logits=logits)
    return dist.rsample()


def init_weights(m: nn.Module):
    """
    Initialize the parameters of the m module acording to a
    Xavier Normal distribution

    Args:
        m (nn.Module): the module to be initialized.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
