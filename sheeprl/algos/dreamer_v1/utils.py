from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal


def compute_stochastic_state(
    state_information: Tensor,
    event_shape: int = 1,
    min_std: float = 0.1,
) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
    """
    Compute the stochastic state from the information of the distribution of the stochastic state.

    Args:
        state_information (Tensor): information about the distribution of the stochastic state,
            it is the output of either the representation model or the transition model.
        event_shape (int): how many batch dimensions have to be reinterpreted as event dims.
            Default to 1.
        min_std (float): the minimum value for the standard deviation.
            Default to 0.1.

    Returns:
        The mean and the standard deviation of the distribution of the stochastic state.
        The sampled stochastic state.
    """
    mean, std = torch.chunk(state_information, 2, -1)
    std = F.softplus(std) + min_std
    state_distribution: Distribution = Normal(mean, std)
    if event_shape:
        # it is necessary an Independent distribution because
        # it is necessary to create (batch_size * sequence_length) independent distributions,
        # each producing a sample of size equal to the stochastic size
        state_distribution = Independent(state_distribution, event_shape)
    stochastic_state = state_distribution.rsample()
    return (mean, std), stochastic_state
