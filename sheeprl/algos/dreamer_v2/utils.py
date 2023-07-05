from typing import TYPE_CHECKING

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
        The mean and the standard deviation of the distribution of the stochastic state.
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = OneHotCategoricalStraightThrough(logits=logits)
    return dist.rsample()
