from torch import Tensor
from torch.distributions import Distribution


def ensemble_loss(embeds_dist: Distribution, embedded_obs: Tensor):
    -embeds_dist.log_prob(embedded_obs).mean()
