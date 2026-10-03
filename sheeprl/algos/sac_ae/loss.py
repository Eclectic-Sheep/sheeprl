from torch import Tensor


def entropy_loss(log_alpha: Tensor, logprobs: Tensor, target_entropy: Tensor) -> Tensor:
    """The loss of the temperature of the official implementation (`SacAeAgent.update_actor_and_alpha` of
    https://github.com/denisyarats/pytorch_sac_ae): the temperature, not its logarithm (as in SAC), times the difference
    between the entropy of the actions and the target one."""
    return (log_alpha.exp() * (-logprobs - target_entropy)).mean()
