import torch

from sheeprl.algos.muzero.utils import scalar_to_support


def reward_loss(predicted_reward, target_reward):
    support_size = (predicted_reward.shape[-1] - 1) // 2
    target_reward_vec = scalar_to_support(target_reward, support_size).float()
    return torch.nn.functional.cross_entropy(predicted_reward, target_reward_vec, reduction="none")


def value_loss(predicted_value, target_value):
    support_size = (predicted_value.shape[-1] - 1) // 2
    target_value_vec = scalar_to_support(target_value, support_size).float()
    return torch.nn.functional.cross_entropy(predicted_value, target_value_vec, reduction="none")


def policy_loss(predicted_policy_logits, target_policy):
    """Compute cross-entropy loss between predicted policy logits and target policy distribution.
    
    Args:
        predicted_policy_logits: Model output logits, shape [batch, num_actions]
        target_policy: Target probability distribution (e.g., from MCTS), shape [batch, num_actions]
    
    Returns:
        Loss per batch element, shape [batch]
    
    Note:
        PyTorch's cross_entropy with float targets (probabilities) computes:
        -sum(target * log_softmax(input)) which is the correct cross-entropy for distributions.
    """
    return torch.nn.functional.cross_entropy(predicted_policy_logits, target_policy, reduction="none")
