import numpy as np
import torch

from sheeprl.algos.muzero.utils import scalar_to_support


def reward_loss(predicted_reward: torch.Tensor, target_reward: np.ndarray):
    support_size = (predicted_reward.shape[-1] - 1) // 2
    target_reward = scalar_to_support(target_reward, support_size)
    target_reward = torch.as_tensor(target_reward, dtype=predicted_reward.dtype, device=predicted_reward.device)
    return torch.nn.functional.cross_entropy(predicted_reward, target_reward)


def value_loss(predicted_value, target_value):
    support_size = (predicted_value.shape[-1] - 1) // 2
    target_value = scalar_to_support(target_value, support_size)
    target_value = torch.as_tensor(target_value, dtype=predicted_value.dtype, device=predicted_value.device)
    return torch.nn.functional.cross_entropy(predicted_value, target_value)


def policy_loss(predicted_policy_logits: torch.Tensor, target_policy: np.ndarray):
    target_policy = torch.as_tensor(
        target_policy, dtype=predicted_policy_logits.dtype, device=predicted_policy_logits.device
    )
    return torch.nn.functional.cross_entropy(predicted_policy_logits, target_policy)
