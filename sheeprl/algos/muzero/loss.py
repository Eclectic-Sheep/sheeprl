import torch

from sheeprl.utils.utils import symsqrt, two_hot_encoder


def reward_loss(predicted_reward, target_reward):
    support_size = (predicted_reward.shape[-1] - 1) // 2
    target_reward = two_hot_encoder(symsqrt(target_reward), support_size)
    return torch.nn.functional.cross_entropy(predicted_reward, target_reward)


def value_loss(predicted_value, target_value):
    support_size = (predicted_value.shape[-1] - 1) // 2
    target_value = two_hot_encoder(symsqrt(target_value), support_size)
    return torch.nn.functional.cross_entropy(predicted_value, target_value)


def policy_loss(predicted_policy_logits, target_policy):
    return torch.nn.functional.cross_entropy(predicted_policy_logits, target_policy)
