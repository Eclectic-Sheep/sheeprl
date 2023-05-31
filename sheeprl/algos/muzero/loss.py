import torch


def reward_loss(predicted_reward, target_reward):
    return torch.nn.functional.mse_loss(predicted_reward, target_reward)


def value_loss(predicted_value, target_value):
    return torch.nn.functional.mse_loss(predicted_value, target_value)


def policy_loss(predicted_policy_logits, target_policy):
    return torch.nn.functional.cross_entropy(predicted_policy_logits, target_policy)
