from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from sheeprl.algos.rlhf.config_store.algo import RM_LOSS_TYPE
from sheeprl.algos.rlhf.utils import compute_masked_logprobs


def reward_loss_last_token(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This loss computes the logsigmoid of the difference between the chosen and
    rejected rewards from last generated non-terminal token"""
    mask_chosen = chosen != pad_token_id
    mask_rejected = rejected != pad_token_id

    # last non-padding token is the terminal one
    # we want to retrieve reward that leads to that state(token)
    last_chosen_token_idx = torch.argmax(torch.cumsum(mask_chosen, dim=1) * mask_chosen, dim=1, keepdim=True) - 1
    last_rejected_token_idx = torch.argmax(torch.cumsum(mask_rejected, dim=1) * mask_rejected, dim=1, keepdim=True) - 1
    last_chosen_rewards = torch.gather(chosen_rewards, dim=-1, index=last_chosen_token_idx).squeeze(-1)
    last_rejected_rewards = torch.gather(rejected_rewards, dim=-1, index=last_rejected_token_idx).squeeze(-1)

    filtered_rewards = last_chosen_rewards - last_rejected_rewards
    return -F.logsigmoid(filtered_rewards).mean(), last_chosen_rewards, last_rejected_rewards


def reward_loss_average(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This loss computes the logsigmoid of the difference between the chosen and
    rejected rewards from average of all output tokens excluding padding tokens
    """
    mask_chosen = chosen != pad_token_id  # (B, T)
    mask_rejected = rejected != pad_token_id  # (B, T)

    divergence = ((chosen - rejected) != 0).int().argmax(1)

    # TODO: Can we implement it in vectorized way?
    for i, d in enumerate(divergence):
        mask_chosen[i, :d] = 0
        mask_rejected[i, :d] = 0

    # last non-padding token is the terminal one
    # we want to retrieve reward that leads to that state(token)
    # TODO: check if everytime this is true
    last_chosen_token_idx = torch.argmax(torch.cumsum(mask_chosen, dim=1) * mask_chosen, dim=1, keepdim=True) - 1
    last_rejected_token_idx = torch.argmax(torch.cumsum(mask_rejected, dim=1) * mask_rejected, dim=1, keepdim=True) - 1
    mask_chosen[:, last_chosen_token_idx + 1] = 0
    mask_rejected[:, last_rejected_token_idx + 1] = 0
    last_chosen_rewards = torch.gather(chosen_rewards, dim=-1, index=last_chosen_token_idx).squeeze(-1)
    last_rejected_rewards = torch.gather(rejected_rewards, dim=-1, index=last_rejected_token_idx).squeeze(-1)

    chosen_rewards_average = chosen_rewards * mask_chosen
    chosen_rewards_average = chosen_rewards_average.sum(dim=1) / mask_chosen.sum(dim=1)
    rejected_rewards_average = rejected_rewards * mask_rejected
    rejected_rewards_average = rejected_rewards_average.sum(dim=1) / mask_rejected.sum(dim=1)

    filtered_rewards = chosen_rewards_average - rejected_rewards_average
    return -F.logsigmoid(filtered_rewards).mean(), last_chosen_rewards, last_rejected_rewards


def reward_loss_per_sample(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This loss computes the logsigmoid of the difference between the chosen and rejected rewards
    from every token in the sequence masked by the pad token id. It is adapted from
    https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py#L37
    for each example in the batch:
        - find the index where the chosen and rejected sequences diverge
        - find the index of the last non terminal token in the sequence
        - compute the loss for the tokens between the divergence index and the last token index

    Returns:
        loss: the mean loss for the batch
        chosen_last_rewards: the last reward for the chosen sequence for each example in the batch
        rejected_last_rewards: the last reward for the rejected sequence for each example in the batch
    """
    batch_size = chosen.shape[0]
    sequence_len = chosen.shape[1]
    loss: torch.Tensor = torch.tensor(0.0, device=chosen.device)
    chosen_last_rewards = []
    rejected_last_rewards = []
    total_num_samples = 0
    for i in range(batch_size):
        # Get the chosen and rejected actions for the current sample
        chosen_actions = chosen[i]
        rejected_actions = rejected[i]

        # Get the rewards for the chosen and rejected actions for the current example
        chosen_complete_rewards = chosen_rewards[i]
        rejected_complete_rewards = rejected_rewards[i]

        # Find the index where the action sequence diverge
        divergence_ind = (chosen_actions != rejected_actions).nonzero()
        if len(divergence_ind) == 0:
            divergence_ind = sequence_len - 1
        else:
            divergence_ind = divergence_ind[0]

        # Find padding tokens
        pad_mask_chosen = (chosen_actions == pad_token_id).nonzero()
        if chosen_actions[0] == pad_token_id:
            # If the first token is a pad token, we want to exclude it from the mask
            pad_mask_chosen = pad_mask_chosen[1:]
        chosen_last_index = pad_mask_chosen[0].item() if len(pad_mask_chosen) > 0 else chosen.shape[1]
        pad_mask_rejected = (rejected_actions == pad_token_id).nonzero()
        if rejected_actions[0] == pad_token_id:
            # If the first token is a pad token, we want to exclude it from the mask
            pad_mask_rejected = pad_mask_rejected[1:]
        rejected_last_index = pad_mask_rejected[0].item() if len(pad_mask_rejected) > 0 else rejected.shape[1]
        end_ind = max(chosen_last_index, rejected_last_index)

        if divergence_ind > end_ind:
            continue

        if divergence_ind == end_ind:
            # If the divergence index is the same as the end index, we want to include the last token
            divergence_ind -= 1

        # Get the rewards for the chosen and rejected sequences after the divergence index
        chosen_filtered_rewards = chosen_complete_rewards[divergence_ind:end_ind]
        rejected_filtered_rewards = rejected_complete_rewards[divergence_ind:end_ind]

        # Compute the loss for the current example
        filtered_rewards = chosen_filtered_rewards - rejected_filtered_rewards
        loss.add_(-F.logsigmoid(filtered_rewards).mean())

        # Get the last non-padding token rewards for the current example
        chosen_last_rewards.append(chosen_complete_rewards[chosen_last_index - 1])
        rejected_last_rewards.append(rejected_complete_rewards[rejected_last_index - 1])
        total_num_samples += 1

    loss.div_(total_num_samples)
    chosen_last_rewards = torch.stack(chosen_last_rewards)
    rejected_last_rewards = torch.stack(rejected_last_rewards)

    return loss, chosen_last_rewards, rejected_last_rewards


def load_reward_loss(reward_loss_type: str):
    if reward_loss_type == RM_LOSS_TYPE.AVERAGE:
        return reward_loss_average
    elif reward_loss_type == RM_LOSS_TYPE.LAST_TOKEN:
        return reward_loss_last_token
    elif reward_loss_type == RM_LOSS_TYPE.PER_SAMPLE:
        return reward_loss_per_sample
    else:
        raise ValueError(f"Invalid reward loss type: {reward_loss_type}")


def finetune_loss(
    outputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100, label_smoothing: float = 0.0
) -> torch.Tensor:
    outputs = outputs[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=ignore_index, label_smoothing=label_smoothing
    )
    return loss


def dpo_loss(
    batch: Dict[str, torch.Tensor],
    actor_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    beta: float,
    ignore_index: int,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Adapted from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L45C1-L50C110"""
    chosen_input_ids = batch["chosen_input_ids"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    chosen_targets = batch["chosen_targets"]
    rejected_input_ids = batch["rejected_input_ids"]
    rejected_attention_mask = batch["rejected_attention_mask"]
    rejected_targets = batch["rejected_targets"]

    with torch.inference_mode():
        ref_chosen_logprobs = ref_model(
            input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False
        )
        ref_rejected_logprobs = ref_model(
            input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False
        )
    actor_chosen_logprobs = actor_model(
        input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False
    )
    actor_rejected_logprobs = actor_model(
        input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False
    )

    masked_actor_chosen_logps = compute_masked_logprobs(actor_chosen_logprobs, chosen_targets, ignore_index)
    masked_actor_rejected_logps = compute_masked_logprobs(actor_rejected_logprobs, rejected_targets, ignore_index)
    masked_reference_chosen_logps = compute_masked_logprobs(ref_chosen_logprobs, chosen_targets, ignore_index)
    masked_reference_rejected_logps = compute_masked_logprobs(ref_rejected_logprobs, rejected_targets, ignore_index)

    actor_logratios = masked_actor_chosen_logps - masked_actor_rejected_logps
    ref_logratios = masked_reference_chosen_logps - masked_reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = actor_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (masked_actor_chosen_logps - masked_reference_chosen_logps).detach()
    rejected_rewards = beta * (masked_actor_rejected_logps - masked_reference_rejected_logps).detach()
    return losses.mean(), chosen_rewards, rejected_rewards


def policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coeff: float,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_ratio = (log_probs - old_log_probs) * action_mask
    ratio = torch.exp(log_ratio)
    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff)
    policy_loss = torch.max(policy_loss_1, policy_loss_2)
    if action_mask is not None:
        policy_loss = torch.sum(policy_loss * action_mask) / action_mask.sum()
    else:
        policy_loss = policy_loss.mean()
    return policy_loss


def value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_coeff: float,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    values_clipped = torch.clamp(values, old_values - clip_coeff, old_values + clip_coeff)
    value_loss1 = F.mse_loss(values, returns, reduction="none")
    value_loss2 = F.mse_loss(values_clipped, returns, reduction="none")
    value_loss = torch.max(value_loss1, value_loss2)
    if action_mask is not None:
        value_loss = torch.sum(value_loss * action_mask) / action_mask.sum()
    else:
        value_loss = value_loss.mean()
    return value_loss
