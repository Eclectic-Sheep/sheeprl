from typing import Dict, Optional, Tuple, Union
import torch
from tensordict import make_tensordict

from sheeprl.algos.rlhf.args import PPOArgs
from sheeprl.algos.rlhf.loss import policy_loss, value_loss

from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))
from transformers import GenerationConfig, PreTrainedTokenizer


class FixedKLController:
    def __init__(self, kl_coeff):
        self.value = kl_coeff

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    def __init__(self, init_kl_coeff, target_kl_coeff, kl_horizon, clip_range):
        self.value = init_kl_coeff
        self.target_kl_coeff = target_kl_coeff
        self.kl_horizon = kl_horizon
        self.clip_range = clip_range

    def update(self, current, n_steps):
        target = self.target_kl_coeff
        proportional_error = torch.clamp(current / target - 1, -self.clip_range, self.clip_range)
        mult = 1 + proportional_error * n_steps / self.kl_horizon
        self.value *= mult


@torch.no_grad()
def estimate_kl_divergence(actor_log_probs: torch.Tensor, ref_log_probs: torch.Tensor):
    # http://joschu.net/blog/kl-approx.html
    ratio = actor_log_probs - ref_log_probs  # (B, T)
    estimated_kl = (torch.exp(ratio) - 1) - ratio
    return estimated_kl


@torch.no_grad()
def compute_advantages_and_returns(
    rewards: torch.Tensor, values: torch.Tensor, start: int = 0, gamma: float = 0.99, lambd: float = 0.95
):
    # Adopted from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ppo_trainer.py
    lastgaelam = 0
    advantages_reversed = []
    end = rewards.size()[-1]
    for t in reversed(range(start, end)):
        nextvalues = values[:, t + 1] if t < end - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages, returns


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim, keepdim=True)
    mask_sum = mask.sum(dim=dim, keepdim=True)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def masked_normalize(
    tensor: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    normalized = mean_centered * var.clamp(min=eps).rsqrt()
    if not shift_mean:
        normalized += mean
    return normalized


@torch.inference_mode()
def collect_rollout(
    batch: Dict[str, torch.Tensor],
    actor_model: torch.nn.Module,
    critic_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    reward_model: torch.nn.Module,
    kl_controller: Union[FixedKLController, AdaptiveKLController],
    generation_config: GenerationConfig,
    train_args: PPOArgs,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    actor_model.eval()
    critic_model.eval()
    num_new_tokens = generation_config.max_new_tokens
    data = {"input_ids": batch["prompt_input_ids"], "attention_mask": batch["prompt_attention_mask"]}
    input_ids = actor_model.generate(**data, generation_config=generation_config, use_cache=True)
    attention_mask = (input_ids != generation_config.pad_token_id).int()
    data = {"input_ids": input_ids, "attention_mask": attention_mask}

    # get actions and values
    actor_log_probs = actor_model(**data)[:, -num_new_tokens:]  # (B, num_new_tokens)
    ref_log_probs = ref_model(**data)[:, -num_new_tokens:]  # (B, num_new_tokens)
    values = critic_model(**data)[:, -num_new_tokens:]  # (B, num_new_tokens)
    action_masks = attention_mask[:, -num_new_tokens:]  # (B, num_new_tokens)
    reward_outputs = reward_model(**data)[:, -num_new_tokens:]  # (B, num_new_tokens)

    last_token_idx = torch.argmax(torch.cumsum(action_masks, dim=1) * action_masks, dim=1, keepdim=True)
    reward_scores = torch.gather(reward_outputs, dim=-1, index=last_token_idx).squeeze(-1)

    estimated_kl_div = estimate_kl_divergence(actor_log_probs=actor_log_probs, ref_log_probs=ref_log_probs)
    mean_kl_div = estimated_kl_div.sum(dim=1).mean()

    rewards = estimated_kl_div.detach().clone()
    rewards.scatter_add_(dim=1, index=last_token_idx, src=reward_scores.unsqueeze(-1))
    kl_controller.update(mean_kl_div, n_steps=rewards.shape[1])

    if train_args.normalize_rewards:
        rewards = masked_normalize(rewards, action_masks, shift_mean=False)
    advantages, returns = compute_advantages_and_returns(
        rewards=rewards * action_masks,
        values=values * action_masks,
        gamma=train_args.gae_gamma,
        lambd=train_args.gae_lambd,
    )
    if train_args.normalize_advantages:
        advantages = masked_normalize(advantages, action_masks)
    rollout = {
        "input_ids": input_ids,  # (B, T) (B, (prompt + generated))
        "attention_mask": attention_mask,  # (B, T) (B, (prompt + generated))
        "actor_log_probs": actor_log_probs,  # (B, num_new_tokens)
        "advantages": advantages,  # (B, num_new_tokens)
        "values": values,  # (B, num_new_tokens)
        "returns": returns,  # (B, num_new_tokens)
        "action_mask": action_masks,  # (B, num_new_tokens)
    }
    rollout = make_tensordict(rollout)
    sample_from_rollout = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    metrics = {
        "train/kl_div": mean_kl_div.item(),
        "train/last_reward_mean": reward_scores.mean().item(),
        "info/last_reward_std": reward_scores.std().item(),
        "info/all_reward_mean": rewards.mean().item(),
        "info/all_reward_std": rewards.std().item(),
        "info/advantage_mean": advantages.mean().item(),
        "info/advantage_std": advantages.std().item(),
        "info/returns_mean": returns.mean().item(),
        "info/returns_std": returns.std().item(),
        "info/rollout_sample": sample_from_rollout,
    }
    actor_model.train()
    critic_model.train()
    return rollout, metrics


def ppo_step(
    batch: Dict[str, torch.Tensor],
    actor_model: torch.nn.Module,
    critic_model: torch.nn.Module,
    ppo_args: PPOArgs,
    num_new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generated_data = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    action_mask = batch["action_mask"]
    old_log_probs = batch["actor_log_probs"]
    log_probs = actor_model(**generated_data)[:, -num_new_tokens:]  # (B, num_new_tokens)
    values = critic_model(**generated_data)[:, -num_new_tokens:]  # (B, num_new_tokens)
    advantages = batch["advantages"]
    # normalized_advantages = masked_normalize(advantages, action_mask)

    p_loss = policy_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        clip_coeff=ppo_args.clip_coeff,
        action_mask=action_mask,
    )
    v_loss = value_loss(
        values=values,
        old_values=batch["values"],
        returns=batch["returns"],
        clip_coeff=ppo_args.clip_coeff,
        vf_coeff=ppo_args.vf_coeff,
        action_mask=action_mask,
    )

    return p_loss, v_loss
