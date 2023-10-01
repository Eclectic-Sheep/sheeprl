from typing import Dict, Tuple, Union

import torch
from tensordict import make_tensordict

from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))

import lightning as L
from torch.utils.data import DataLoader
from transformers import GenerationConfig, PreTrainedTokenizer

from sheeprl.algos.rlhf.config_store.algo import PPOAlgoConfig
from sheeprl.algos.rlhf.loss import policy_loss, value_loss
from sheeprl.algos.rlhf.metrics import PPOMetricManager
from sheeprl.algos.rlhf.models import ActorModel, CriticModel


class FixedKLController:
    "Dummy KL controller that does not update."

    def __init__(self, kl_coeff):
        self.value = kl_coeff

    def update(self, current, n_steps):
        pass


# TODO: Adaptive KL horizon does not clearly understood.
# How do we assign the horizon?
class AdaptiveKLController:
    def __init__(self, init_kl_coeff: float, target_kl_coeff: float, kl_horizon: float, clip_range: float):
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


# These functions for mask computations are taken from TRL. They mask out not played tokens
# such as padding tokens.
def normalize(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


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
    actor_model: ActorModel,
    critic_model: CriticModel,
    ref_model: ActorModel,
    reward_model: CriticModel,
    kl_controller: Union[FixedKLController, AdaptiveKLController],
    generation_config: GenerationConfig,
    algo_cfg: PPOAlgoConfig,
    tokenizer: PreTrainedTokenizer,
    fabric: L.Fabric,
    metrics: PPOMetricManager,
) -> Dict[str, torch.Tensor]:
    actor_model.eval()
    critic_model.eval()

    # We have the batch as dictionary let's create tensordict
    # so we can create dataloader with Fabric that transfers the data
    # to correct devices.
    batch_tdict = make_tensordict(batch)
    mini_batch_dataloader = DataLoader(
        batch_tdict,
        shuffle=False,
        batch_size=algo_cfg.rollout_mini_batch_size,
        collate_fn=lambda x: x,
        num_workers=0,
        drop_last=False,
    )
    mini_batch_dataloader = fabric.setup_dataloaders(mini_batch_dataloader, use_distributed_sampler=False)
    rollout_dict_list = []

    # We use first generated token index - 1 to obtain correct logprobs.
    # Here we have batch of data fed into all models we have here is the input looks like:
    # Assuming padding tokens are `O` and input tokens are `I`
    # O O I I I
    # O O O I I (left padded batch)
    # O I I I I
    # After responses are generated we have new data assuming response tokens are `R`
    # O O I I I R R R O O O
    # O O O I I R R R R R O (padded from right side to longest text)
    # O I I I I R R R R R R
    start_token_idx = batch["prompt_input_ids"].size(1) - 1
    for i, mini_batch in enumerate(mini_batch_dataloader):
        prompt_input_ids = mini_batch["prompt_input_ids"]
        prompt_attention_mask = mini_batch["prompt_attention_mask"]
        data = {"input_ids": prompt_input_ids, "attention_mask": prompt_attention_mask}

        input_ids = actor_model.generate(**data, generation_config=generation_config)
        max_len_diff = generation_config.max_new_tokens - (input_ids.size(1) - prompt_input_ids.size(1))
        if max_len_diff > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, max_len_diff), value=tokenizer.pad_token_id)
        attention_masks = (input_ids != generation_config.pad_token_id).int()

        data = {"input_ids": input_ids, "attention_mask": attention_masks}
        # for logprobs we already omit the last tokens from computation
        actor_log_probs = actor_model(**data)[:, start_token_idx:]
        ref_log_probs = ref_model(**data)[:, start_token_idx:]
        # We need to also do the same for value and reward outputs
        values = critic_model(**data)[:, start_token_idx:-1]
        reward_outputs = reward_model(**data)[:, start_token_idx:-1]

        mini_batch_rollout = {
            "input_ids": input_ids,  # (B, T) (B, (prompt + generated))
            "attention_mask": attention_masks,  # (B, T) (B, (prompt + generated))
            "actor_log_probs": actor_log_probs,  # (B, num_new_tokens)
            "ref_log_probs": ref_log_probs,  # (B, num_new_tokens)
            "values": values,  # (B, num_new_tokens)
            "reward_outputs": reward_outputs,  # (B, num_new_tokens)
        }
        mini_batch_tdict = make_tensordict(mini_batch_rollout).cpu()
        rollout_dict_list.append(mini_batch_tdict)
        if i == 0:
            sample_from_rollout = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    rollout = torch.cat(rollout_dict_list, 0)
    action_mask = rollout["attention_mask"][:, start_token_idx:-1].int()
    reward_outputs = rollout.pop("reward_outputs")
    # we already removed the last token from action mask
    # we dont need to remove it from last_token_idx
    last_token_idx = torch.argmax(torch.cumsum(action_mask, dim=1) * action_mask, dim=1, keepdim=True)
    reward_scores = torch.gather(reward_outputs, dim=-1, index=last_token_idx).squeeze(-1)
    kl_div = rollout["actor_log_probs"] - rollout["ref_log_probs"]

    mean_kl_div = masked_mean(kl_div, action_mask).mean()
    if algo_cfg.clip_rewards:
        torch.clip_(reward_scores, -algo_cfg.reward_clip_value, algo_cfg.reward_clip_value)

    if algo_cfg.normalize_rewards:
        # we normalize the reward but do not shift the mean
        # TODO: Does it really important to normalize the rewards?
        reward_scores = normalize(reward_scores, shift_mean=False)

    # Rewards are made of two components:
    # 1. Per token kl divergence
    # 2. Last token reward
    # Combination of these two component creates the reward signal
    rewards = kl_div.detach().clone() * -kl_controller.value
    rewards.scatter_add_(dim=1, index=last_token_idx, src=reward_scores.unsqueeze(-1))
    values = rollout["values"]

    advantages, returns = compute_advantages_and_returns(
        rewards=rewards * action_mask,
        values=values * action_mask,
        gamma=algo_cfg.gae_gamma,
        lambd=algo_cfg.gae_lambd,
    )
    rollout["advantages"] = advantages
    rollout["returns"] = returns
    kl_controller.update(mean_kl_div, rollout["input_ids"].size(0))
    metrics.train_kl_div_mean.update(mean_kl_div.item())
    metrics.train_reward_mean.update(reward_scores.mean().item())
    metrics.debug_reward_scores(reward_scores)
    metrics.debug_advantages(advantages)
    metrics.debug_returns(returns)

    actor_model.train()
    critic_model.train()
    return rollout, sample_from_rollout


def ppo_step(
    batch: Dict[str, torch.Tensor],
    actor_model: torch.nn.Module,
    critic_model: torch.nn.Module,
    algo_cfg: PPOAlgoConfig,
    max_prompt_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generated_data = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

    old_log_probs = batch["actor_log_probs"]
    old_values = batch["values"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    start_token_idx = max_prompt_length - 1
    log_probs = actor_model(**generated_data)[:, start_token_idx:]  # (B, num_new_tokens)
    values = critic_model(**generated_data)[:, start_token_idx:-1]  # (B, num_new_tokens)
    action_mask = batch["attention_mask"][:, start_token_idx:-1].int()
    if algo_cfg.normalize_advantages:
        advantages = masked_normalize(advantages, action_mask)

    p_loss = policy_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        clip_coeff=algo_cfg.clip_coeff,
        action_mask=action_mask,
    )
    v_loss = value_loss(
        values=values,
        old_values=old_values,
        returns=returns,
        clip_coeff=algo_cfg.clip_coeff,
        action_mask=action_mask,
    )

    return p_loss, v_loss
