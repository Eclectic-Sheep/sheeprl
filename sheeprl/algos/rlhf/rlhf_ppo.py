import copy
import time
from pathlib import Path
from typing import Dict

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from sheeprl.algos.rlhf.agent import PPOAgent
from sheeprl.algos.rlhf.loss import policy_loss, value_loss
from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))
from transformers import GenerationConfig, PreTrainedTokenizer

from sheeprl.algos.rlhf.collate import LeftPadCollate
from sheeprl.algos.rlhf.config_store.algo import PPOAlgoConfig
from sheeprl.algos.rlhf.config_store.data import DataConfig, GenConfig
from sheeprl.algos.rlhf.config_store.model import ModelConfig
from sheeprl.algos.rlhf.data import TextDataset
from sheeprl.algos.rlhf.metrics import PPOMetricManager
from sheeprl.algos.rlhf.models import ActorModel
from sheeprl.algos.rlhf.ppo_utils import AdaptiveKLController, FixedKLController, collect_rollout, masked_normalize
from sheeprl.algos.rlhf.utils import (
    compute_grad_norm,
    log_text,
    prepare_generation_config,
    prepare_optimizer_parameters,
    validate_dataset,
)
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.registry import register_algorithm

__all__ = ["main"]


@torch.inference_mode()
def generate(
    agent: PPOAgent,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    example_prompt: Dict[str, torch.Tensor],
    device: torch.device,
):
    agent.actor.eval()
    generated_input_ids = agent.actor.module.generate(
        input_ids=example_prompt["input_ids"].to(device),
        attention_mask=example_prompt["attention_mask"].to(device),
        generation_config=generation_config,
        use_cache=True,
    )
    prompt_length = example_prompt["input_ids"].shape[1]
    generated_attention_mask = (generated_input_ids != generation_config.pad_token_id).int()
    generated_data = {"input_ids": generated_input_ids, "attention_mask": generated_attention_mask}
    reward = agent.reward(**generated_data)[:, prompt_length:]
    action_mask = (generated_input_ids != generation_config.pad_token_id).int()[:, prompt_length:]
    last_token_idx = torch.argmax(torch.cumsum(action_mask, dim=1) * action_mask, dim=1, keepdim=True)
    reward_score = torch.gather(reward, dim=-1, index=last_token_idx).squeeze(-1)
    agent.actor.train()
    return tokenizer.decode(generated_input_ids[0], skip_special_tokens=True), reward_score.item()


@register_algorithm()
def main(fabric: L.Fabric, cfg: Dict):
    algo_cfg = PPOAlgoConfig(**cfg.algo)
    model_cfg = ModelConfig(**cfg.model)
    data_cfg = DataConfig(**cfg.data)
    gen_cfg = GenConfig(**cfg.generation)
    actor_optim_cfg = cfg.actor_optimizer
    critic_optim_cfg = cfg.critic_optimizer

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, cfg)
    experiment_dir = Path(log_dir).parent
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)

    # Setup Metrics
    metrics = PPOMetricManager(log_interval=algo_cfg.log_interval).to(fabric.device)

    # Setup Dataloaders
    data_processor = validate_dataset(fabric, data_cfg)
    dataset_path = Path(data_processor.full_path)
    tokenizer = data_processor.tokenizer

    collator = LeftPadCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    train_dataset = TextDataset(dataframe_path=dataset_path / "finetune_train.pkl")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=algo_cfg.micro_batch_size,
        collate_fn=collator,
        num_workers=algo_cfg.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    example_prompt = torch.load(dataset_path / "example_prompt.pt")

    agent = PPOAgent(
        fabric=fabric,
        model_cfg=model_cfg,
        init_critic_with_rm=algo_cfg.init_critic_with_rm,
        sft_experiment_dir=algo_cfg.sft_experiment_dir,
        rm_experiment_dir=algo_cfg.rm_experiment_dir,
    )

    # Setup Generation Configs
    generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        gen_cfg=gen_cfg,
        fabric=fabric,
    )
    eval_gen_cfg = copy.deepcopy(gen_cfg)
    eval_gen_cfg.do_sample = False
    eval_generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        gen_cfg=eval_gen_cfg,
        fabric=fabric,
    )

    # Setup Optimizer Scheduler fabric models

    actor_trainable_params, _, _ = prepare_optimizer_parameters(agent.actor, weight_decay=actor_optim_cfg.weight_decay)
    actor_optimizer = hydra.utils.instantiate(
        actor_optim_cfg,
        params=actor_trainable_params,
        _convert_="partial",
    )
    actor_optimizer = fabric.setup_optimizers(actor_optimizer)

    critic_trainable_params, _, _ = prepare_optimizer_parameters(
        agent.critic, weight_decay=critic_optim_cfg.weight_decay
    )
    critic_optimizer = hydra.utils.instantiate(
        critic_optim_cfg,
        params=critic_trainable_params,
        _convert_="partial",
    )
    critic_optimizer = fabric.setup_optimizers(critic_optimizer)

    if fabric.is_global_zero:
        gen_text, score = generate(
            agent=agent,
            tokenizer=tokenizer,
            generation_config=eval_generation_config,
            example_prompt=example_prompt,
            device=fabric.device,
        )
        log_text(fabric, gen_text, "info/example_sample", step=0)
        fabric.log("info/example_last_reward", score, step=0)

    num_training_steps = algo_cfg.epochs * len(train_dataloader)

    # KL Controller
    if algo_cfg.adaptive_kl_coeff:
        kl_controller = AdaptiveKLController(
            init_kl_coef=algo_cfg.init_kl_coeff, target=algo_cfg.target_kl_coeff, kl_horizon=num_training_steps
        )
    else:
        kl_controller = FixedKLController(kl_coeff=algo_cfg.init_kl_coeff)

    iterator = tqdm(range(num_training_steps), disable=not fabric.is_global_zero)
    data_iterator = iter(train_dataloader)

    for k in iterator:
        agent.actor.train()
        agent.critic.train()
        # Setup counters and data
        if k % len(train_dataloader) == 0 or data_iterator is None:
            data_iterator = iter(train_dataloader)
        is_accumulating = (k) % algo_cfg.gradient_accumulation_steps != 0
        last_step = k == num_training_steps - 1

        # Setup batch data
        batch = next(data_iterator)
        max_prompt_length = batch["prompt_input_ids"].shape[1]
        t0 = time.time()

        rollout, sample_output = collect_rollout(
            batch=batch,
            agent=agent,
            generation_config=generation_config,
            kl_controller=kl_controller,
            algo_cfg=algo_cfg,
            tokenizer=tokenizer,
            fabric=fabric,
            metrics=metrics,
        )
        time_rollout = time.time() - t0
        rollout_dataloader = DataLoader(
            rollout, batch_size=algo_cfg.micro_batch_size, shuffle=True, collate_fn=lambda x: x
        )
        rollout_dataloader = fabric.setup_dataloaders(rollout_dataloader, use_distributed_sampler=False)
        for _ in range(algo_cfg.ppo_epochs):
            accumulator_counter = 0
            for micro_batch in rollout_dataloader:
                is_accumulating = (accumulator_counter) % algo_cfg.gradient_accumulation_steps != 0

                generated_data = {
                    "input_ids": micro_batch["input_ids"],
                    "attention_mask": micro_batch["attention_mask"],
                }
                old_log_probs = micro_batch["actor_log_probs"]
                old_values = micro_batch["values"]
                advantages = micro_batch["advantages"]
                returns = micro_batch["returns"]
                start_token_idx = max_prompt_length - 1
                action_mask = micro_batch["attention_mask"][:, start_token_idx:-1].int()
                if algo_cfg.normalize_advantages:
                    advantages = masked_normalize(advantages, action_mask)

                with fabric.no_backward_sync(agent.actor, enabled=is_accumulating):
                    log_probs = agent.actor(**generated_data)[:, start_token_idx:]  # (B, num_new_tokens)
                    p_loss = policy_loss(
                        log_probs=log_probs,
                        old_log_probs=old_log_probs,
                        advantages=advantages,
                        clip_coeff=algo_cfg.clip_coeff,
                        action_mask=action_mask,
                    )
                    fabric.backward(p_loss / algo_cfg.gradient_accumulation_steps)

                with fabric.no_backward_sync(agent.critic, enabled=is_accumulating):
                    values = agent.critic(**generated_data)[:, start_token_idx:-1]  # (B, num_new_tokens)
                    v_loss = value_loss(
                        values=values,
                        old_values=old_values,
                        returns=returns,
                        clip_coeff=algo_cfg.clip_coeff,
                        action_mask=action_mask,
                    )
                    fabric.backward((v_loss * algo_cfg.vf_coeff) / algo_cfg.gradient_accumulation_steps)

                if not is_accumulating:
                    actor_grads = compute_grad_norm(model=agent.actor)
                    fabric.clip_gradients(
                        agent.actor, actor_optimizer, max_norm=algo_cfg.gradient_clip_val, error_if_nonfinite=True
                    )
                    actor_optimizer.step()
                    actor_optimizer.zero_grad(set_to_none=True)

                    critic_grads = compute_grad_norm(model=agent.critic)
                    fabric.clip_gradients(
                        agent.critic, critic_optimizer, max_norm=algo_cfg.gradient_clip_val, error_if_nonfinite=True
                    )
                    critic_optimizer.step()
                    critic_optimizer.zero_grad(set_to_none=True)
                accumulator_counter += 1

        time_ppo = time.time() - t0 - time_rollout
        with torch.no_grad():
            metrics.info_rollout_time.update(time_rollout)
            metrics.info_ppo_time.update(time_ppo)
            metrics.train_actor_loss.update(p_loss.item())
            metrics.train_critic_loss.update(v_loss.item())
            metrics.info_actor_grad_norm.update(actor_grads)
            metrics.info_critic_grad_norm.update(critic_grads)
            metrics.info_kl_coeff.update(kl_controller.value)

        if k > 0 and (k % algo_cfg.eval_interval == 0 or last_step):
            if fabric.is_global_zero:
                gen_text, score = generate(
                    agent=agent,
                    tokenizer=tokenizer,
                    generation_config=eval_generation_config,
                    example_prompt=example_prompt,
                    device=fabric.device,
                )
                log_text(fabric, sample_output, "info/rollout_sample", step=k)
                log_text(fabric, gen_text, "info/example_sample", step=k)
                fabric.log("info/example_last_reward", score, step=k)

        fabric.barrier()
        if k % algo_cfg.log_interval == 0 or last_step:
            computed_metrics = metrics.compute_all()
            metrics.log_all(fabric=fabric, step=k, metrics_dict=computed_metrics)

            if not iterator.disable:
                description = f"iter {k}, rollout-time: {time_rollout*1000:.2f}ms, ppo-time: {time_ppo*1000:.2f}ms"
                for metric_name, metric_value in computed_metrics.items():
                    if metric_name.startswith("info/") or metric_name.startswith("debug/"):
                        continue
                    description += f", {metric_name}: {metric_value:.3f}"
                iterator.set_description(description)

        if k > 0 and (k % algo_cfg.save_interval == 0 or last_step):
            checkpoint_model: ActorModel = agent.actor.module
            checkpoint_model.save_checkpoint(
                fabric=fabric,
                experiment_dir=experiment_dir,
                model_cfg=model_cfg,
                step=k,
            )
    fabric.print("Experiment output folder: ", experiment_dir)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    main()
