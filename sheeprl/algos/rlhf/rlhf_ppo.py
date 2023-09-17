import os
import time
from pathlib import Path
from typing import Dict, Union

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from sheeprl.algos.rlhf.collate import LeftPadCollate
from sheeprl.algos.rlhf.metrics import PPOMetricManager
from sheeprl.algos.rlhf.ppo_utils import AdaptiveKLController, FixedKLController, collect_rollout, ppo_step
from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))
from transformers import GenerationConfig, PreTrainedTokenizer

from sheeprl.algos.rlhf.args import ExploreArgs, GenerationArgs, ModelArgs, PPOArgs, TextDataArgs
from sheeprl.algos.rlhf.models import ActorModel, CasualModel, CriticModel, RewardModel
from sheeprl.algos.rlhf.utils import (
    compute_grad_norm,
    get_last_checkpoint_path,
    load_args_from_json,
    log_text,
    prepare_generation_config,
    prepare_optimizer_parameters,
    prepare_tokenizer,
    save_args_to_json,
    setup_finetuning,
    trainable_parameter_summary,
)
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm

__all__ = ["main"]


@torch.inference_mode()
def generate(
    actor_model: CasualModel,
    reward_model: CriticModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    example_prompt: Dict[str, torch.Tensor],
    device: torch.device,
):
    actor_model.eval()
    generated_input_ids = actor_model.generate(
        input_ids=example_prompt["input_ids"].to(device),
        attention_mask=example_prompt["attention_mask"].to(device),
        generation_config=generation_config,
        use_cache=True,
    )
    prompt_length = example_prompt["input_ids"].shape[1]
    generated_attention_mask = (generated_input_ids != generation_config.pad_token_id).int()
    generated_data = {"input_ids": generated_input_ids, "attention_mask": generated_attention_mask}
    reward = reward_model(**generated_data)[:, prompt_length:]
    action_mask = (generated_input_ids != generation_config.pad_token_id).int()[:, prompt_length:]
    last_token_idx = torch.argmax(torch.cumsum(action_mask, dim=1) * action_mask, dim=1, keepdim=True)
    reward_score = torch.gather(reward, dim=-1, index=last_token_idx).squeeze(-1)
    actor_model.train()
    return tokenizer.decode(generated_input_ids[0], skip_special_tokens=True), reward_score.item()


@register_algorithm()
def main():
    # Retrieve arguments
    parser = HfArgumentParser([PPOArgs, ModelArgs, ExploreArgs])
    dataclasses = parser.parse_args_into_dataclasses()
    train_args: PPOArgs = dataclasses[0]
    model_args: ModelArgs = dataclasses[1]
    gen_args: ExploreArgs = dataclasses[2]

    if train_args.sft_experiment_dir is None or train_args.rm_experiment_dir is None:
        raise ValueError("For PPO, checkpoints coming from SFT and RM training are required.")

    data_args_path = Path(train_args.data_dir) / "args.json"
    data_args = TextDataArgs.from_json(str(data_args_path))

    # Evaluation arguments
    eval_args = GenerationArgs()

    # Setup Fabric
    fabric = L.Fabric(accelerator="auto")
    fabric.launch()
    fabric.seed_everything(train_args.seed + fabric.global_rank)
    fabric.print(f"Fabric Rank: {fabric.global_rank}")

    # Setup Metrics
    metrics = PPOMetricManager(log_interval=train_args.log_interval).to(fabric.device)

    # Setup for rank 0
    if fabric.is_global_zero:
        # Setup Logger
        logger = TensorBoardLogger(train_args.experiment_dir)
        fabric._loggers = [logger]
        # Save args
        os.makedirs(train_args.experiment_dir, exist_ok=True)
        all_args = save_args_to_json(
            train_args=train_args,
            model_args=model_args,
            data_args=data_args,
            gen_args=gen_args,
        )
        # Log hyperparameters
        fabric.logger.log_hyperparams(all_args)

    # Setup Tokenizer
    tokenizer = prepare_tokenizer(model_args.model_name)

    sft_exp_args = load_args_from_json(experiment_dir=train_args.sft_experiment_dir)
    sft_model_args = ModelArgs(**sft_exp_args["model_args"])
    sft_checkpoint_path = get_last_checkpoint_path(train_args.sft_experiment_dir)

    # Reference actor model for checking kl divergence
    fabric.print("\nLoading reference model")
    ref_model = ActorModel.from_checkpoint(
        device=fabric.device,
        model_args=sft_model_args,
        path=sft_checkpoint_path,
        freeze=True,
    )
    ref_model.eval()
    trainable_parameter_summary(model=ref_model, show_names=False, fabric=fabric)

    # Actor model for PPO training
    fabric.print("\nLoading actor model")
    actor_model = ActorModel.from_checkpoint(
        device=fabric.device, model_args=sft_model_args, path=sft_checkpoint_path, freeze=True
    )
    setup_finetuning(fabric, actor_model, model_args)
    trainable_parameter_summary(model=actor_model, show_names=False, fabric=fabric)

    rm_exp_args = load_args_from_json(experiment_dir=train_args.rm_experiment_dir)
    rm_model_args = ModelArgs(**rm_exp_args["model_args"])
    rm_checkpoint_path = get_last_checkpoint_path(train_args.rm_experiment_dir)

    # Critic Model for PPO training
    fabric.print("\nLoading critic model")
    critic_model = CriticModel.from_checkpoint(
        device=fabric.device, model_args=rm_model_args, path=rm_checkpoint_path, freeze=True
    )
    setup_finetuning(fabric, critic_model, model_args)
    trainable_parameter_summary(model=critic_model, show_names=False, fabric=fabric)

    # Reward model
    fabric.print("\nLoading reward model")
    reward_model = RewardModel.from_checkpoint(
        device=fabric.device,
        model_args=rm_model_args,
        path=rm_checkpoint_path,
        freeze=True,
    )
    reward_model.eval()
    trainable_parameter_summary(model=reward_model, show_names=False, fabric=fabric)

    # Setup Generation Configs
    generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_args=model_args,
        gen_args=gen_args,
        fabric=fabric,
    )
    eval_generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_args=model_args,
        gen_args=eval_args,
        fabric=fabric,
    )

    # Setup Dataloaders
    collator = LeftPadCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)
    train_data = torch.load(Path(train_args.data_dir) / "finetune_train.pt")
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=train_args.rollout_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    example_prompt = torch.load(Path(train_args.data_dir) / "example_prompt.pt")

    # Setup Optimizer Scheduler fabric models
    optimizer_cls: Union[torch.optim.Adam, torch.optim.AdamW] = getattr(torch.optim, train_args.optimizer)

    actor_trainable_params, _, _ = prepare_optimizer_parameters(actor_model, weight_decay=train_args.weight_decay)
    actor_optimizer = optimizer_cls(
        actor_trainable_params,
        lr=train_args.actor_learning_rate,
        eps=train_args.optimizer_eps,
        betas=(train_args.optimizer_beta1, train_args.optimizer_beta2),
    )
    actor_model, actor_optimizer = fabric.setup(actor_model, actor_optimizer)

    critic_trainable_params, _, _ = prepare_optimizer_parameters(critic_model, weight_decay=train_args.weight_decay)
    critic_optimizer = optimizer_cls(
        critic_trainable_params,
        lr=train_args.critic_learning_rate,
        eps=train_args.optimizer_eps,
        betas=(train_args.optimizer_beta1, train_args.optimizer_beta2),
    )
    critic_model, critic_optimizer = fabric.setup(critic_model, critic_optimizer)

    reward_model = fabric.setup_module(reward_model)
    ref_model = fabric.setup_module(ref_model)

    if fabric.is_global_zero:
        gen_text, score = generate(
            actor_model=actor_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            generation_config=eval_generation_config,
            example_prompt=example_prompt,
            device=fabric.device,
        )
        log_text(fabric, gen_text, "info/example_sample", step=0)
        fabric.log("info/example_last_reward", score, step=0)

    num_training_steps = train_args.epochs * len(train_dataloader)

    # KL Controller
    if train_args.adaptive_kl_coeff:
        kl_controller = AdaptiveKLController(
            init_kl_coef=train_args.init_kl_coeff, target=train_args.target_kl_coeff, kl_horizon=num_training_steps
        )
    else:
        kl_controller = FixedKLController(kl_coeff=train_args.init_kl_coeff)

    actor_model.train()
    critic_model.train()
    iterator = tqdm(range(num_training_steps), disable=not fabric.is_global_zero)
    data_iterator = iter(train_dataloader)

    for k in iterator:
        # Setup counters and data
        if k % len(train_dataloader) == 0 or data_iterator is None:
            data_iterator = iter(train_dataloader)
        is_accumulating = (k) % train_args.gradient_accumulation_steps != 0
        last_step = k == num_training_steps - 1

        # Setup batch data
        batch = next(data_iterator)
        max_prompt_length = batch["prompt_input_ids"].shape[1]
        t0 = time.time()

        rollout, sample_output = collect_rollout(
            batch=batch,
            actor_model=actor_model,
            critic_model=critic_model,
            ref_model=ref_model,
            reward_model=reward_model,
            generation_config=generation_config,
            kl_controller=kl_controller,
            train_args=train_args,
            tokenizer=tokenizer,
            fabric=fabric,
            metrics=metrics,
        )
        time_rollout = time.time() - t0
        rollout_dataloader = DataLoader(
            rollout, batch_size=train_args.micro_batch_size, shuffle=True, collate_fn=lambda x: x
        )
        rollout_dataloader = fabric.setup_dataloaders(rollout_dataloader, use_distributed_sampler=False)
        for _ in range(train_args.ppo_epochs):
            accumulator_counter = 0
            for micro_batch in rollout_dataloader:
                is_accumulating = (accumulator_counter) % train_args.gradient_accumulation_steps != 0

                with fabric.no_backward_sync(actor_model, enabled=is_accumulating), fabric.no_backward_sync(
                    critic_model, enabled=is_accumulating
                ):
                    policy_loss, value_loss = ppo_step(
                        batch=micro_batch,
                        actor_model=actor_model,
                        critic_model=critic_model,
                        ppo_args=train_args,
                        max_prompt_length=max_prompt_length,
                    )
                    fabric.backward(policy_loss / train_args.gradient_accumulation_steps)
                    fabric.backward((value_loss * train_args.vf_coeff) / train_args.gradient_accumulation_steps)
                if not is_accumulating:
                    actor_grads = compute_grad_norm(model=actor_model)
                    fabric.clip_gradients(
                        actor_model, actor_optimizer, max_norm=train_args.gradient_clip_val, error_if_nonfinite=True
                    )
                    actor_optimizer.step()
                    actor_optimizer.zero_grad(set_to_none=True)

                    critic_grads = compute_grad_norm(model=critic_model)
                    fabric.clip_gradients(
                        critic_model, critic_optimizer, max_norm=train_args.gradient_clip_val, error_if_nonfinite=True
                    )
                    critic_optimizer.step()
                    critic_optimizer.zero_grad(set_to_none=True)
                accumulator_counter += 1

        time_ppo = time.time() - t0 - time_rollout
        with torch.no_grad():
            metrics.info_rollout_time.update(time_rollout)
            metrics.info_ppo_time.update(time_ppo)
            metrics.train_actor_loss.update(policy_loss.item())
            metrics.train_critic_loss.update(value_loss.item())
            metrics.info_actor_grad_norm.update(actor_grads)
            metrics.info_critic_grad_norm.update(critic_grads)
            metrics.info_kl_coeff.update(kl_controller.value)

        if k > 0 and (k % train_args.eval_interval == 0 or last_step):
            if fabric.is_global_zero:
                gen_text, score = generate(
                    actor_model=actor_model,
                    reward_model=reward_model,
                    tokenizer=tokenizer,
                    generation_config=eval_generation_config,
                    example_prompt=example_prompt,
                    device=fabric.device,
                )
                log_text(fabric, sample_output, "info/rollout_sample", step=k)
                log_text(fabric, gen_text, "info/example_sample", step=k)
                fabric.log("info/example_last_reward", score, step=k)

        fabric.barrier()
        if k % train_args.log_interval == 0 or last_step:
            computed_metrics = metrics.compute_all()
            metrics.log_all(fabric=fabric, step=k, metrics_dict=computed_metrics)

            if not iterator.disable:
                description = f"iter {k}, rollout-time: {time_rollout*1000:.2f}ms, ppo-time: {time_ppo*1000:.2f}ms"
                for metric_name, metric_value in computed_metrics.items():
                    if metric_name.startswith("info/") or metric_name.startswith("debug/"):
                        continue
                    description += f", {metric_name}: {metric_value:.3f}"
                iterator.set_description(description)

        if k > 0 and (k % train_args.save_interval == 0 or last_step):
            checkpoint_model: ActorModel = actor_model.module
            checkpoint_model.save_checkpoint(
                fabric=fabric,
                train_args=train_args,
                model_args=model_args,
                step=k,
            )
    fabric.print("Experiment output folder: ", train_args.experiment_dir)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    main()
