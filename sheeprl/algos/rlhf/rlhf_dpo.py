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

from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))
from transformers import GenerationConfig, PreTrainedTokenizer

from sheeprl.algos.rlhf.args import DPOArgs, GenerationArgs, ModelArgs, TextDataArgs
from sheeprl.algos.rlhf.collate import CompareCollate
from sheeprl.algos.rlhf.loss import dpo_loss
from sheeprl.algos.rlhf.metrics import DPOMetricManager, reward_accuracy
from sheeprl.algos.rlhf.models import ActorModel, CasualModel
from sheeprl.algos.rlhf.scheduler import CosineSchedulerWithWarmup
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
def evaluate(
    actor_model: CasualModel,
    ref_model: CasualModel,
    train_args: DPOArgs,
    data_args: TextDataArgs,
    val_dataloader: DataLoader,
) -> float:
    actor_model.eval()
    ref_model.eval()
    eval_counter = 0
    total_loss = 0.0
    total_acc = 0.0
    eval_iters = train_args.eval_iters
    for batch in val_dataloader:
        # TODO: Can we make this shorter so we will not repeat the same code
        # for both train and eval?
        chosen_input_ids = batch["chosen_input_ids"]  # type: ignore[index]
        chosen_attention_mask = batch["chosen_attention_mask"]  # type: ignore[index]

        chosen_targets = (
            batch["chosen_masked_targets"] if train_args.use_masked_targets else chosen_input_ids.detach().clone()
        )

        rejected_input_ids = batch["rejected_input_ids"]  # type: ignore[index]
        rejected_attention_mask = batch["rejected_attention_mask"]  # type: ignore[index]
        rejected_targets = (
            batch["rejected_masked_targets"] if train_args.use_masked_targets else rejected_input_ids.detach().clone()
        )
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
        loss, chosen_rewards, rejected_rewards = dpo_loss(
            actor_chosen_logps=actor_chosen_logprobs,
            actor_rejected_logps=actor_rejected_logprobs,
            reference_chosen_logps=ref_chosen_logprobs,
            reference_rejected_logps=ref_rejected_logprobs,
            chosen_targets=chosen_targets,
            rejected_targets=rejected_targets,
            beta=train_args.beta,
            ignore_index=data_args.ignore_index,
            reference_free=train_args.reference_free,
        )
        acc = reward_accuracy(chosen_rewards, rejected_rewards)
        total_loss += loss
        total_acc += acc
        eval_counter += 1

        if eval_iters is not None and eval_counter >= eval_iters:
            break
    average_loss = total_loss / eval_counter
    average_acc = total_acc / eval_counter
    actor_model.train()
    ref_model.train()
    return average_loss, average_acc


@torch.inference_mode()
def generate(
    model: CasualModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    example_prompt: Dict[str, torch.Tensor],
    device: torch.device,
) -> str:
    model.eval()
    generated = model.generate(
        input_ids=example_prompt["input_ids"].to(device),
        attention_mask=example_prompt["attention_mask"].to(device),
        generation_config=generation_config,
    )
    generated_text = tokenizer.decode(generated[0])
    model.train()
    return generated_text


@register_algorithm()
def main():
    # Retrieve arguments
    parser = HfArgumentParser([DPOArgs, ModelArgs, GenerationArgs])
    dataclasses = parser.parse_args_into_dataclasses()
    train_args: DPOArgs = dataclasses[0]
    model_args: ModelArgs = dataclasses[1]
    gen_args: GenerationArgs = dataclasses[2]

    data_args_path = Path(train_args.data_dir) / "args.json"
    data_args = TextDataArgs.from_json(str(data_args_path))

    # Setup Fabric
    fabric = L.Fabric(accelerator="auto")
    fabric.launch()
    fabric.seed_everything(train_args.seed + fabric.global_rank)
    fabric.print(f"Fabric Rank: {fabric.global_rank}")

    # Argument Validation
    if not model_args.casual:
        fabric.print("Model must be casual for supervised-finetuning. Setting it automatically.")
        model_args.casual = True

    # Setup Metrics
    metrics = DPOMetricManager(log_interval=train_args.log_interval).to(fabric.device)

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

    # Load Checkpoint path
    sft_exp_args = load_args_from_json(experiment_dir=train_args.sft_experiment_dir)
    ModelArgs(**sft_exp_args["model_args"])
    sft_checkpoint_path = get_last_checkpoint_path(train_args.sft_experiment_dir)

    # Setup Tokenizer
    tokenizer = prepare_tokenizer(model_args.model_name)

    # Setup Reference Model
    ref_model = ActorModel.from_checkpoint(
        device=fabric.device,
        model_args=model_args,
        freeze=True,
        path=sft_checkpoint_path,
    )
    ref_model = ref_model.to(fabric.device)
    trainable_parameter_summary(ref_model, show_names=False, fabric=fabric)

    # Setup Actor Model
    actor_model = ActorModel.from_checkpoint(
        device=fabric.device,
        model_args=model_args,
        freeze=True,
        path=sft_checkpoint_path,
    )
    setup_finetuning(fabric, actor_model, model_args)
    actor_model = actor_model.to(fabric.device)
    trainable_parameter_summary(actor_model, show_names=False, fabric=fabric)

    # Setup Generation Config
    generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_args=model_args,
        gen_args=gen_args,
        fabric=fabric,
    )

    # Setup Dataloaders
    collator = CompareCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)
    train_data = torch.load(Path(train_args.data_dir) / "finetune_train.pt")
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=train_args.micro_batch_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    val_data = torch.load(Path(train_args.data_dir) / "finetune_validation.pt")
    val_dataloader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=train_args.micro_batch_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    example_prompt = torch.load(Path(train_args.data_dir) / "example_prompt.pt")

    # Setup Optimizer Scheduler
    trainable_params, _, _ = prepare_optimizer_parameters(actor_model, train_args.weight_decay)
    optimizer_cls: Union[torch.optim.Adam, torch.optim.AdamW] = getattr(torch.optim, train_args.optimizer)
    optimizer = optimizer_cls(
        trainable_params,
        lr=train_args.learning_rate,
        eps=train_args.optimizer_eps,
        betas=(train_args.optimizer_beta1, train_args.optimizer_beta2),
    )
    num_training_steps = train_args.epochs * len(train_dataloader)
    lr_scheduler = CosineSchedulerWithWarmup(
        learning_rate=train_args.learning_rate,
        warmup_steps=train_args.lr_warmup_steps,
        lr_decay_steps=num_training_steps,
    )
    actor_model, optimizer = fabric.setup(actor_model, optimizer)

    gen_text = generate(
        model=actor_model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        example_prompt=example_prompt,
        device=fabric.device,
    )
    log_text(fabric, gen_text, "info/example_sample", step=0)
    iterator = tqdm(range(num_training_steps), disable=not fabric.is_global_zero)

    data_iterator = iter(train_dataloader)
    for k in iterator:
        # Setup counters and data
        if k % len(train_dataloader) == 0:
            data_iterator = iter(train_dataloader)
        is_accumulating = (k) % train_args.gradient_accumulation_steps != 0
        last_step = k == num_training_steps - 1

        # Setup learning rate
        lr = lr_scheduler.get_lr(it=k)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        metrics.info_lr.update(lr)

        # Setup batch data
        batch = next(data_iterator)
        chosen_input_ids = batch["chosen_input_ids"]  # type: ignore[index]
        chosen_attention_mask = batch["chosen_attention_mask"]  # type: ignore[index]
        chosen_targets = (
            batch["chosen_masked_targets"] if train_args.use_masked_targets else chosen_input_ids.detach().clone()
        )

        rejected_input_ids = batch["rejected_input_ids"]  # type: ignore[index]
        rejected_attention_mask = batch["rejected_attention_mask"]  # type: ignore[index]
        rejected_targets = (
            batch["rejected_masked_targets"] if train_args.use_masked_targets else rejected_input_ids.detach().clone()
        )

        # Forward and Backward Pass
        t0 = time.time()
        with torch.inference_mode():
            ref_chosen_logprobs = ref_model(
                input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False
            )
            ref_rejected_logprobs = ref_model(
                input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False
            )
        with fabric.no_backward_sync(actor_model, enabled=is_accumulating):
            actor_chosen_logprobs = actor_model(
                input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False
            )
            actor_rejected_logprobs = actor_model(
                input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False
            )
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                actor_chosen_logps=actor_chosen_logprobs,
                actor_rejected_logps=actor_rejected_logprobs,
                reference_chosen_logps=ref_chosen_logprobs,
                reference_rejected_logps=ref_rejected_logprobs,
                chosen_targets=chosen_targets,
                rejected_targets=rejected_targets,
                beta=train_args.beta,
                ignore_index=data_args.ignore_index,
                reference_free=train_args.reference_free,
            )
            fabric.backward(loss / train_args.gradient_accumulation_steps)

        dt = time.time() - t0
        if not is_accumulating:
            metrics.info_grad_norm.update(compute_grad_norm(actor_model))
            fabric.clip_gradients(
                actor_model, optimizer, max_norm=train_args.gradient_clip_val, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            metrics.info_time.update(dt)
            metrics.train_loss.update(loss.item())
            metrics.info_choosen_reward.update(chosen_rewards.mean())
            metrics.info_rejected_reward.update(rejected_rewards.mean())
            metrics.info_reward_margin.update(chosen_rewards.mean() - rejected_rewards.mean())

            train_acc = reward_accuracy(chosen_rewards, rejected_rewards)
            metrics.train_acc.update(train_acc)

        if k > 0 and (k % train_args.eval_interval == 0 or last_step):
            val_loss, val_acc = evaluate(
                actor_model=actor_model,
                ref_model=ref_model,
                train_args=train_args,
                data_args=data_args,
                val_dataloader=val_dataloader,
            )
            # we don't want to take average of different val losses
            # we already computed average inside evaluate function
            with torch.no_grad():
                metrics.val_loss.reset()
                metrics.val_loss.update(val_loss)
                metrics.val_acc.reset()
                metrics.val_acc.update(val_acc)

            if fabric.is_global_zero:
                gen_text = generate(
                    model=actor_model.module,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    example_prompt=example_prompt,
                    device=fabric.device,
                )
                log_text(fabric, gen_text, "info/example_sample", step=k)
        fabric.barrier()
        if k > 0 and (k % train_args.log_interval == 0 or last_step):
            computed_metrics = metrics.compute_all()
            metrics.log_all(fabric=fabric, step=k, metrics_dict=computed_metrics)

            if not iterator.disable:
                description = f"iter {k}, time: {dt*1000:.2f}ms"
                for metric_name, metric_value in computed_metrics.items():
                    if metric_name.startswith("info/"):
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
