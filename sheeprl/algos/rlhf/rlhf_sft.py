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

from sheeprl.algos.rlhf.args import GenerationArgs, ModelArgs, SFTArgs, TextDataArgs
from sheeprl.algos.rlhf.data import SFTCollate
from sheeprl.algos.rlhf.loss import finetune_loss
from sheeprl.algos.rlhf.metrics import SFTMetricManager
from sheeprl.algos.rlhf.models import ActorModel, CasualModel
from sheeprl.algos.rlhf.scheduler import CosineSchedulerWithWarmup
from sheeprl.algos.rlhf.utils import (
    compute_grad_norm,
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
    model: CasualModel,
    train_args: SFTArgs,
    data_args: TextDataArgs,
    val_dataloader: DataLoader,
) -> float:
    model.eval()
    eval_counter = 0
    total_loss = 0.0
    eval_iters = train_args.eval_iters
    for batch in val_dataloader:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        targets = batch["targets"] if train_args.use_targets else batch["input_ids"].detach().clone()
        loss = finetune_loss(
            outputs=outputs,
            targets=targets,
            ignore_index=data_args.ignore_index,
            label_smoothing=train_args.label_smoothing,
        )
        total_loss += loss
        eval_counter += 1

        if eval_iters is not None and eval_counter >= eval_iters:
            break
    average_loss = total_loss / eval_counter
    model.train()
    return average_loss


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
    parser = HfArgumentParser([SFTArgs, ModelArgs, GenerationArgs])
    dataclasses = parser.parse_args_into_dataclasses()
    train_args: SFTArgs = dataclasses[0]
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
    metrics = SFTMetricManager(log_interval=train_args.log_interval).to(fabric.device)

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
    # Setup Model
    model = CasualModel.from_checkpoint(device=fabric.device, model_args=model_args, freeze=True)
    setup_finetuning(fabric, model, model_args)
    model = model.to(fabric.device)

    trainable_parameter_summary(model, show_names=False, fabric=fabric)

    # Setup Generation Config
    generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_args=model_args,
        gen_args=gen_args,
        fabric=fabric,
    )

    # Setup Dataloaders
    collator = SFTCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)
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
    trainable_params, _, _ = prepare_optimizer_parameters(model, train_args.weight_decay)
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
    model, optimizer = fabric.setup(model, optimizer)

    gen_text = generate(
        model=model,
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
        input_ids = batch["input_ids"]  # type: ignore[index]
        attention_mask = batch["attention_mask"]  # type: ignore[index]
        targets = batch["targets"] if train_args.use_targets else input_ids.detach().clone()  # type: ignore[index]

        num_tokens = input_ids.numel()
        padding_pct = 100 * (attention_mask == 0).sum().item() / num_tokens

        # Forward and Backward Pass
        t0 = time.time()
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            loss = finetune_loss(
                outputs=outputs,
                targets=targets,
                ignore_index=data_args.ignore_index,
                label_smoothing=train_args.label_smoothing,
            )
            fabric.backward(loss / train_args.gradient_accumulation_steps)

        dt = time.time() - t0
        if not is_accumulating:
            metrics.info_grad_norm.update(compute_grad_norm(model))
            fabric.clip_gradients(model, optimizer, max_norm=train_args.gradient_clip_val, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            metrics.info_time.update(dt)
            metrics.train_loss.update(loss.item())
            metrics.info_tokens_per_seconds.update(num_tokens / dt)
            metrics.info_padding_percentage.update(padding_pct)

        if k > 0 and (k % train_args.eval_interval == 0 or last_step):
            val_loss = evaluate(
                model=model,
                train_args=train_args,
                data_args=data_args,
                val_dataloader=val_dataloader,
            )
            # we don't want to take average of different val losses
            # we already computed average inside evaluate function
            with torch.no_grad():
                metrics.val_loss.reset()
                metrics.val_loss.update(val_loss)

            if fabric.is_global_zero:
                gen_text = generate(
                    model=model.module,
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
            checkpoint_model: ActorModel = model.module
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
