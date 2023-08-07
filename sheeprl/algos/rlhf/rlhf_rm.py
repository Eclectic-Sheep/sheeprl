import os
import sys
import time
from pathlib import Path
from typing import Callable, Union

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from sheeprl.algos.rlhf.metrics import RMMetricManager
from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))


from sheeprl.algos.rlhf.args import GenerationArgs, ModelArgs, RMArgs, TextDataArgs
from sheeprl.algos.rlhf.data import RMCollate
from sheeprl.algos.rlhf.loss import load_reward_loss
from sheeprl.algos.rlhf.models import CriticModel, RewardModel
from sheeprl.algos.rlhf.scheduler import CosineSchedulerWithWarmup
from sheeprl.algos.rlhf.utils import (
    compute_grad_norm,
    get_last_checkpoint_path,
    load_args_from_json,
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
def accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor):
    tp = torch.count_nonzero(chosen_rewards > rejected_rewards)
    total = chosen_rewards.shape[0]
    acc = tp / total
    return acc


@torch.inference_mode()
def evaluate(model: RewardModel, val_dataloader: DataLoader, loss: Callable, pad_token_id: int, eval_iters: int):
    model.eval()
    eval_counter = 0
    average_acc = 0
    average_loss = 0
    for batch in val_dataloader:
        chosen_rewards = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"])
        rejected_rewards = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])
        val_loss, choosen_last_rewards, rejected_last_rewards = loss(
            chosen=batch["chosen_input_ids"],
            rejected=batch["rejected_input_ids"],
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            pad_token_id=pad_token_id,
        )
        average_loss += val_loss.detach()
        acc = accuracy(choosen_last_rewards, rejected_last_rewards)
        average_acc += acc
        eval_counter += 1
        if eval_iters is not None and eval_counter >= eval_iters:
            break
    average_acc /= eval_counter
    average_loss /= eval_counter
    model.train()
    return (
        average_loss,
        average_acc,
    )


@register_algorithm()
def main():
    # Retrieve arguments
    if len(sys.argv) > 1:
        parser = HfArgumentParser([RMArgs, ModelArgs, GenerationArgs])
        dataclasses = parser.parse_args_into_dataclasses()
        train_args: RMArgs = dataclasses[0]
        model_args: ModelArgs = dataclasses[1]
        gen_args: GenerationArgs = dataclasses[2]

    data_args_path = Path(train_args.data_dir) / "args.json"
    data_args = TextDataArgs.from_json(str(data_args_path))

    # Setup Fabric
    fabric = L.Fabric(accelerator="auto")
    fabric.launch()
    fabric.seed_everything(train_args.seed + fabric.global_rank)
    fabric.print(f"Fabric Rank: {fabric.global_rank}")
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

    # Setup Metrics

    metrics = RMMetricManager(log_interval=train_args.log_interval).to(fabric.device)

    # Setup Reward Loss

    reward_loss = load_reward_loss(train_args.loss_type)

    # Setup Tokenizer
    tokenizer = prepare_tokenizer(model_args.model_name)

    # Setup Model
    sft_experiment_dir = train_args.sft_experiment_dir if train_args.sft_experiment_dir is not None else None
    sft_checkpoint_path = None
    if sft_experiment_dir is not None:
        fabric.print(f"Loading finetuned transformer from {sft_experiment_dir}")
        sft_exp_args = load_args_from_json(experiment_dir=sft_experiment_dir)
        model_args = ModelArgs(**sft_exp_args["model_args"])
        sft_checkpoint_path = get_last_checkpoint_path(sft_experiment_dir)
    model = CriticModel.from_checkpoint(
        device=fabric.device, model_args=model_args, freeze=True, path=sft_checkpoint_path
    )
    setup_finetuning(fabric=fabric, model=model, model_args=model_args)
    model = model.to(fabric.device)

    trainable_parameter_summary(model, show_names=False, fabric=fabric)

    # Setup Dataloaders
    collator = RMCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)
    train_data = torch.load(Path(train_args.data_dir) / f"preference_train.pt")
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=train_args.micro_batch_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    val_data = torch.load(Path(train_args.data_dir) / f"preference_validation.pt")
    val_dataloader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=train_args.micro_batch_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

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

    iterator = tqdm(range(num_training_steps), disable=not fabric.is_global_zero)
    data_iterator = iter(train_dataloader)
    for k in iterator:
        # Setup counters and data
        if k % len(train_dataloader) == 0 or data_iterator is None:
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
        chosen = batch["chosen_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        rejected = batch["rejected_input_ids"]
        rejected_mask = batch["rejected_attention_mask"]

        t0 = time.time()

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            chosen_rewards = model(input_ids=chosen, attention_mask=chosen_mask, use_cache=False)
            rejected_rewards = model(input_ids=rejected, attention_mask=rejected_mask, use_cache=False)
            loss, choosen_last_rewards, rejected_last_rewards = reward_loss(
                chosen=chosen,
                rejected=rejected,
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
                pad_token_id=tokenizer.pad_token_id,
            )
            fabric.backward(loss / train_args.gradient_accumulation_steps)
            # DDP + gradient accumulation does not work with GPT2
            # https://github.com/huggingface/transformers/issues/22994

        dt = time.time() - t0
        if not is_accumulating:
            metrics.info_grad_norm.update(compute_grad_norm(model))
            fabric.clip_gradients(model, optimizer, max_norm=train_args.gradient_clip_val, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            metrics.info_time.update(dt)
            metrics.train_loss.update(loss.item())
            metrics.info_choosen_reward.update(choosen_last_rewards.mean())
            metrics.info_rejected_reward.update(rejected_last_rewards.mean())

            train_acc = accuracy(choosen_last_rewards, rejected_last_rewards)
            metrics.train_acc.update(train_acc)

        if k > 0 and (k % train_args.eval_interval == 0 or last_step):
            val_loss, val_acc = evaluate(
                model=model,
                val_dataloader=val_dataloader,
                loss=reward_loss,
                pad_token_id=tokenizer.pad_token_id,
                eval_iters=train_args.eval_iters,
            )
            with torch.no_grad():
                metrics.val_loss.reset()
                metrics.val_acc.reset()
                metrics.val_loss.update(val_loss)
                metrics.val_acc.update(val_acc)
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
            checkpoint_model: CriticModel = model.module
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
