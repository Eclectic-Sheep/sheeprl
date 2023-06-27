import os
from pathlib import Path
import sys
import time
from dataclasses import asdict
from typing import Callable

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))
from transformers import AutoTokenizer, GenerationConfig

from sheeprl.algos.rlhf.args import OPT, GenerationArgs, ModelArgs, RMArgs, TextDataArgs
from sheeprl.algos.rlhf.data import RMCollate
from sheeprl.algos.rlhf.loss import load_reward_loss
from sheeprl.algos.rlhf.models import CriticModel
from sheeprl.algos.rlhf.scheduler import CosineSchedulerWithWarmup
from sheeprl.algos.rlhf.utils import (
    prepare_optimizer_parameters,
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
def evaluate(model: CriticModel, test_dataloader: DataLoader, loss: Callable, pad_token_id: int, eval_iters: int):
    model.eval()
    eval_counter = 0
    average_acc = 0
    average_loss = 0
    for batch in test_dataloader:
        chosen_rewards = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"])
        rejected_rewards = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])
        test_loss, choosen_last_rewards, rejected_last_rewards = loss(
            chosen=batch["chosen_input_ids"],
            rejected=batch["rejected_input_ids"],
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            pad_token_id=pad_token_id,
        )
        average_loss += test_loss.detach()
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
    else:
        train_args = RMArgs(data_dir="data/Dahoas/static-hh")
        model_args = OPT()
        gen_args = GenerationArgs()

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

    # Setup Reward Loss

    reward_loss = load_reward_loss(train_args.loss_type)

    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Setup Model
    model = CriticModel.from_checkpoint(device=fabric.device, model_args=model_args, freeze=True)
    setup_finetuning(fabric=fabric, model=model, model_args=model_args)
    model = model.to(fabric.device)

    trainable_parameter_summary(model, show_names=False, fabric=fabric)

    # Setup Generation Config
    try:
        generation_config = GenerationConfig.from_pretrained(model_args.model_name, **asdict(gen_args))
    except EnvironmentError:
        # If the model does not have `generation_config.json` file, we create from scratch
        fabric.print("`generation_config.json` not found, creating `GenerationConfig` from scratch")
        generation_config = GenerationConfig(**asdict(gen_args))
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.bos_token_id = tokenizer.bos_token_id

    # Setup Dataloaders
    collator = RMCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)
    train_data = torch.load(Path(train_args.data_dir) / f"rm_train.pt")
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=train_args.micro_batch_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    test_data = torch.load(Path(train_args.data_dir) / f"rm_test.pt")
    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=train_args.micro_batch_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    # Setup Optimizer Scheduler
    trainable_params, _, _ = prepare_optimizer_parameters(model, train_args.weight_decay)
    optimizer = torch.optim.AdamW(trainable_params, lr=train_args.learning_rate)
    num_training_steps = train_args.epochs * len(train_dataloader)
    lr_scheduler = CosineSchedulerWithWarmup(
        learning_rate=train_args.learning_rate,
        warmup_steps=train_args.lr_warmup_steps,
        lr_decay_steps=num_training_steps,
    )
    model, optimizer = fabric.setup(model, optimizer)

    iterator = tqdm(range(num_training_steps)) if fabric.is_global_zero else range(num_training_steps)
    test_loss = None
    test_acc = None
    data_iterator = None
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
        dt = time.time() - t0
        train_acc = accuracy(choosen_last_rewards, rejected_last_rewards)
        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=train_args.gradient_clip_val, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if k > 0 and (k % train_args.eval_interval == 0 or last_step):
            test_loss, test_acc = evaluate(
                model=model,
                test_dataloader=test_dataloader,
                loss=reward_loss,
                pad_token_id=tokenizer.pad_token_id,
                eval_iters=train_args.eval_iters,
            )
            fabric.log("test/loss", test_loss, step=k)
            fabric.log("test/acc", test_acc, step=k)
        if k % train_args.log_interval == 0 or last_step:
            fabric.log("train/loss", loss.item(), step=k)
            fabric.log("train/chosen_reward", chosen_rewards.mean(), step=k)
            fabric.log("train/rejected_reward", rejected_rewards.mean(), step=k)
            fabric.log("train/accuracy", train_acc, step=k)
            fabric.log("train/lr", lr, step=k)
            if isinstance(iterator, tqdm):
                description = f"iter {k}, train/loss {loss.item():.4f},  time: {dt*1000:.2f}ms"
                description += f", train/acc {train_acc:.2f}"
                if test_loss is not None:
                    description += f", test/loss {test_loss:.2f}"
                if test_acc is not None:
                    description += f", test/acc {test_acc:.2f}"
                iterator.set_description(description)

        if k > 0 and (k % train_args.save_interval == 0 or last_step):
            checkpoint_model: CriticModel = model.module
            checkpoint_model.save_checkpoint(
                fabric=fabric,
                train_args=train_args,
                model_args=model_args,
                step=k,
            )
    fabric.print("Experiment is saved in", train_args.experiment_dir)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    main()
