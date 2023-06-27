import os
from pathlib import Path
import sys
import time
from dataclasses import asdict
from typing import Dict, Optional

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizer

from sheeprl.algos.rlhf.args import OPT, GenerationArgs, ModelArgs, SFTArgs, TextDataArgs
from sheeprl.algos.rlhf.data import SFTCollate
from sheeprl.algos.rlhf.loss import finetune_loss
from sheeprl.algos.rlhf.models import ActorModel, CasualModel
from sheeprl.algos.rlhf.scheduler import CosineSchedulerWithWarmup
from sheeprl.algos.rlhf.utils import (
    log_text,
    prepare_optimizer_parameters,
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
    test_dataloader: DataLoader,
):
    model.eval()
    eval_counter = 0
    total_loss = 0.0
    eval_iters = train_args.eval_iters
    for batch in test_dataloader:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        targets = batch["targets"] if train_args.use_targets else batch["input_ids"]
        loss = finetune_loss(
            outputs=outputs,
            targets=targets,
            ignore_index=data_args.ignore_index,
            label_smoothing=train_args.label_smoothing,
        )
        total_loss += loss.item()
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
    if len(sys.argv) > 1:
        parser = HfArgumentParser([SFTArgs, ModelArgs, GenerationArgs])
        dataclasses = parser.parse_args_into_dataclasses()
        train_args: SFTArgs = dataclasses[0]
        model_args: ModelArgs = dataclasses[1]
        gen_args: GenerationArgs = dataclasses[2]
    else:
        train_args = SFTArgs(data_dir="data/Dahoas/static-hh")
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

    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Setup Model
    model = CasualModel.from_checkpoint(device=fabric.device, model_args=model_args, freeze=True)
    setup_finetuning(fabric, model, model_args)
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
    collator = SFTCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)
    train_data = torch.load(Path(train_args.data_dir) / f"sft_train.pt")
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=train_args.micro_batch_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    test_data = torch.load(Path(train_args.data_dir) / f"sft_test.pt")
    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=train_args.micro_batch_size,
        collate_fn=collator,
        num_workers=train_args.num_workers,
    )
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    example_prompt = torch.load(Path(train_args.data_dir) / f"example_prompt.pt")

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

    gen_text = generate(
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        example_prompt=example_prompt,
        device=fabric.device,
    )
    log_text(fabric, gen_text, "test/example", step=0)
    iterator = tqdm(range(num_training_steps)) if fabric.is_global_zero else range(num_training_steps)
    test_loss = None
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
        input_ids = batch["input_ids"]  # type: ignore[index]
        attention_mask = batch["attention_mask"]  # type: ignore[index]
        targets = batch["targets"] if train_args.use_targets else input_ids  # type: ignore[index]

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
            fabric.clip_gradients(model, optimizer, max_norm=train_args.gradient_clip_val, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if k > 0 and (k % train_args.eval_interval == 0 or last_step):
            test_loss = evaluate(
                model=model,
                train_args=train_args,
                data_args=data_args,
                test_dataloader=test_dataloader,
            )
            gen_text = generate(
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                example_prompt=example_prompt,
                device=fabric.device,
            )
            log_text(fabric, gen_text, "test/example", step=k)
            fabric.log("test/loss", test_loss, step=k)
        if k % train_args.log_interval == 0 or last_step:
            fabric.log("train/time", dt, step=k)
            fabric.log("train/lr", lr, step=k)
            fabric.log("train/loss", loss.item(), step=k)
            if isinstance(iterator, tqdm):
                description = f"iter {k}, train/loss {loss.item():.4f}, time: {dt*1000:.2f}ms"
                if test_loss is not None:
                    description += f", test/loss {test_loss:.2f}"
                iterator.set_description(description)
        if k > 0 and (k % train_args.save_interval == 0 or last_step):
            checkpoint_model: ActorModel = model.module
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
