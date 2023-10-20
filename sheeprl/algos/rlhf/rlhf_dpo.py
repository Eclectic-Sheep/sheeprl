import time
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import lightning as L
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sheeprl.algos.rlhf.agent import DPOAgent
from sheeprl.utils.imports import _IS_TRANSFORMERS_AVAILABLE

if not _IS_TRANSFORMERS_AVAILABLE:
    raise ModuleNotFoundError(str(_IS_TRANSFORMERS_AVAILABLE))

from transformers import GenerationConfig, PreTrainedTokenizer

from sheeprl.algos.rlhf.collate import CompareCollate
from sheeprl.algos.rlhf.config_store.algo import DPOAlgoConfig
from sheeprl.algos.rlhf.config_store.data import DataConfig, GenConfig
from sheeprl.algos.rlhf.config_store.model import ModelConfig
from sheeprl.algos.rlhf.data import TextDataset
from sheeprl.algos.rlhf.loss import dpo_loss
from sheeprl.algos.rlhf.metrics import DPOMetricManager, reward_accuracy
from sheeprl.algos.rlhf.models import ActorModel, CasualModel
from sheeprl.algos.rlhf.scheduler import CosineSchedulerWithWarmup
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
def evaluate(
    agent: DPOAgent,
    algo_cfg: DPOAlgoConfig,
    data_cfg: DataConfig,
    val_dataloader: DataLoader,
) -> float:
    actor_model.eval()
    ref_model.eval()
    eval_counter = 0
    total_loss = 0.0
    total_acc = 0.0
    eval_iters = algo_cfg.eval_iters
    for batch in val_dataloader:
        if not algo_cfg.use_masked_targets:
            batch["chosen_targets"] = batch["chosen_input_ids"].detach().clone()
            batch["chosen_targets"] = batch["rejected_input_ids"].detach().clone()

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            batch=batch,
            agent=agent,
            beta=algo_cfg.beta,
            ignore_index=data_cfg.ignore_index,
            reference_free=algo_cfg.reference_free,
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
def main(fabric: L.Fabric, cfg: Dict[str, Any]):
    algo_cfg = DPOAlgoConfig(**cfg.algo)
    model_cfg = ModelConfig(**cfg.model)
    data_cfg = DataConfig(**cfg.data)
    gen_cfg = GenConfig(**cfg.generation)
    optim_cfg = cfg.optim
    sft_experiment_dir: Optional[str] = algo_cfg.sft_experiment_dir

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, cfg)
    experiment_dir = Path(log_dir).parent
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)

    # Setup Metrics
    metrics = DPOMetricManager(log_interval=algo_cfg.log_interval).to(fabric.device)

    # Setup Dataloaders
    data_processor = validate_dataset(fabric, data_cfg)
    dataset_path = Path(data_processor.full_path)
    tokenizer = data_processor.tokenizer

    collator = CompareCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    train_dataset = TextDataset(dataframe_path=dataset_path / "finetune_train.pkl")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=algo_cfg.micro_batch_size,
        collate_fn=collator,
        num_workers=algo_cfg.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    val_dataset = TextDataset(dataframe_path=dataset_path / "finetune_validation.pkl")
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=algo_cfg.micro_batch_size,
        collate_fn=collator,
        num_workers=algo_cfg.num_workers,
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    example_prompt = torch.load(dataset_path / "example_prompt.pt")

    with fabric.init_module(empty_init=model_cfg.fabric_empty_init):
        agent = DPOAgent(fabric=fabric, model_cfg=model_cfg, sft_experiment_dir=sft_experiment_dir)

    # Setup Generation Config
    generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        gen_cfg=gen_cfg,
        fabric=fabric,
    )

    # Setup Optimizer Scheduler
    trainable_params, _, _ = prepare_optimizer_parameters(agent.actor, optim_cfg.weight_decay)
    optimizer = hydra.utils.instantiate(
        optim_cfg,
        params=trainable_params,
        _convert_="partial",
    )
    num_training_steps = algo_cfg.epochs * len(train_dataloader)
    lr_scheduler = CosineSchedulerWithWarmup(
        lr=optim_cfg.lr,
        warmup_steps=algo_cfg.lr_warmup_steps,
        lr_decay_steps=num_training_steps,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    gen_text = generate(
        model=agent.actor.module,
        tokenizer=tokenizer,
        generation_config=generation_config,
        example_prompt=example_prompt,
        device=fabric.device,
    )
    log_text(fabric, gen_text, "info/example_sample", step=0)
    iterator = tqdm(range(num_training_steps), disable=not fabric.is_global_zero)

    data_iterator = iter(train_dataloader)
    agent.actor.train()
    for k in iterator:
        # Setup counters and data
        if k % len(train_dataloader) == 0:
            data_iterator = iter(train_dataloader)
        is_accumulating = (k) % algo_cfg.gradient_accumulation_steps != 0
        last_step = k == num_training_steps - 1

        # Setup learning rate
        lr = lr_scheduler.get_lr(it=k)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        metrics.info_lr.update(lr)

        # Setup batch data
        batch = next(data_iterator)
        if not algo_cfg.use_masked_targets:
            batch["chosen_targets"] = batch["chosen_input_ids"].detach().clone()
            batch["rejected_targets"] = batch["rejected_input_ids"].detach().clone()

        # Forward and Backward Pass
        t0 = time.time()
        with fabric.no_backward_sync(agent.actor, enabled=is_accumulating):
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                batch=batch,
                agent=agent,
                beta=algo_cfg.beta,
                ignore_index=data_cfg.ignore_index,
                reference_free=algo_cfg.reference_free,
            )
            fabric.backward(loss / algo_cfg.gradient_accumulation_steps)

        dt = time.time() - t0
        if not is_accumulating:
            metrics.info_grad_norm.update(compute_grad_norm(agent.actor))
            fabric.clip_gradients(agent.actor, optimizer, max_norm=algo_cfg.gradient_clip_val, error_if_nonfinite=True)
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

        if k > 0 and (k % algo_cfg.eval_interval == 0 or last_step):
            val_loss, val_acc = evaluate(
                agent=agent,
                algo_cfg=algo_cfg,
                data_cfg=data_cfg,
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
                    model=agent.actor.module,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    example_prompt=example_prompt,
                    device=fabric.device,
                )
                log_text(fabric, gen_text, "info/example_sample", step=k)
        fabric.barrier()
        if k > 0 and (k % algo_cfg.log_interval == 0 or last_step):
            computed_metrics = metrics.compute_all()
            metrics.log_all(fabric=fabric, step=k, metrics_dict=computed_metrics)

            if not iterator.disable:
                description = f"iter {k}, time: {dt*1000:.2f}ms"
                for metric_name, metric_value in computed_metrics.items():
                    if metric_name.startswith("info/"):
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
