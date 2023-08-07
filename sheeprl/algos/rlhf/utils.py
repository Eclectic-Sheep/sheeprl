import json
import os
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import lightning
from lightning.fabric.fabric import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizer

from sheeprl.algos.rlhf.args import GenerationArgs, ModelArgs, TextDataArgs, TrainArgs
from sheeprl.algos.rlhf.lora_utils import add_lora
from sheeprl.algos.rlhf.models import CasualModel, CriticModel, RewardModel


@rank_zero_only
def log_text(fabric: lightning.Fabric, text: str, name: str, step: int):
    if fabric.logger is not None:
        if isinstance(fabric.logger, lightning.fabric.loggers.tensorboard.TensorBoardLogger):
            fabric.logger.experiment.add_text(name, text, step)
        else:
            warnings.warn(f"Logging text is not supported for {type(fabric.logger)}")


def trainable_parameter_summary(
    model: torch.nn.Module, show_names: bool = False, fabric: Optional[lightning.Fabric] = None
):
    print_fn = fabric.print if fabric is not None else print
    trainable = {"int8": 0, "bf16": 0, "fp16": 0, "fp32": 0, "other": 0}
    non_trainable = {"int8": 0, "bf16": 0, "fp16": 0, "fp32": 0, "other": 0}
    param_count = {"trainable": trainable, "non_trainable": non_trainable}
    trainable_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            dict_name = "trainable"
            trainable_param_names.append(name)
        else:
            dict_name = "non_trainable"
        num_params = param.numel()
        if param.dtype == torch.int8:
            param_count[dict_name]["int8"] += num_params
        elif param.dtype == torch.bfloat16:
            param_count[dict_name]["bf16"] += num_params
        elif param.dtype == torch.float16:
            param_count[dict_name]["fp16"] += num_params
        elif param.dtype == torch.float32:
            param_count[dict_name]["fp32"] += num_params
        else:
            param_count[dict_name]["other"] += num_params

    if show_names:
        print_fn("Trainable parameter names:")
        print_fn(trainable_param_names)
    print_fn("Parameter dtypes:")
    print_fn(f"Trainable {trainable}")
    print_fn(f"Non-Trainable {non_trainable}")
    total_params = sum([sum(v.values()) for v in param_count.values()])
    total_trainable_params = sum([v for k, v in param_count["trainable"].items()])
    print_fn(
        f"Total: {total_params}, Trainable: {total_trainable_params}, Percentage: {total_trainable_params/total_params:.2%}"
    )


def prepare_optimizer_parameters(model: torch.nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Taken from  https://github.com/karpathy/nanoGPT"""
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    return optim_groups, num_decay_params, num_nodecay_params


def load_args_from_json(experiment_dir: str):
    with open(os.path.join(experiment_dir, "args.json"), "r") as f:
        args_dict = json.load(f)
    return args_dict


def save_args_to_json(
    train_args: TrainArgs,
    model_args: ModelArgs,
    data_args: TextDataArgs,
    gen_args: GenerationArgs,
):
    args = {}
    args.update(train_args.to_dict())
    args.update(model_args.to_dict())
    args.update(data_args.to_dict())
    args.update(gen_args.to_dict())
    with open(os.path.join(train_args.experiment_dir, "args.json"), "w") as f:
        json.dump(args, f, indent=4)
    return args


def get_last_checkpoint_path(experiment_dir: str):
    model_dir = os.path.join(experiment_dir, "model")
    checkpoints = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[-2].split("-")[-1]))
    return checkpoints[-1]


def setup_finetuning(fabric: lightning.Fabric, model: torch.nn.Module, model_args: ModelArgs):
    finetune_mode = model_args.finetune_mode
    if finetune_mode == "all":
        fabric.print("Using all layers parameters for finetuning")
        for param in model.parameters():
            param.requires_grad = True

    elif finetune_mode == "lora":
        fabric.print("Adding LORA parameters for finetuning")
        add_lora(model, model_args)
        if isinstance(model, CriticModel):
            for param in model.head.parameters():
                param.requires_grad = True

    elif finetune_mode == "last_layer":
        fabric.print("Using only head layer parameters for finetuning")
        for name, param in model.named_parameters():
            param.requires_grad = False
        if isinstance(model, CasualModel):
            if hasattr(model.model, "get_input_embeddings"):
                model.model.get_input_embeddings().weight.requires_grad = True
            else:
                raise ValueError("No input embeddings found in model for finetuning. Cannot use head mode.")
        elif isinstance(model, CriticModel):
            for param in model.head.parameters():
                param.requires_grad = True
            if isinstance(model, RewardModel):
                model.bias.requires_grad = True
                model.gain.requires_grad = True
        else:
            raise ValueError(f"Unknown model type {type(model)}")

    else:
        raise ValueError(f"Unknown finetuning mode {finetune_mode}")


def compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().cpu().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def prepare_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special_tokens = tokenizer.special_tokens_map
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "bos_token" not in special_tokens.keys() or special_tokens["bos_token"] == special_tokens["eos_token"]:
        # we don't resize the tokenizer here because we want to keep the original vocab size
        # However, we need something to represent the start of the text
        # we use <|startoftext|> from gptj
        # or if they are the same, we use another word to represent the start of the text
        # this is useful for gpt2 models where bos_token and eos_token are the same
        tokenizer.bos_token = "<|startoftext|>"
    return tokenizer


def prepare_generation_config(
    tokenizer: PreTrainedTokenizer, model_args: ModelArgs, gen_args: GenerationArgs, fabric: lightning.Fabric
) -> Dict[str, Any]:
    try:
        generation_config = GenerationConfig.from_pretrained(model_args.model_name, **asdict(gen_args))
    except EnvironmentError:
        # If the model does not have `generation_config.json` file, we create from scratch
        fabric.print("`generation_config.json` not found, creating `GenerationConfig` from scratch")
        generation_config = GenerationConfig(**asdict(gen_args))
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.bos_token_id = tokenizer.bos_token_id
    return generation_config
