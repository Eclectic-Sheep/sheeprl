import os
from typing import Any, Dict, List, Optional
import lightning
import warnings
import json
from lightning.fabric.fabric import torch

from sheeprl.algos.rlhf.args import GenerationArgs, ModelArgs, TextDataArgs, TrainArgs
from sheeprl.algos.rlhf.lora_utils import add_lora
from sheeprl.algos.rlhf.models import CasualModel, CriticModel


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
        fabric.print("Using all layer arameters for finetuning")
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
        else:
            raise ValueError(f"Unknown model type {type(model)}")

    else:
        raise ValueError(f"Unknown finetuning mode {finetune_mode}")
