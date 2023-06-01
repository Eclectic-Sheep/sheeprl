import os
from typing import Any, Dict, List
import lightning
import warnings
import json
from lightning.fabric.fabric import torch

from sheeprl.algos.rlhf.args import GenerationArgs, ModelArgs, TextDataArgs, TrainArgs


def log_text(fabric: lightning.Fabric, text: str, name: str, step: int):
    if fabric.logger is not None:
        if isinstance(fabric.logger, lightning.fabric.loggers.tensorboard.TensorBoardLogger):
            fabric.logger.experiment.add_text(name, text, step)
        else:
            warnings.warn(f"Logging text is not supported for {type(fabric.logger)}")


def trainable_parameter_summary(fabric: lightning.Fabric, model: torch.nn.Module, show_names: bool = False):
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
        fabric.print("Trainable parameter names:")
        fabric.print(trainable_param_names)
    fabric.print("Parameter dtypes:")
    fabric.print(f"Trainable {trainable}")
    fabric.print(f"Non-Trainable {non_trainable}")
    total_params = sum([sum(v.values()) for v in param_count.values()])
    total_trainable_params = sum([v for k, v in param_count["trainable"].items()])
    fabric.print(
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


def load_model_args_from_json(experiment_dir: str):
    with open(os.path.join(experiment_dir, "args.json"), "r") as f:
        args_dict = json.load(f)
    if "model_args" in args_dict:
        model_args_dict = args_dict["model_args"]
        model_args = ModelArgs(**model_args_dict)
        return model_args
    else:
        raise ValueError("No model args found in args.json")


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


def get_last_checkpoint_path(experimet_dir: str):
    model_dir = os.path.join(experimet_dir, "model")
    checkpoints = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[-2].split("-")[-1]))
    return checkpoints[-1]
