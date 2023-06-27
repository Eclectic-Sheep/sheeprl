import os
from typing import Optional

import lightning as L
from lightning_utilities.core.rank_zero import rank_zero_only
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
)

from sheeprl.algos.rlhf.args import ModelArgs, TrainArgs
from sheeprl.algos.rlhf.lora_utils import add_lora, get_lora_state_dict, merge_lora


def load_hf_transformer(model_args: ModelArgs) -> PreTrainedModel:
    model_cls = AutoModel if not model_args.casual else AutoModelForCausalLM
    model_config = AutoConfig.from_pretrained(model_args.model_name)
    model_config.dropout = 0.0 if model_args.disable_dropout else model_config.dropout
    model_config.use_cache = model_args.use_cache
    model_config.torch_dtype = torch.get_default_dtype()
    model = model_cls.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code,
        load_in_8bit=model_args.load_in_8bit,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        config=model_config,
    )
    if model_args.freeze_transformer:
        for param in model.parameters():
            param.requires_grad = False
    return model


class CasualModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model(**kwargs).logits

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        device: torch.device,
        model_args: ModelArgs,
        path: Optional[str] = None,
        freeze: bool = False,
    ):
        model_args.casual = True
        model = load_hf_transformer(model_args)
        model = cls(model=model)
        if path is not None:
            sd = torch.load(path, map_location=device)
            if model_args.finetune_mode == "last_layer":
                embedding_weights = sd["last_layer_weights"]
                model.model.set_input_embeddings(embedding_weights)
            elif model_args.finetune_mode == "lora":
                add_lora(model, model_args)
                model.load_state_dict(sd, strict=False)
                merge_lora(model)
            else:
                model.load_state_dict(sd)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        return model

    @rank_zero_only
    def save_checkpoint(self, fabric: L.Fabric, train_args: TrainArgs, model_args: ModelArgs, step):
        output_file = os.path.join(train_args.experiment_dir, "model", f"checkpoint-{step}.pt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if model_args.finetune_mode == "last_layer":
            embedding_weights = self.model.get_input_embeddings().weight
            sd = {"last_layer_weights": embedding_weights}
        if model_args.finetune_mode == "lora":
            sd = get_lora_state_dict(self)
        else:
            sd = self.state_dict()
        fabric.save(output_file, sd)


class ActorModel(CasualModel):
    def __init__(self, model):
        super().__init__(model=model)
        self.model = model

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        out = self.model(**kwargs)
        actor_log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)  # (B, T, vocab_size)
        selected_actor_log_probs = actor_log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))  # (B, T, 1)
        return selected_actor_log_probs.squeeze(-1)


class CriticModel(torch.nn.Module):
    def __init__(self, model, embedding_dim, transformer_name):
        super().__init__()
        if transformer_name is None:
            self.transformer = model
        else:
            self.transformer = getattr(model, transformer_name)
        self.head = torch.nn.Linear(embedding_dim, 1, bias=False)
        self.head.apply(self.init_normal)

    def init_normal(self, module):
        if type(module) == torch.nn.Linear:
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)[0]
        value = self.head(out)
        return value.squeeze(-1)

    def get_head_state_dict(self):
        sd = self.state_dict()
        sd = {k: v for k, v in sd.items() if "head" in k}
        return sd

    @classmethod
    def from_checkpoint(
        cls,
        device: torch.device,
        model_args: ModelArgs,
        path: Optional[str] = None,
        freeze: bool = False,
    ):
        model_args.casual = False
        model = load_hf_transformer(model_args)
        transformer_config = model.base_model.config
        embedding_dim = getattr(transformer_config, model_args.embedding_dim_name, None)
        model = cls(model=model, embedding_dim=embedding_dim, transformer_name=None)
        if path is not None:
            sd = torch.load(path, map_location=device)
            new_sd = {}
            for k, v in sd.items():
                new_k = k.replace("model.model", "transformer")
                new_sd[new_k] = v
            if model_args.finetune_mode == "lora":
                add_lora(model, model_args)
                model.load_state_dict(new_sd, strict=False)
                merge_lora(model)
            elif model_args.finetune_mode == "last_layer":
                model.load_state_dict(new_sd, strict=False)
            else:
                model.load_state_dict(new_sd)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        return model

    @rank_zero_only
    def save_checkpoint(self, fabric: L.Fabric, train_args: TrainArgs, model_args: ModelArgs, step):
        output_file = os.path.join(train_args.experiment_dir, "model", f"checkpoint-{step}.pt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if model_args.finetune_mode == "lora":
            sd = get_lora_state_dict(self)
            head_sd = self.get_head_state_dict()
            sd.update(head_sd)
        elif model_args.finetune_mode == "last_layer":
            sd = self.get_head_state_dict()
        else:
            sd = self.state_dict()
        fabric.save(output_file, sd)
