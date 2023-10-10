import os
from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning_utilities.core.rank_zero import rank_zero_only
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel

from sheeprl.algos.rlhf.config_store.model import FINETUNE_MODE, HuggingFaceConfig, ModelConfig
from sheeprl.algos.rlhf.lora_utils import add_lora, get_lora_state_dict, merge_lora


def load_hf_transformer(model_cfg: ModelConfig) -> PreTrainedModel:
    model_cls = AutoModel if not model_cfg.casual else AutoModelForCausalLM
    hf_cfg: HuggingFaceConfig = model_cfg.library_cfg
    auto_config = AutoConfig.from_pretrained(model_cfg.name, trust_remote_code=hf_cfg.trust_remote_code)
    if hasattr(auto_config, "dropout"):
        auto_config.dropout = 0.0 if model_cfg.disable_dropout else auto_config.dropout
    auto_config.use_cache = hf_cfg.use_cache
    auto_config.torch_dtype = torch.get_default_dtype()
    model = model_cls.from_pretrained(
        model_cfg.name,
        trust_remote_code=hf_cfg.trust_remote_code,
        load_in_8bit=hf_cfg.load_in_8bit,
        low_cpu_mem_usage=hf_cfg.low_cpu_mem_usage,
        config=auto_config,
    )
    if model_cfg.freeze_transformer:
        for param in model.parameters():
            param.requires_grad = False
    return model


class CasualModel(torch.nn.Module):
    def __init__(self, model, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.model = model

    def forward(self, **kwargs):
        if self.training and not self.model_cfg.use_attention_mask:
            kwargs.pop("attention_mask")
        return self.model(**kwargs).logits

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        device: torch.device,
        model_cfg: ModelConfig,
        path: Optional[str] = None,
        freeze: bool = False,
    ):
        model = load_hf_transformer(model_cfg)
        model = cls(model=model, model_cfg=model_cfg).to(device)
        if path is not None:
            sd = torch.load(path, map_location=device)
            if model_cfg.finetune_mode == FINETUNE_MODE.LAST_LAYER:
                embedding_weights = sd["last_layer_weights"]
                model.model.set_input_embeddings(embedding_weights)
            elif model_cfg.finetune_mode == FINETUNE_MODE.LORA:
                add_lora(model, lora_cfg=model_cfg.lora_cfg, device=device)
                model.load_state_dict(sd, strict=False)
                merge_lora(model)
            elif model_cfg.finetune_mode == FINETUNE_MODE.ALL:
                model.load_state_dict(sd)
            else:
                raise ValueError(f"Unknown finetune mode {model_cfg.finetune_mode}")
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        return model

    @rank_zero_only
    def save_checkpoint(self, fabric: L.Fabric, experiment_dir: str, model_cfg: ModelConfig, step):
        output_file = os.path.join(experiment_dir, "model", f"checkpoint-{step}.pt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if model_cfg.finetune_mode == FINETUNE_MODE.LAST_LAYER:
            embedding_weights = self.model.get_input_embeddings().weight
            sd = {"last_layer_weights": embedding_weights}
        if model_cfg.finetune_mode == FINETUNE_MODE.LORA:
            sd = get_lora_state_dict(self)
        else:
            sd = self.state_dict()
        fabric.save(output_file, sd)


class ActorModel(CasualModel):
    def __init__(self, model, model_cfg: ModelConfig):
        super().__init__(model=model, model_cfg=model_cfg)

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        if self.training and not self.model_cfg.use_attention_mask:
            kwargs.pop("attention_mask")
        out = self.model(**kwargs)
        # Model predicts next token log probability here.
        actor_log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)
        selected_actor_log_probs = actor_log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        return selected_actor_log_probs.squeeze(-1)


class CriticModel(torch.nn.Module):
    def __init__(self, model, model_cfg: ModelConfig, embedding_dim: int, transformer_name: Optional[str] = None):
        super().__init__()
        self.model_cfg = model_cfg
        if transformer_name is None:
            # If any transformer name is provided, we search for common attribute names usually
            # avaliable inside huggingface library.
            if hasattr(model, "transformer"):
                self.transformer = model.transformer
            elif hasattr(model, "model"):
                self.transformer = model.model
            else:
                raise ValueError(
                    f"{model} Could not find transformer, searched for 'transformer' and 'model' attributes, "
                    "if your model has a different attribute name, please specify it in `transformer_name`"
                )
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
        if self.training and not self.model_cfg.use_attention_mask:
            kwargs.pop("attention_mask")
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
        model_cfg: ModelConfig,
        path: Optional[str] = None,
        freeze: bool = False,
    ):
        model = load_hf_transformer(model_cfg)
        transformer_config = model.base_model.config
        if model_cfg.embedding_dim_name is None:
            if hasattr(model, "get_input_embeddings"):
                embedding_dim = model.get_input_embeddings().weight.shape[-1]
            else:
                raise ValueError("embedding_dim_name is None and model does not have `get_input_embeddings` method")
        else:
            embedding_dim = getattr(transformer_config, model_cfg.embedding_dim_name, None)
            if embedding_dim is None:
                raise ValueError(f"`embedding_dim_name={model_cfg.embedding_dim_name}` not found in transformer_config")
        model = cls(model=model, embedding_dim=embedding_dim, transformer_name=model_cfg.transformer_name).to(device)
        if path is not None:
            sd = torch.load(path, map_location=device)
            new_sd = {}
            for k, v in sd.items():
                new_k = k.replace("model.model", "transformer")
                new_k = new_k.replace("model.", "")
                new_sd[new_k] = v
            if model_cfg.finetune_mode == FINETUNE_MODE.LORA:
                add_lora(model, lora_cfg=model_cfg.lora_cfg, device=device)
                model.load_state_dict(new_sd, strict=False)
                merge_lora(model)
            elif model_cfg.finetune_mode == FINETUNE_MODE.LAST_LAYER or model_cfg.casual:
                model.load_state_dict(new_sd, strict=False)
            else:
                model.load_state_dict(new_sd)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        return model

    @rank_zero_only
    def save_checkpoint(self, fabric: L.Fabric, experiment_dir: str, model_cfg: ModelConfig, step):
        output_file = os.path.join(experiment_dir, "model", f"checkpoint-{step}.pt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if model_cfg.finetune_mode == FINETUNE_MODE.LORA:
            sd = get_lora_state_dict(self)
            head_sd = self.get_head_state_dict()
            sd.update(head_sd)
        elif model_cfg.finetune_mode == FINETUNE_MODE.LAST_LAYER:
            sd = self.get_head_state_dict()
        else:
            sd = self.state_dict()
        fabric.save(output_file, sd)


class RewardModel(CriticModel):
    def __init__(self, model, embedding_dim, transformer_name):
        super().__init__(model, embedding_dim, transformer_name)
        self.gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self._disable_bias_gain = False

    def disable_bias_gain(self):
        self._disable_bias_gain = True

    def enable_bias_gain(self):
        self._disable_bias_gain = False

    def forward(self, **kwargs):
        value_out = super().forward(**kwargs)
        if self._disable_bias_gain:
            return value_out
        return value_out * self.gain + self.bias

    def get_head_state_dict(self):
        head_state_dict = super().get_head_state_dict()
        if not self._disable_bias_gain:
            head_state_dict.update({"gain": self.gain, "bias": self.bias})
        return head_state_dict
