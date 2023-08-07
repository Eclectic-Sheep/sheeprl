import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from sheeprl.utils.parser import Arg

### Training Arguments ###


@dataclass
class TrainArgs:
    experiment_name: str = Arg(help="Name of the experiment")
    log_dir: str = Arg(default="logs", help="Parent path to save logs")
    experiment_dir: str = Arg(
        help="Name of the experiment. This will be auto-generated during the training.", init=False
    )
    data_dir: str = Arg(default="data", help="Path to data directory")
    seed: int = Arg(default=42, help="Seed for reproducibility")
    epochs: int = Arg(default=1, help="Number of epochs for training")
    save_interval: int = Arg(default=500, help="Every save interval steps model will be saved")
    eval_interval: int = Arg(
        default=100, help="Every eval interval steps model will be evaluated or text will be generated."
    )
    log_interval: int = Arg(
        default=5,
        help="Every log interval steps metrics will be logged to selected logger and progress bar will be updated",
    )
    eval_iters: Optional[int] = Arg(
        default=None,
        help="Number of iterations to evaluate on. If None, evaluate on full validation set. Setting a smaller number makes the training script run faster with less reliable validation score to track.",
    )
    num_workers: int = Arg(default=4, help="Number of workers for dataloaders")
    mini_batch_size: int = Arg(
        default=4,
        help="Mini batch size for training. This is the expected batch size when there is no gradient accumulation applied.",
    )
    micro_batch_size: int = Arg(
        default=4,
        help="Micro batch size for training. If mini_batch_size // micro_batch_size == 1, no gradient accumulation is performed",
    )
    gradient_clip_val: float = Arg(default=1.0, help="Gradient clipping value")
    gradient_accumulation_steps: int = Arg(
        help="Number of gradient accumulation steps. It will be calculated automatically.", init=False
    )
    weight_decay: float = Arg(default=0.0, help="Weight decay value for optimizer")
    optimizer: str = Arg(
        default="AdamW", help="Optimizer to use for training. Only Adam and AdamW optimizers are supported."
    )
    optimizer_eps: float = Arg(default=1e-6, help="Epsilon value for optimizer")
    optimizer_beta1: float = Arg(default=0.9, help="Beta1 value for optimizer")
    optimizer_beta2: float = Arg(default=0.95, help="Beta2 value for optimizer")

    def __post_init__(self):
        self.gradient_accumulation_steps = int(self.mini_batch_size // self.micro_batch_size)
        timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path(self.log_dir)
        data_dir = Path(self.data_dir)
        data_dir_name = data_dir.name  # dataset name
        parent_data_dir = data_dir.parent.name  # data directory name
        self.experiment_dir = str(log_dir / self.experiment_name / parent_data_dir / data_dir_name / timestamp)

    def to_dict(self) -> dict:
        return {"train_args": asdict(self)}


@dataclass
class SFTArgs(TrainArgs):
    experiment_name: str = Arg(default="rlhf-supervised-finetuning", help="Name of the experiment")
    label_smoothing: float = Arg(
        default=0.0,
        help="Label smoothing value for cross entropy loss. When it is bigger than 0.0, it will be applied. Label smoothing helps when the model is overconfident according to Hugginface documentation.",
    )
    use_targets: bool = Arg(
        default=False,
        help="Whether to use masked targets for training. Using masked targets may lead to overfitting. If targets are used model will only compute the loss based on the masked targets such as answer on question/answer based dataset.",
    )
    learning_rate: float = Arg(default=1e-4, help="Learning rate for optimizer")
    lr_warmup_steps: int = Arg(
        default=100, help="Number of warmup steps for learning rate scheduler. Default scheduler has linear warmup."
    )


@dataclass
class RMArgs(TrainArgs):
    experiment_name: str = Arg(default="rlhf-reward-modelling", help="Name of the experiment")
    sft_experiment_dir: Optional[str] = Arg(
        default=None, help="Path to supervised finetuning experiment directory to load model from."
    )
    learning_rate: float = Arg(default=1e-4, help="Learning rate for optimizer.")
    lr_warmup_steps: int = Arg(
        default=100, help="Number of warmup steps for learning rate scheduler. Default scheduler has linear warmup."
    )
    loss_type: Literal["average", "last_token", "per_sample"] = Arg(
        default="per_sample",
        help="Loss type for reward modelling. Each loss type computes reward differently. `per_sample` loss type uses similar reward loss to DeepSpeedChat.",
    )


@dataclass
class PPOArgs(TrainArgs):
    experiment_name: str = Arg(default="rlhf-ppo", help="Name of the experiment")
    rollout_size: int = Arg(
        default=128,
        help="Rollout size for PPO. For every training iteration this number of samples will be sampled from dataset and each will be used for generating response.",
    )
    rollout_mini_batch_size: int = Arg(
        default=32,
        help="Rollout mini batch size for PPO. This number is useful when the GPU memory is not sufficient for running all generation code with single batch.",
    )
    ppo_epochs: int = Arg(
        default=1, help="Number of ppo epochs to training. `ppo_step` function will be called `ppo_epochs` times"
    )

    normalize_rewards: bool = Arg(default=True, help="Whether to whiten rewards")
    normalize_advantages: bool = Arg(default=True, help="Whether to whiten advantages")
    adaptive_kl_coeff: bool = Arg(default=False, help="Whether to use adaptively changing KL divergence coefficient")
    clip_rewards: bool = Arg(default=True, help="Whether to clip rewards")
    reward_clip_value: float = Arg(default=5.0, help="Reward clipping value")
    init_kl_coeff: float = Arg(
        default=0.1,
        help="KL divergence coefficient for comparing actor model with reference model. Higher value means more trust to reference model.",
    )
    target_kl_coeff: float = Arg(default=0.1, help="Target KL divergence coefficient")
    clip_coeff: float = Arg(default=0.2, help="Clip coefficient for PPO loss")
    vf_coeff: float = Arg(default=0.1, help="Value loss coefficient for PPO loss")
    gae_gamma: float = Arg(default=1.0, help="Discount factor for GAE(Generalized Advantage Estimation)")
    gae_lambd: float = Arg(default=0.95, help="Lambda for GAE(Generalized Advantage Estimation)")
    sft_experiment_dir: Optional[str] = Arg(
        default=None, help="Path to supervised finetuning experiment directory. Latest checkpoint will be loaded."
    )
    rm_experiment_dir: Optional[str] = Arg(
        default=None, help="Path to reward modelling experiment directory. Latest checkpoint will be loaded."
    )
    actor_learning_rate: float = Arg(default=1e-6, help="Learning rate for actor optimizer")
    critic_learning_rate: float = Arg(default=1e-6, help="Learning rate for critic optimizer")

    def __post_init__(self):
        super().__post_init__()
        if self.rollout_mini_batch_size > self.rollout_size:
            raise ValueError(
                f"Rollout mini batch size ({self.rollout_mini_batch_size}) cannot be bigger than rollout size ({self.rollout_size})"
            )
        if self.normalize_rewards and self.rollout_size < 8:
            raise ValueError(f"Rollout size ({self.rollout_size}) cannot be smaller than 8 when rewards are normalized")


#### Evaluation Arguments ####


@dataclass
class EvaluateArgs:
    experiment_dir: str = Arg(help="Path to experiment directory to load casual model from")
    seed: int = Arg(default=42, help="Seed for reproducibility")
    mini_batch_size: int = Arg(default=4, help="Mini batch size for evaluation")
    num_workers: int = Arg(default=4, help="Number of workers for data loading")
    use_pretrained: bool = Arg(
        default=False,
        help="Whether to use pretrained model for evaluation or finetuned model. It is useful to evaluate both pretrained and finetuned model to measure the performance and compare the improvement.",
    )


@dataclass
class EvaluateRewardArgs(EvaluateArgs):
    loss_type: Literal["average", "last_token", "per_sample"] = Arg(
        default="last_token", help="Loss type for reward modelling"
    )


@dataclass
class EvaluatePerplexityArgs(EvaluateArgs):
    label_smoothing: float = Arg(
        default=0.0,
        help="Label smoothing value for cross entropy loss. For evaluation, it is better to use 0.0",
    )
    use_targets: bool = Arg(
        default=True,
        help="Whether to use masked targets for training. We would like to evaluate models on possible generated responses.",
    )


### Model Arguments ###


@dataclass
class ModelArgs:
    model_name: str = Arg(help="Name of the model. It will be used to load huggingface model.")
    embedding_dim_name: Optional[str] = Arg(
        default=None,
        help="Name of the embedding dimension in the model config. If it is None, the code will try to call `get_input_embeddings` method. It is used for Critic models where we add head layer after the embedding layer.",
    )
    transformer_name: Optional[str] = Arg(
        default=None,
        help="Name of the transformer module that is loaded from huggingface. If it is provided this attribute of the transformer module will be used to retrieve the part of the model except head layer.",
    )
    casual: bool = Arg(
        default=True,
        help="Whether to use casual attention mask for transformer.If it is true the model will be loaded from `AutoModelForCausalLM`. Currently scripts are expecting this parameter is set to `true`.",
    )
    use_cache: bool = Arg(
        default=False,
        help="Whether to use cache for huggingface transformer. It is better to set this false and control where the cache is used inside the script. It is advised to use during the evaluation.",
    )
    trust_remote_code: bool = Arg(default=False, help="Whether to trust remote code for huggingface transformer.")
    load_in_8bit: bool = Arg(
        default=False,
        help="Whether to load model in 8bit precision. It is not tested to load models with this parameter set to true.",
    )
    low_cpu_mem_usage: bool = Arg(
        default=False, help="Whether to use low cpu memory usage for huggingface transformer."
    )
    freeze_transformer: bool = Arg(default=True, help="Freeze transformer weights when it is loading.")
    disable_dropout: bool = Arg(
        default=False,
        help="Whether to disable dropout layers in the model. During SFT dropout allows not to overfit to the dataset. However, Disabling dropout layers helps stabilizing training during PPO training since `log_ratio` for the first iteration becomes one. ",
    )
    finetune_mode: Literal["all", "last_layer", "lora"] = Arg(
        default="all",
        help="Whether to finetune the model. If it is `all`, all the layers will be finetuned. If it is `head`, only the head layer will be finetuned. If it is 'lora', lora approximation will be used for the model finetuning.",
    )
    lora_targets: Optional[str] = Arg(
        default=None,
        help="LoRA target layer names for the model.",
    )
    lora_rank: int = Arg(default=8, help="Rank of the LoRA approximation.")
    lora_alpha: int = Arg(default=1, help="Alpha value for LoRA approximation.")
    lora_dropout: float = Arg(default=0.0, help="Dropout rate for LoRA approximation.")

    def to_dict(self) -> dict:
        return {"model_args": asdict(self)}


@dataclass
class GPT2(ModelArgs):
    model_name: str = Arg(default="gpt2-medium", help="Name of the model. It will be used to load huggingface model.")
    embedding_dim_name: str = Arg(
        default="n_embd",
        help="Name of the embedding dimension in the model config. It is useful for Critic models where we attach head layer.",
    )
    lora_targets: Optional[str] = Arg(default="('c_attn',)", help="LoRA target layer names for the model.")


@dataclass
class OPT(ModelArgs):
    model_name: str = Arg(
        default="facebook/opt-350m", help="Name of the model. It will be used to load huggingface model."
    )
    embedding_dim_name: str = Arg(
        default="word_embed_proj_dim",
        help="Name of the embedding dimension in the model config. It is useful for Critic models where we attach head layer.",
    )
    lora_targets: Optional[str] = Arg(
        default="('q_proj','v_proj')",
        help="LoRA target layer names for the model.",
    )


@dataclass
class Falcon(ModelArgs):
    model_name: str = Arg(
        default="tiiuae/falcon-rw-1b", help="Name of the model. It will be used to load huggingface model."
    )
    lora_targets: Optional[str] = Arg(
        default="('query_key_value',)",
        help="LoRA target layer names for the model.",
    )


@dataclass
class Pythia(ModelArgs):
    model_name: str = Arg(
        default="EleutherAI/pythia-410m-deduped", help="Name of the model. It will be used to load huggingface model."
    )
    embedding_dim_name: str = Arg(
        default="hidden_size",
        help="Name of the embedding dimension in the model config. It is useful for Critic models where we attach head layer.",
    )
    lora_targets: Optional[str] = Arg(default="('query_key_value',)", help="LoRA target layer names for the model.")


### Data Configs ###


@dataclass
class TextDataArgs:
    destination_dir: str = Arg(help="Path to the directory where the dataset will be created.")
    tokenizer_name: str = Arg(help="Name of the tokenizer. It will be used to load huggingface tokenizer.")
    stage: str = Arg(
        default="finetune",
        metadata={"choices": ["finetune", "preference"]},
        help="Stage of the experiment. It can be `finetune` or `preference`.",
    )
    max_length: int = Arg(default=512, help="Maximum length of the input sequence.")
    max_prompt_length: int = Arg(default=256, help="Maximum length of the prompt sequence.")
    num_samples: Optional[int] = Arg(
        default=None, help="Number of samples to use from the dataset. If None, all samples will be used."
    )
    mask_prompt: bool = Arg(default=True, help="Whether to mask prompt tokens.")
    ignore_index: int = Arg(
        default=-1,
        help="Ignore index for loss calculation. This value will be used for masking targets in cross-entropy loss calculation if it is enabled.",
    )
    remove_same_responses: bool = Arg(
        default=True, help="Whether to remove samples with same chosen and rejected outputs."
    )
    remove_same_inputs: bool = Arg(default=True, help="Whether to remove samples with same prompt and chosen pairs.")
    minimum_response_length: int = Arg(default=2, help="Minimum length of the response.")
    save_skipped_examples: bool = Arg(
        default=False, help="Whether to save skipped examples to a json file for debugging."
    )
    shuffle: bool = Arg(default=True, help="Whether to shuffle the dataset.")
    seed: int = Arg(default=42, help="Seed for reproducibility")
    validation_percentage: float = Arg(default=0.1, help="Validation split percentage for the dataset")

    def to_dict(self) -> dict:
        return {"data_args": asdict(self)}

    @classmethod
    def from_json(cls, json_path: str) -> "TextDataArgs":
        with open(json_path, "r") as f:
            data_args = json.load(f)
        return cls(**data_args)


### Generation Configs ###


@dataclass
class GenerationArgs:
    max_new_tokens: int = Arg(
        default=128, help="Maximum number of tokens to generate. It is used for `generate` method."
    )
    num_beams: int = Arg(default=1, help="Number of beams to use for beam search. It is used for `generate` method.")
    do_sample: bool = Arg(
        default=True, help="Whether to use sampling for generation. It is used for `generate` method."
    )
    top_k: int = Arg(default=50, help="Top k value for top-k sampling. It is used for `generate` method.")
    top_p: float = Arg(default=1.0, help="Top p value for top-p sampling. It is used for `generate` method.")
    temperature: float = Arg(default=1.0, help="Temperature value for sampling. It is used for `generate` method.")
    num_return_sequences: int = Arg(default=1, help="Number of sequences to return. It is used for `generate` method.")

    def to_dict(self) -> dict:
        return {"gen_args": asdict(self)}


@dataclass
class ExploreArgs(GenerationArgs):
    top_k: int = Arg(default=0.0, help="Top k value for top-k sampling. It is used for `generate` method.")
    top_p: float = Arg(default=1.0, help="Top p value for top-p sampling. It is used for `generate` method.")
    do_sample: bool = Arg(
        default=True, help="Whether to use sampling for generation. It is used for `generate` method."
    )
