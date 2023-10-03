from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING


def remove_forward_slash(text: str) -> str:
    return text.replace("/", "_")


@dataclass
class DataConfig:
    _target_: str = "sheeprl.algos.rlhf.data.base.DataProcessor"
    name: str = MISSING
    dataset_name: str = MISSING
    root_dir: str = Path("./rlhf_data")
    tokenizer_name: str = II("model.name")
    max_length: int = 256
    max_prompt_length: int = 128
    num_samples: Optional[int] = None
    ignore_index: int = -1
    remove_same_responses: bool = True
    remove_same_inputs: bool = True
    minimum_response_length: int = 2
    save_skipped_examples: bool = False
    shuffle: bool = True
    seed: int = II("seed")
    validation_split: float = 0.1
    reward_model_split: float = 0.5
    split_names: Tuple[str] = ("train", "test")
    debug: bool = II("debug")


@dataclass
class HelpfulHarmlessConfig(DataConfig):
    _target_: str = "sheeprl.algos.rlhf.data.HelpfulHarmlessData"
    name: str = "helpful_harmless"
    dataset_name: str = "Dahoas/full-hh-rlhf"


@dataclass
class SummarizationConfig(DataConfig):
    _target_: str = "sheeprl.algos.rlhf.data.SummarizationData"
    name: str = "summarization"
    dataset_name: str = "CarperAI/openai_summarize_comparisons"


@dataclass
class GenConfig:
    # We cannot call this GenerationConfig because it will conflict with transformers.GenerationConfig
    max_new_tokens: int = 128
    num_beams: int = 1
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 1.0
    num_return_sequences: int = 1


def register_data_configs(cs: ConfigStore) -> None:
    cs.store(
        group="data",
        name="helpful_harmless",
        node=HelpfulHarmlessConfig,
    )
    cs.store(
        group="data",
        name="summarization",
        node=SummarizationConfig,
    )
    cs.store(
        group="generation",
        name="default",
        node=GenConfig,
    )
