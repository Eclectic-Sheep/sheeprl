from pathlib import Path
from typing import Optional

import evaluate
import hydra
import lightning
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig

from sheeprl.algos.rlhf.collate import EvaluateCollate
from sheeprl.algos.rlhf.config_store.data import DataConfig, GenConfig
from sheeprl.algos.rlhf.config_store.model import ModelConfig
from sheeprl.algos.rlhf.data.base import TextDataset
from sheeprl.algos.rlhf.models import CasualModel
from sheeprl.algos.rlhf.utils import get_last_checkpoint_path, prepare_generation_config, validate_dataset
from sheeprl.utils.utils import dotdict

rouge_metric = evaluate.load("rouge")


@torch.inference_mode()
def evaluate(
    model: CasualModel,
    generation_config: GenerationConfig,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
):
    generated_list = []
    target_list = []
    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    for batch in pbar:
        generated_input_ids = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            generation_config=generation_config,
            use_cache=True,
        )
        response_ground_truth = []
        response_generated = []
        for i in range(len(batch["input_ids"])):
            prompt_len = batch["prompt_len"][i]
            response_ground_truth.append(batch["input_ids"][i][prompt_len:])
            response_generated.append(generated_input_ids[i][prompt_len:])

        generated_response_text = tokenizer.batch_decode(response_generated, skip_special_tokens=True)
        target_response_text = tokenizer.batch_decode(response_ground_truth, skip_special_tokens=True)

        generated_list.extend(generated_response_text)
        target_list.extend(target_response_text)
    rouge_score = rouge_metric.compute(predictions=generated_list, references=target_list)
    return rouge_score


@hydra.main(version_base=None, config_path="../sheeprl/configs", config_name="rlhf_eval_config")
def main(cfg: DictConfig) -> None:
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    gen_cfg: GenConfig = GenConfig(**cfg.generation)

    experiment_dir: Optional[str] = cfg.experiment_dir
    data_split: str = cfg.data_split
    num_workers: int = cfg.num_workers
    mini_batch_size: int = cfg.mini_batch_size
    seed = cfg.seed

    exp_cfg = OmegaConf.load(Path(experiment_dir) / ".hydra/config.yaml")
    ckpt_model_cfg: ModelConfig = ModelConfig(**exp_cfg.model)
    data_cfg: DataConfig = DataConfig(**exp_cfg.data)
    checkpoint_path = get_last_checkpoint_path(experiment_dir)

    fabric = lightning.Fabric(accelerator="auto")
    fabric.launch()
    fabric.seed_everything(seed + fabric.global_rank)
    fabric.print(f"Fabric Rank: {fabric.global_rank}")
    fabric.print(f"Evaluation Experiment: {experiment_dir}")

    model = CasualModel.from_checkpoint(
        device=fabric.device,
        model_cfg=ckpt_model_cfg,
        path=checkpoint_path,
        freeze=True,
    )
    model.to(fabric.device)
    model.eval()
    fabric.print("Model loaded")

    # Setup Dataloaders
    data_processor = validate_dataset(fabric, data_cfg)
    dataset_path = Path(data_processor.full_path)
    tokenizer = data_processor.tokenizer

    # Setup Dataloaders
    collator = EvaluateCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    dataset = TextDataset(dataframe_path=dataset_path / f"finetune_{data_split}.pkl")
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=mini_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    # Setup Generation Config
    eval_generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_cfg=ckpt_model_cfg,
        gen_cfg=gen_cfg,
        fabric=fabric,
    )

    result = evaluate(
        model=model,
        generation_config=eval_generation_config,
        dataloader=dataloader,
        tokenizer=tokenizer,
    )
    fabric.print(f"Rouge Scores: {result}")


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    main()
