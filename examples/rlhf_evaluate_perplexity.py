from pathlib import Path
from typing import Optional

import hydra
import lightning
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sheeprl.algos.rlhf.collate import SFTCollate
from sheeprl.algos.rlhf.config_store.data import DataConfig
from sheeprl.algos.rlhf.config_store.model import ModelConfig
from sheeprl.algos.rlhf.data.base import TextDataset
from sheeprl.algos.rlhf.loss import finetune_loss
from sheeprl.algos.rlhf.models import CasualModel
from sheeprl.algos.rlhf.utils import get_last_checkpoint_path, validate_dataset
from sheeprl.utils.utils import dotdict


@torch.inference_mode()
def evaluate(
    model: CasualModel,
    use_masked_targets: bool,
    label_smoothing: float,
    data_cfg: DataConfig,
    dataloader: DataLoader,
):
    eval_counter = 0
    total_loss = 0.0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    for batch in pbar:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        targets = batch["targets"] if use_masked_targets else batch["input_ids"].detach().clone()
        loss = finetune_loss(
            outputs=outputs,
            targets=targets,
            ignore_index=data_cfg.ignore_index,
            label_smoothing=label_smoothing,
        )
        total_loss += loss
        eval_counter += 1
    average_loss = total_loss / eval_counter
    try:
        perplexity = torch.exp(average_loss).item()
    except OverflowError:
        perplexity = float("inf")
    return perplexity


@hydra.main(version_base=None, config_path="../sheeprl/configs", config_name="rlhf_eval_config")
def main(cfg: DictConfig) -> None:
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    # gen_cfg: GenConfig = GenConfig(**cfg.generation)

    experiment_dir: Optional[str] = cfg.experiment_dir
    data_split: str = cfg.data_split
    num_workers: int = cfg.num_workers
    mini_batch_size: int = cfg.mini_batch_size
    seed = cfg.seed
    use_masked_targets: bool = cfg.use_masked_targets
    label_smoothing: float = cfg.label_smoothing

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

    collator = SFTCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    dataset = TextDataset(dataframe_path=dataset_path / f"finetune_{data_split}.pkl")
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=mini_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
    )
    dataloader = fabric.setup_dataloaders(dataloader)
    result = evaluate(
        model=model,
        use_masked_targets=use_masked_targets,
        label_smoothing=label_smoothing,
        data_cfg=data_cfg,
        dataloader=dataloader,
    )
    fabric.print(f"Perplexity on {data_split}: {result}")


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    main()
