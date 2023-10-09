from pathlib import Path
from typing import Callable, Optional

import hydra
import lightning
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sheeprl.algos.rlhf.collate import CompareCollate
from sheeprl.algos.rlhf.config_store.algo import RMAlgoConfig
from sheeprl.algos.rlhf.config_store.data import DataConfig
from sheeprl.algos.rlhf.config_store.model import ModelConfig
from sheeprl.algos.rlhf.data.base import TextDataset
from sheeprl.algos.rlhf.loss import load_reward_loss
from sheeprl.algos.rlhf.models import RewardModel
from sheeprl.algos.rlhf.utils import get_last_checkpoint_path, validate_dataset
from sheeprl.utils.utils import dotdict


@torch.inference_mode()
def accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor):
    tp = torch.count_nonzero(chosen_rewards > rejected_rewards)
    total = chosen_rewards.shape[0]
    acc = tp / total
    return acc


@torch.inference_mode()
def evaluate(model: RewardModel, dataloader: DataLoader, loss: Callable, pad_token_id: int):
    eval_counter = 0
    average_acc = 0
    average_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    for batch in pbar:
        chosen_rewards = model(
            input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"], use_cache=True
        )
        rejected_rewards = model(
            input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"], use_cache=True
        )
        test_loss, choosen_last_rewards, rejected_last_rewards = loss(
            chosen=batch["chosen_input_ids"],
            rejected=batch["rejected_input_ids"],
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            pad_token_id=pad_token_id,
        )
        average_loss += test_loss.detach()
        acc = accuracy(choosen_last_rewards, rejected_last_rewards)
        average_acc += acc
        eval_counter += 1
    average_acc /= eval_counter
    average_loss /= eval_counter
    return (
        average_loss,
        average_acc,
    )


@hydra.main(version_base=None, config_path="../sheeprl/configs", config_name="rlhf_eval_config")
def main(cfg: DictConfig) -> None:
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    # gen_cfg: GenConfig = GenConfig(**cfg.generation)

    experiment_dir: Optional[str] = cfg.experiment_dir
    data_split: str = cfg.data_split
    num_workers: int = cfg.num_workers
    mini_batch_size: int = cfg.mini_batch_size
    seed = cfg.seed

    exp_cfg = OmegaConf.load(Path(experiment_dir) / ".hydra/config.yaml")
    ckpt_model_cfg: ModelConfig = ModelConfig(**exp_cfg.model)
    algo_cfg = RMAlgoConfig(**exp_cfg.algo)
    data_cfg: DataConfig = DataConfig(**exp_cfg.data)
    checkpoint_path = get_last_checkpoint_path(experiment_dir)

    fabric = lightning.Fabric(accelerator="auto")
    fabric.launch()
    fabric.seed_everything(seed + fabric.global_rank)
    fabric.print(f"Fabric Rank: {fabric.global_rank}")
    fabric.print(f"Evaluation Experiment: {experiment_dir}")

    model = RewardModel.from_checkpoint(
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

    collator = CompareCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    dataset = TextDataset(dataframe_path=dataset_path / f"reward_model_{data_split}.pkl")
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=mini_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    # Setup Reward Loss
    reward_loss = load_reward_loss(algo_cfg.loss_type)

    avg_loss, avg_acc = evaluate(
        model=model,
        dataloader=dataloader,
        pad_token_id=tokenizer.pad_token_id,
        loss=reward_loss,
    )
    fabric.print(f"Average Loss: {avg_loss}")
    fabric.print(f"Average Accuracy: {avg_acc}")


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    main()
