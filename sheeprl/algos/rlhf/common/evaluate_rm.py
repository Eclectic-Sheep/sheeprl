import os
import sys
from typing import Callable

import lightning
import torch
from _pytest.cacheprovider import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sheeprl.algos.rlhf.args import EvaluateArgs, EvaluateRewardArgs, ModelArgs, TextDataArgs
from sheeprl.algos.rlhf.collate import CompareCollate
from sheeprl.algos.rlhf.loss import load_reward_loss
from sheeprl.algos.rlhf.models import CriticModel
from sheeprl.algos.rlhf.utils import get_last_checkpoint_path, load_args_from_json, prepare_tokenizer
from sheeprl.utils.parser import HfArgumentParser

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


@torch.inference_mode()
def accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor):
    tp = torch.count_nonzero(chosen_rewards > rejected_rewards)
    total = chosen_rewards.shape[0]
    acc = tp / total
    return acc


@torch.inference_mode()
def evaluate(model: CriticModel, test_dataloader: DataLoader, loss: Callable, pad_token_id: int):
    eval_counter = 0
    average_acc = 0
    average_loss = 0
    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc="Evaluating")
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


def main():
    if len(sys.argv) > 1:
        parser = HfArgumentParser([EvaluateArgs])
        dataclasses = parser.parse_args_into_dataclasses()
        eval_args: EvaluateRewardArgs = dataclasses[0]
    else:
        eval_args = EvaluateRewardArgs(
            experiment_dir=os.environ.get("EVAL_DIR"), use_pretrained=False, mini_batch_size=32, num_workers=4, seed=42
        )
    exp_args = load_args_from_json(experiment_dir=eval_args.experiment_dir)
    model_args = ModelArgs(**exp_args["model_args"])
    data_args = TextDataArgs(**exp_args["data_args"])
    checkpoint_path = get_last_checkpoint_path(experiment_dir=eval_args.experiment_dir)

    fabric = lightning.Fabric(accelerator="auto")
    fabric.launch()
    fabric.seed_everything(eval_args.seed + fabric.global_rank)
    fabric.print(f"Fabric Rank: {fabric.global_rank}")
    fabric.print(f"Evaluation Experiment: {eval_args.experiment_dir}")

    # Setup Tokenizer
    tokenizer = prepare_tokenizer(tokenizer_name=model_args.model_name)

    if eval_args.use_pretrained:
        fabric.print("\nLoading Pretrained model")
        path = None
    else:
        fabric.print("\nLoading Finetuned model")
        path = checkpoint_path

    model = CriticModel.from_checkpoint(
        device=fabric.device,
        model_args=model_args,
        path=path,
        freeze=True,
    )
    model.to(fabric.device)
    model.eval()
    fabric.print("Model loaded")

    # Setup Dataloaders
    collator = CompareCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)

    for split in ["train", "test"]:
        fabric.print(f"\nEvaluating on {split} split")
        test_data = torch.load(Path(data_args.destination_dir) / f"preference_{split}.pt")
        test_dataloader = DataLoader(
            test_data,
            shuffle=False,
            batch_size=eval_args.mini_batch_size,
            collate_fn=collator,
            num_workers=eval_args.num_workers,
        )
        test_dataloader = fabric.setup_dataloaders(test_dataloader)

        # Setup Reward Loss
        reward_loss = load_reward_loss(eval_args.loss_type)

        avg_loss, avg_acc = evaluate(
            model=model,
            test_dataloader=test_dataloader,
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
