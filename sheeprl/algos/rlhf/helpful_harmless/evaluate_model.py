import os
import sys

import lightning
import torch
from _pytest.cacheprovider import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sheeprl.algos.rlhf.args import EvaluateArgs, EvaluatePerplexityArgs, ModelArgs, TextDataArgs
from sheeprl.algos.rlhf.data import SFTCollate
from sheeprl.algos.rlhf.loss import finetune_loss
from sheeprl.algos.rlhf.models import CasualModel
from sheeprl.algos.rlhf.utils import get_last_checkpoint_path, load_args_from_json, prepare_tokenizer
from sheeprl.utils.parser import HfArgumentParser


@torch.inference_mode()
def evaluate(
    model: CasualModel,
    eval_args: EvaluatePerplexityArgs,
    data_args: TextDataArgs,
    test_dataloader: DataLoader,
):
    eval_counter = 0
    total_loss = 0.0
    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc="Evaluating")
    for batch in pbar:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        targets = batch["targets"] if eval_args.use_targets else batch["input_ids"].detach().clone()
        loss = finetune_loss(
            outputs=outputs,
            targets=targets,
            ignore_index=data_args.ignore_index,
            label_smoothing=eval_args.label_smoothing,
        )
        total_loss += loss
        eval_counter += 1
    average_loss = total_loss / eval_counter
    try:
        perplexity = torch.exp(average_loss).item()
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def main():
    if len(sys.argv) > 1:
        parser = HfArgumentParser([EvaluateArgs])
        dataclasses = parser.parse_args_into_dataclasses()
        eval_args: EvaluatePerplexityArgs = dataclasses[0]
    else:
        eval_args = EvaluatePerplexityArgs(experiment_dir=os.environ.get("EVAL_DIR"), use_pretrained=False)
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
    tokenizer = prepare_tokenizer(model_args.model_name)

    if eval_args.use_pretrained:
        fabric.print("\nLoading Pretrained model")
        path = None
    else:
        fabric.print("\nLoading Finetuned model")
        path = checkpoint_path

    model = CasualModel.from_checkpoint(
        device=fabric.device,
        model_args=model_args,
        path=path,
        freeze=True,
    )
    model.to(fabric.device)
    model.eval()
    fabric.print("Model loaded")

    # Setup Dataloaders
    collator = SFTCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)

    test_data = torch.load(Path(data_args.destination_dir) / f"finetune_test.pt")
    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=eval_args.mini_batch_size,
        collate_fn=collator,
        num_workers=eval_args.num_workers,
    )
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    result = evaluate(
        model=model,
        eval_args=eval_args,
        data_args=data_args,
        test_dataloader=test_dataloader,
    )
    fabric.print(f"Perplexity: {result}")


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    main()
