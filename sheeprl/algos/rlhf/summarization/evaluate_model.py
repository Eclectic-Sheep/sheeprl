import os
import sys
from dataclasses import asdict

import evaluate
import lightning
import torch
from _pytest.cacheprovider import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig

from sheeprl.algos.rlhf.args import EvaluateArgs, GenerationArgs, ModelArgs, TextDataArgs
from sheeprl.algos.rlhf.data import EvaluateCollate
from sheeprl.algos.rlhf.models import CasualModel
from sheeprl.algos.rlhf.utils import get_last_checkpoint_path, load_args_from_json
from sheeprl.utils.parser import HfArgumentParser

rouge_metric = evaluate.load("rouge")


@torch.inference_mode()
def evaluate(
    model: CasualModel,
    generation_config: GenerationConfig,
    tokenizer: AutoTokenizer,
    test_dataloader: DataLoader,
):
    generated_list = []
    target_list = []
    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc="Evaluating")
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


def main():
    if len(sys.argv) > 1:
        parser = HfArgumentParser([EvaluateArgs])
        dataclasses = parser.parse_args_into_dataclasses()
        eval_args: EvaluateArgs = dataclasses[0]
    else:
        eval_args = EvaluateArgs(experiment_dir=os.environ.get("EVAL_DIR"), use_pretrained=False)
    exp_args = load_args_from_json(experiment_dir=eval_args.experiment_dir)
    model_args = ModelArgs(**exp_args["model_args"])
    data_args = TextDataArgs(**exp_args["data_args"])
    gen_args = GenerationArgs(**exp_args["gen_args"])
    checkpoint_path = get_last_checkpoint_path(experiment_dir=eval_args.experiment_dir)

    fabric = lightning.Fabric(accelerator="auto")
    fabric.launch()
    fabric.seed_everything(eval_args.seed + fabric.global_rank)
    fabric.print(f"Fabric Rank: {fabric.global_rank}")
    fabric.print(f"Evaluation Experiment: {eval_args.experiment_dir}")

    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    collator = EvaluateCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_args.ignore_index)
    test_data = torch.load(Path(data_args.destination_dir) / "finetune_test.pt")

    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=eval_args.mini_batch_size,
        collate_fn=collator,
        num_workers=eval_args.num_workers,
    )
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    # Setup Generation Config
    try:
        eval_generation_config = GenerationConfig.from_pretrained(model_args.model_name, **asdict(gen_args))
    except EnvironmentError:
        # If the model does not have `generation_config.json` file, we create from scratch
        fabric.print("`generation_config.json` not found, creating `GenerationConfig` from scratch")
        eval_generation_config = GenerationConfig(**asdict(eval_args))
        eval_generation_config.pad_token_id = tokenizer.pad_token_id
        eval_generation_config.eos_token_id = tokenizer.eos_token_id
        eval_generation_config.bos_token_id = tokenizer.bos_token_id

    result = evaluate(
        model=model,
        generation_config=eval_generation_config,
        test_dataloader=test_dataloader,
        tokenizer=tokenizer,
    )
    fabric.print(f"Rouge Scores: {result}")


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    main()
