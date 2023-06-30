from dataclasses import asdict
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from sheeprl.algos.rlhf.args import OPT, TextDataArgs

from sheeprl.utils.parser import HfArgumentParser
import json

import torch
from datasets import load_dataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def prepare(
    destination_dir: str,
    tokenizer_name: str,
    stage: str = "finetune",
    max_length: int = 512,
    max_prompt_length: int = 512,
    mask_prompt: bool = True,
    num_samples: Optional[int] = None,
    ignore_index: int = -1,
    remove_same_output: bool = True,
) -> None:
    destination_dir = Path(destination_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    os.makedirs(destination_dir, exist_ok=True)
    cache_dir = destination_dir / "cache"
    skipped = 0
    for split in ["train", "test"]:
        print(f"Processing {split} split ...")
        dataset = load_dataset("CarperAI/openai_summarize_comparisons", split=split, cache_dir=cache_dir)
        if stage == "finetune":
            # first half of the dataset
            dataset = dataset.select(range(len(dataset) // 2))
        elif stage == "preference":
            # second half of the dataset
            dataset = dataset.select(range(len(dataset) // 2, len(dataset)))
        else:
            raise ValueError(f"stage must be one of 'finetune', 'preference', but got {stage}")

        if num_samples is not None:
            dataset = dataset.select(range(num_samples))
        samples = []
        for sample in tqdm(dataset):
            output = {}
            prompt = sample["prompt"] + "\nTL;DR: "
            encoded_prompt = tokenizer(
                prompt,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded_prompt_input_ids = encoded_prompt["input_ids"].squeeze()
            if len(encoded_prompt_input_ids) > max_prompt_length:
                skipped += 1
                continue
            if stage == "sft":
                # we use prompt and choosen as data
                chosen = sample["chosen"][8:]  # remove "TL;DR: "
                prompt_response = prompt + chosen + tokenizer.eos_token
                encoded_prompt_response = tokenizer(
                    prompt_response,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = encoded_prompt_response["input_ids"].squeeze()
                targets = input_ids.clone()
                output["input_ids"] = input_ids
                if mask_prompt:
                    targets[: len(encoded_prompt_input_ids)] = ignore_index
                output["targets"] = targets
                output["prompt_input_ids"] = encoded_prompt["input_ids"].squeeze()
            elif stage == "rm":
                # we need pairs of prompt and choosen and prompt and rejected
                chosen = sample["chosen"][8:]  # remove "TL;DR: "
                rejected = sample["rejected"][8:]  # remove "TL;DR: "
                prompt_chosen = prompt + chosen + tokenizer.eos_token
                encoded_prompt_chosen = tokenizer(
                    prompt_chosen,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                prompt_rejected = prompt + rejected + tokenizer.eos_token
                encoded_prompt_rejected = tokenizer(
                    prompt_rejected,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                if remove_same_output and prompt_chosen == prompt_rejected:
                    skipped += 1
                    continue
                output["chosen_input_ids"] = encoded_prompt_chosen["input_ids"].squeeze()
                output["rejected_input_ids"] = encoded_prompt_rejected["input_ids"].squeeze()
            samples.append(output)
        print(f"Processed {len(samples)} samples, skipped {skipped} samples")
        torch.save(samples, destination_dir / f"{stage}_{split}.pt")

    example_prompt_path = destination_dir / "example_prompt.pt"
    example_prompt = create_example_prompt(tokenizer)
    torch.save(example_prompt, example_prompt_path)


def wrap_prompt(subreddit: str, title: str, prompt: str) -> str:
    return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {prompt}\nTL;DR: "


def create_example_prompt(tokenizer: PreTrainedTokenizer, max_length: int = 256) -> Dict[str, Any]:
    prompt = "Hello everyone, I've been having some trouble with my computer and I was hoping someone could help me out. I've had this computer for about 2 years. Recently, my computer has been running really slow and it takes forever to load anything. I've tried running virus scans and deleting unnecessary files, but nothing seems to be working. Sometimes, the computer even freezes completely and I have to restart it. One thing that I have noticed is that the fan on my laptop seems to be running constantly and sometimes it's quite loud, even when I'm not doing anything particularly demanding on the computer. I'm not sure if this is related to the performance issues, but it's something that I thought might be worth mentioning. I'm really hoping that someone can help me figure out what's causing these problems and what I can do to fix them. I don't have a lot of experience with troubleshooting hardware issues, so any advice or guidance would be greatly appreciated! Does anyone have any ideas for what I can do to fix this?"
    wrapped_prompt = wrap_prompt(
        subreddit="TechSupport",
        title="Need help with my slow and freezing computer, fan running constantly",
        prompt=prompt,
    )
    encoded_prompt = tokenizer(wrapped_prompt, max_length=max_length, truncation=True, return_tensors="pt")
    output = {
        "prompt": wrapped_prompt,
        "input_ids": encoded_prompt["input_ids"],
        "attention_mask": encoded_prompt["attention_mask"],
    }
    return output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = HfArgumentParser([TextDataArgs])
        dataclasses = parser.parse_args_into_dataclasses()
        data_args: TextDataArgs = dataclasses[0]
    else:
        data_args = TextDataArgs(
            destination_dir="data/CarperAI/openai_summarize_comparisons",
            tokenizer_name=OPT().model_name,
            stage="finetune",
            max_length=512,
            max_prompt_length=512,
        )
    prepare(**asdict(data_args))
    with open(Path(data_args.destination_dir) / "args.json", "w") as f:
        json.dump(asdict(data_args), f, indent=4)
