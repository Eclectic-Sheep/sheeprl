import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from sheeprl.algos.rlhf.args import TextDataArgs
from sheeprl.algos.rlhf.utils import prepare_tokenizer
from sheeprl.utils.parser import HfArgumentParser

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
    remove_same_responses: bool = True,
    remove_same_inputs: bool = True,
    minimum_response_length: int = 2,
    save_skipped_examples: bool = False,
    validation_percentage: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    destination_dir = Path(destination_dir)
    tokenizer = prepare_tokenizer(tokenizer_name)
    os.makedirs(destination_dir, exist_ok=True)
    cache_dir = destination_dir.parent / "cache"
    for split in ["train", "test"]:
        print(f"Processing {split} split ...")
        dataset = load_dataset("CarperAI/openai_summarize_comparisons", split=split, cache_dir=cache_dir)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
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
        hashes = []
        skipped_samples = []
        for sample in tqdm(dataset):
            output = {}
            prompt = sample["prompt"] + "\nTL;DR: "
            chosen = sample["chosen"][8:]  # remove "TL;DR: "
            rejected = sample["rejected"][8:]  # remove "TL;DR: "
            input_hash = hash(prompt + chosen)
            if remove_same_inputs and input_hash in hashes:
                skipped_samples.append(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "duplicate input"}
                )
                continue
            encoded_prompt = tokenizer(
                tokenizer.bos_token + prompt,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=False,
            )

            num_prompt_input_ids = len(encoded_prompt["input_ids"].squeeze())
            output["prompt_len"] = num_prompt_input_ids
            if num_prompt_input_ids > max_prompt_length:
                skipped_samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "too long"})
                continue
            if stage == "finetune":
                # we use prompt and choosen as data
                if len(chosen) < minimum_response_length:
                    skipped_samples.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "too short"}
                    )
                    continue
                prompt_response = tokenizer.bos_token + prompt + chosen + tokenizer.eos_token
                encoded_prompt_response = tokenizer(
                    prompt_response,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                input_ids = encoded_prompt_response["input_ids"].squeeze()
                targets = input_ids.clone()
                output["input_ids"] = input_ids
                if mask_prompt:
                    targets[:num_prompt_input_ids] = ignore_index
                output["targets"] = targets

            elif stage == "preference":
                # we need pairs of prompt and choosen and prompt and rejected
                if len(chosen) < minimum_response_length or len(rejected) < minimum_response_length:
                    skipped_samples.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "too short"}
                    )
                    continue
                prompt_chosen = tokenizer.bos_token + prompt + chosen + tokenizer.eos_token
                encoded_prompt_chosen = tokenizer(
                    prompt_chosen,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                prompt_rejected = tokenizer.bos_token + prompt + rejected + tokenizer.eos_token
                encoded_prompt_rejected = tokenizer(
                    prompt_rejected,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                if remove_same_responses and prompt_chosen == prompt_rejected:
                    skipped_samples.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "same response"}
                    )
                    continue
                output["chosen_input_ids"] = encoded_prompt_chosen["input_ids"].squeeze()
                output["rejected_input_ids"] = encoded_prompt_rejected["input_ids"].squeeze()
            else:
                raise ValueError(f"stage must be one of 'finetune', 'preference', but got {stage}")
            samples.append(output)
            hashes.append(input_hash)
        print(f"Processed {len(samples)} samples, skipped {len(skipped_samples)} samples")
        if split == "train":
            print(f"Using {validation_percentage * 100}% of the training data for validation")
            num_validation_samples = int(len(samples) * validation_percentage)
            validation_samples = samples[:num_validation_samples]
            train_samples = samples[num_validation_samples:]
            torch.save(train_samples, destination_dir / f"{stage}_train.pt")
            torch.save(validation_samples, destination_dir / f"{stage}_validation.pt")
        else:
            torch.save(samples, destination_dir / f"{stage}_{split}.pt")
        if save_skipped_examples:
            json.dump(skipped_samples, open(destination_dir / f"{stage}_{split}_skipped.json", "w"), indent=4)

    example_prompt_path = destination_dir / "example_prompt.pt"
    example_prompt = create_example_prompt(tokenizer, max_length=max_length)
    torch.save(example_prompt, example_prompt_path)


def wrap_prompt(subreddit: str, title: str, prompt: str) -> str:
    return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {prompt}\nTL;DR: "


def create_example_prompt(tokenizer: PreTrainedTokenizer, max_length: int) -> Dict[str, Any]:
    prompt = "Hello everyone, I've been having some trouble with my computer and I was hoping someone could "
    "help me out. I've had this computer for about 2 years. Recently, my computer has been running really slow "
    "and it takes forever to load anything. I've tried running virus scans and deleting unnecessary files, "
    "but nothing seems to be working. Sometimes, the computer even freezes completely and I have to restart it. "
    "One thing that I have noticed is that the fan on my laptop seems to be running constantly and sometimes "
    "it's quite loud, even when I'm not doing anything particularly demanding on the computer. I'm not sure if "
    "this is related to the performance issues, but it's something that I thought might be worth mentioning. "
    "I'm really hoping that someone can help me figure out what's causing these problems and what I can do to "
    "fix them. I don't have a lot of experience with troubleshooting hardware issues, so any advice or guidance "
    "would be greatly appreciated! Does anyone have any ideas for what I can do to fix this?"
    wrapped_prompt = wrap_prompt(
        subreddit="TechSupport",
        title="Need help with my slow and freezing computer, fan running constantly",
        prompt=prompt,
    )
    wrapped_prompt = tokenizer.bos_token + wrapped_prompt
    encoded_prompt = tokenizer(
        wrapped_prompt,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    output = {
        "prompt": wrapped_prompt,
        "input_ids": encoded_prompt["input_ids"],
        "attention_mask": encoded_prompt["attention_mask"],
    }
    return output


if __name__ == "__main__":
    parser = HfArgumentParser([TextDataArgs])
    dataclasses = parser.parse_args_into_dataclasses()
    data_args: TextDataArgs = dataclasses[0]

    prepare(**asdict(data_args))
    with open(Path(data_args.destination_dir) / "args.json", "w") as f:
        json.dump(asdict(data_args), f, indent=4)
