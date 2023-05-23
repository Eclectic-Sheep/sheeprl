import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
from lit_parrot.tokenizer import Tokenizer
from datasets import load_dataset


def prepare(
    destination_path: Path = Path("data/Dahoas/static-hh"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    stage: str = "sft",
    max_seq_length: int = 512,
    max_prompt_length: int = 512,
    mask_inputs: bool = False,
    num_samples: Optional[int] = None,
    ignore_index: int = -1,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")
    os.makedirs(destination_path, exist_ok=True)

    for split in ["train", "test"]:
        print(f"Processing {split} split ...")
        dataset = load_dataset("Dahoas/static-hh", split=split)
        if stage == "sft" or stage == "ppo":
            # first half of the dataset
            dataset = dataset.select(range(len(dataset) // 2))
        elif stage == "rm":
            # second half of the dataset
            dataset = dataset.select(range(len(dataset) // 2, len(dataset)))
        else:
            raise ValueError(f"stage must be one of 'rm', 'sft', 'ppo', but got {stage}")

        if num_samples is not None:
            dataset = dataset.select(range(num_samples))
        samples = []
        for sample in tqdm(dataset):
            output = {}
            encoded_prompt = tokenizer.encode(sample["prompt"], max_length=max_seq_length)

            if len(encoded_prompt) > max_prompt_length:
                continue
            output["prompt_input_ids"] = encoded_prompt["input_ids"].squeeze()
            if stage == "sft":
                # we use prompt and choosen as data
                prompt_response = sample["prompt"] + sample["chosen"]
                encoded_prompt_response = tokenizer.encode(prompt_response, eos=True, max_length=max_seq_length)
                labels = encoded_prompt_response.clone()
                output["input_ids"] = encoded_prompt_response["input_ids"]
                if mask_inputs:
                    labels[: len(encoded_prompt)] = ignore_index
                output["labels"] = labels
            elif stage == "rm":
                # we need pairs of prompt and choosen and prompt and rejected
                prompt_chosen = sample["prompt"] + sample["chosen"]
                encoded_prompt_chosen = tokenizer.encode(prompt_chosen, eos=True, max_length=max_seq_length)
                output["chosen_input_ids"] = encoded_prompt_chosen["input_ids"].squeeze()

                prompt_rejected = sample["prompt"] + sample["rejected"]
                encoded_prompt_rejected = tokenizer.encode(prompt_rejected, eos=True, max_length=max_seq_length)
                output["rejected_input_ids"] = encoded_prompt_rejected["input_ids"].squeeze()
            samples.append(output)
        print(f"Processed {len(samples)} samples")
        torch.save(samples, destination_path / f"{stage}-{split}.pt")

    example_prompt = create_example_prompt(tokenizer)
    torch.save(example_prompt, destination_path / "example_prompt.pt")


def wrap_prompt(prompt: str) -> str:
    return "\n\nHuman: " + prompt + "\n\nAssistant: "


def create_example_prompt(tokenizer: Tokenizer, max_seq_length: int = 256) -> Dict[str, Any]:
    prompt = "How does the computer work?"
    wrapped_prompt = wrap_prompt(prompt)
    encoded_prompt = tokenizer(wrapped_prompt, max_length=max_seq_length, return_tensors="pt", truncation=True)
    return encoded_prompt


if __name__ == "__main__":
    prepare()
