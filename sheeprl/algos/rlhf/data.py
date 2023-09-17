import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from sheeprl.algos.rlhf.utils import prepare_tokenizer


class DataProcessor:
    """
    The main class for processing data for the RLHF algorithm.

    Args:
        dataset_name (str): The name of the dataset to load.
        destination_dir (str): The directory where the processed data will be saved.
        tokenizer_name (str): The name of the tokenizer to use.
        max_length (int, optional): The maximum length of the input sequences. Defaults to 512.
        max_prompt_length (int, optional): The maximum length of the prompt sequences. Defaults to 512.
        num_samples (int, optional): The number of samples to use. Defaults to None.
        ignore_index (int, optional): The index to use for ignored tokens. Defaults to -1.
        remove_same_responses (bool, optional): Whether to remove samples with the same response. Defaults to True.
        remove_same_inputs (bool, optional): Whether to remove samples with the same input. Defaults to True.
        minimum_response_length (int, optional): The minimum length of the response sequences. Defaults to 2.
        save_skipped_examples (bool, optional): Whether to save skipped examples. Defaults to False.
        validation_split (float, optional): The validation split. Defaults to 0.1.
        reward_model_split (float, optional): The reward model split. Defaults to 0.5.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        seed (int, optional): The random seed. Defaults to 42.
        split_names (List[str], optional): The names of the splits. Defaults to ("train", "val", "test").
    """

    def __init__(
        self,
        dataset_name: str,
        destination_dir: str,
        tokenizer_name: str,
        max_length: int = 512,
        max_prompt_length: int = 512,
        num_samples: Optional[int] = None,
        ignore_index: int = -1,
        remove_same_responses: bool = True,
        remove_same_inputs: bool = True,
        minimum_response_length: int = 2,
        save_skipped_examples: bool = False,
        validation_split: float = 0.1,
        reward_model_split: float = 0.5,
        shuffle: bool = True,
        seed: int = 42,
        split_names: List[str] = ("train", "test"),
    ):
        self.dataset_name = dataset_name
        self.destination_dir = destination_dir
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.num_samples = num_samples
        self.ignore_index = ignore_index
        self.remove_same_responses = remove_same_responses
        self.remove_same_inputs = remove_same_inputs
        self.minimum_response_length = minimum_response_length
        self.save_skipped_examples = save_skipped_examples
        self.validation_split = validation_split
        self.reward_model_split = reward_model_split
        self.shuffle = shuffle
        self.seed = seed
        self.split_names = split_names
        self.tokenizer = prepare_tokenizer(tokenizer_name)

    def process(self):
        """
        The main method for processing the data.
        """
        destination_dir = Path(self.destination_dir)
        os.makedirs(destination_dir, exist_ok=True)
        cache_dir = destination_dir.parent / "cache"
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        for split in self.split_names:
            print(f"Processing {split} split ...")
            dataset = load_dataset(self.dataset_name, split=split, cache_dir=cache_dir)
            # shuffle the dataset
            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed)
            if self.num_samples is not None:
                dataset = dataset.select(range(self.num_samples))
            samples = []
            hashes = []
            skipped_samples = []
            for sample in tqdm(dataset):
                output = {}
                prompt = self.get_prompt(sample)
                chosen = self.get_chosen(sample)
                rejected = self.get_rejected(sample)
                input_hash = hash(prompt + chosen)
                if self.remove_same_inputs and input_hash in hashes:
                    skipped_samples.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "duplicate input"}
                    )
                    continue
                encoded_prompt = self.tokenizer(
                    bos_token + prompt,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                num_prompt_input_ids = len(encoded_prompt["input_ids"].squeeze())
                output["prompt_len"] = num_prompt_input_ids
                if num_prompt_input_ids > self.max_prompt_length:
                    skipped_samples.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "too long"}
                    )
                    continue
                # we need pairs of prompt and choosen and prompt and rejected
                if len(chosen) < self.minimum_response_length or len(rejected) < self.minimum_response_length:
                    skipped_samples.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "too short"}
                    )
                    continue
                prompt_chosen = bos_token + prompt + chosen + eos_token
                encoded_prompt_chosen = self.tokenizer(
                    prompt_chosen,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                prompt_rejected = bos_token + prompt + rejected + eos_token
                encoded_prompt_rejected = self.tokenizer(
                    prompt_rejected,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                if self.remove_same_responses and prompt_chosen == prompt_rejected:
                    skipped_samples.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected, "reason": "same response"}
                    )
                    continue
                output["chosen_input_ids"] = encoded_prompt_chosen["input_ids"].squeeze()
                output["rejected_input_ids"] = encoded_prompt_rejected["input_ids"].squeeze()
                chosen_masked_targets = output["chosen_input_ids"].clone()
                chosen_masked_targets[:num_prompt_input_ids] = self.ignore_index
                output["chosen_masked_targets"] = chosen_masked_targets
                rejected_masked_targets = output["rejected_input_ids"].clone()
                rejected_masked_targets[:num_prompt_input_ids] = self.ignore_index
                output["rejected_masked_targets"] = rejected_masked_targets
                samples.append(output)
                hashes.append(input_hash)
            print(f"Processed {len(samples)} samples, skipped {len(skipped_samples)} samples")
            output_data = {}
            if self.reward_model_split > 0:
                print(f"Using {self.reward_model_split * 100}% of the training data for the reward model training")
                num_reward_model_samples = int(len(samples) * self.reward_model_split)
                reward_model_samples = samples[:num_reward_model_samples]
                finetune_samples = samples[num_reward_model_samples:]
                output_data["reward_model"] = reward_model_samples
                output_data["finetune"] = finetune_samples
            else:
                output_data["finetune"] = samples
            if split == "train":
                print(f"Using {self.validation_split * 100}% of the training data for validation split")
                for data_name, data in output_data.items():
                    num_validation_samples = int(len(data) * self.validation_split)
                    validation_data = data[:num_validation_samples]
                    train_data = data[num_validation_samples:]
                    print(
                        f"Saving {len(train_data)} training samples and"
                        f"{len(validation_data)} validation samples for {data_name}"
                    )
                    torch.save(train_data, destination_dir / f"{data_name}_train.pt")
                    torch.save(validation_data, destination_dir / f"{data_name}_validation.pt")
            else:
                for data_name, data in output_data.items():
                    print(f"Saving {len(data)} {split} samples for {data_name}")
                    torch.save(data, destination_dir / f"{data_name}_test.pt")
            if self.save_skipped_examples:
                json.dump(skipped_samples, open(destination_dir / f"{split}_skipped.json", "w"), indent=4)

        example_prompt_path = destination_dir / "example_prompt.pt"
        example_prompt = self.create_example_prompt()
        torch.save(example_prompt, example_prompt_path)

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        raise NotImplementedError

    def get_chosen(self, sample: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

    def get_rejected(self, sample: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

    def get_example_prompt(self) -> str:
        raise NotImplementedError

    def wrap_prompt(self, prompt: str) -> str:
        raise NotImplementedError

    def create_example_prompt(self) -> Dict[str, Any]:
        prompt = self.get_example_prompt()
        wrapped_prompt = self.wrap_prompt(prompt)

        wrapped_prompt = self.tokenizer.bos_token + wrapped_prompt
        encoded_prompt = self.tokenizer(
            wrapped_prompt, max_length=self.max_length, truncation=True, return_tensors="pt", add_special_tokens=False
        )
        output = {
            "prompt": wrapped_prompt,
            "input_ids": encoded_prompt["input_ids"],
            "attention_mask": encoded_prompt["attention_mask"],
        }
        return output
