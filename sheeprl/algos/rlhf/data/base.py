import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import PreTrainedTokenizer

HuggingfaceDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
TextSample = Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe_path: str):
        self.dataframe = pd.read_pickle(dataframe_path).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_dict()
        return row

    def __len__(self):
        return len(self.dataframe)


class DataProcessor:
    """
    The main class for processing data for the RLHF algorithm.

    Args:
        name (str): The name of the processor.
        dataset_name (str): The name of the dataset to load.
        root_dir (str): The directory where the processed data will be saved.
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
        split_names (Tuple[str], optional): The names of the splits. Defaults to ("train", "val", "test").
    """

    def __init__(
        self,
        name: str,
        dataset_name: str,
        root_dir: str,
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
        num_threads: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        split_names: Tuple[str] = ("train", "test"),
        debug: bool = False,
    ):
        self.name = name
        self.dataset_name = dataset_name
        self.root_dir = root_dir
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
        self.num_threads = num_threads
        self.debug = debug

    @property
    def tokenizer(self) -> "PreTrainedTokenizer":
        """
        The tokenizer property.

        Returns:
            PreTrainedTokenizer: The tokenizer.
        """
        return self._tokenizer

    @property
    def full_path(self) -> Path:
        root_dir = Path(self.root_dir)
        processed_name = self.remove_forward_slash(self.name)
        processed_tokenizer_name = self.remove_forward_slash(self.tokenizer_name)
        processed_dataset_name = self.remove_forward_slash(self.dataset_name)
        _full_path = root_dir / processed_name / processed_dataset_name / processed_tokenizer_name
        return _full_path

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def load_dataframe(self, **kwargs) -> pd.DataFrame:
        dataset: HuggingfaceDataset = load_dataset(**kwargs)
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed)
        if self.num_samples is not None:
            dataset = dataset.select(range(self.num_samples))
        dataframe = dataset.to_pandas()
        return dataframe

    def _load_dataframe(self, **kwargs) -> pd.DataFrame:
        dataframe = self.load_dataframe(**kwargs)
        return dataframe

    def remove_forward_slash(self, text: str) -> str:
        """
        Removes forward slashes from a string.

        Args:
            text (str): The string to remove forward slashes from.

        Returns:
            str: The string with forward slashes removed.
        """
        return text.replace("/", "_")

    def process(self):
        """
        The main method for processing the data.
        """
        full_path = self.full_path
        os.makedirs(full_path, exist_ok=True)
        cache_dir = full_path.parent / "cache"
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        for split in self.split_names:
            skipped_samples = []
            print(f"Processing {split} split ...")
            time_start = time.perf_counter()
            dataframe = self._load_dataframe(path=self.dataset_name, split=split, cache_dir=cache_dir)
            dataframe["prompt"] = dataframe.apply(lambda x: bos_token + self.get_prompt(x), axis=1)
            dataframe["chosen"] = dataframe.apply(lambda x: self.get_chosen(x) + eos_token, axis=1)
            dataframe["rejected"] = dataframe.apply(lambda x: self.get_rejected(x) + eos_token, axis=1)
            dataframe["skip_reason"] = None
            dataframe.reset_index(inplace=True)
            if self.remove_same_inputs:
                duplicate_filter = dataframe.duplicated(subset=["prompt", "chosen"], keep="first")
                dataframe.loc[duplicate_filter, "skip_reason"] = "duplicate"
                skipped_dataframe = dataframe[duplicate_filter]
                skipped_samples.append(skipped_dataframe)
                print(f"Removed {len(duplicate_filter[duplicate_filter])} duplicate samples")

            too_short_chosen_filter = dataframe.apply(lambda x: len(x["chosen"]) < self.minimum_response_length, axis=1)
            too_short_rejected_filter = dataframe.apply(
                lambda x: len(x["rejected"]) < self.minimum_response_length, axis=1
            )
            too_short_filter = too_short_chosen_filter | too_short_rejected_filter
            dataframe.loc[too_short_filter, "skip_reason"] = "too short"
            skipped_dataframe = dataframe[too_short_filter]
            skipped_samples.append(skipped_dataframe)
            print(f"Removed {len(too_short_filter[too_short_filter])} too short responses")
            dataframe = dataframe[~too_short_filter]

            encoded_prompts = self.tokenizer(
                dataframe["prompt"].tolist(),
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            dataframe["encoded_prompt"] = encoded_prompts["input_ids"]
            dataframe["prompt_len"] = dataframe["encoded_prompt"].apply(lambda x: len(x))

            too_long_filter = dataframe["prompt_len"] > self.max_prompt_length
            dataframe.loc[too_long_filter, "skip_reason"] = "too long"
            skipped_dataframe = dataframe[too_long_filter]
            skipped_samples.append(skipped_dataframe)
            print(f"Removed {len(too_long_filter[too_long_filter])} too long prompts")
            dataframe = dataframe[~too_long_filter]

            dataframe.loc[:, "encoded_chosen"] = self.tokenizer(
                dataframe["chosen"].tolist(),
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
            dataframe.loc[:, "encoded_rejected"] = self.tokenizer(
                dataframe["rejected"].tolist(),
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
            dataframe = dataframe.assign(chosen_input_ids=lambda x: x["encoded_prompt"] + x["encoded_chosen"])
            dataframe = dataframe.assign(rejected_input_ids=lambda x: x["encoded_prompt"] + x["encoded_rejected"])

            if not self.debug:
                dataframe = dataframe.drop(
                    columns=[
                        "prompt",
                        "chosen",
                        "rejected",
                        "encoded_prompt",
                        "encoded_chosen",
                        "encoded_rejected",
                    ]
                )

            all_skipped_samples = pd.concat(skipped_samples)
            print(f"Processed {len(dataframe)} samples, skipped {len(skipped_samples)} samples")
            output_data = {}
            if self.reward_model_split > 0:
                print(f"Using {self.reward_model_split * 100}% of the training data for the reward model training")
                num_reward_model_samples = int(len(dataframe) * self.reward_model_split)
                reward_model_samples = dataframe.iloc[:num_reward_model_samples]
                finetune_samples = dataframe.iloc[num_reward_model_samples:]
                output_data["reward_model"] = reward_model_samples
                output_data["finetune"] = finetune_samples
            else:
                output_data["finetune"] = dataframe
            if split == "train":
                print(f"Using {self.validation_split * 100}% of the training data for validation split")
                for data_name, data in output_data.items():
                    num_validation_samples = int(len(data) * self.validation_split)
                    validation_data = data[:num_validation_samples]
                    train_data = data[num_validation_samples:]
                    print(
                        f"Saving {len(train_data)} training samples and "
                        f"{len(validation_data)} validation samples for {data_name}"
                    )
                    train_data.reset_index(inplace=True, drop=True)
                    validation_data.reset_index(inplace=True, drop=True)
                    train_data.to_pickle(full_path / f"{data_name}_train.pkl")
                    validation_data.to_pickle(full_path / f"{data_name}_validation.pkl")
            else:
                for data_name, data in output_data.items():
                    print(f"Saving {len(data)} {split} samples for {data_name}")
                    data.reset_index(inplace=True, drop=True)
                    data.to_pickle(full_path / f"{data_name}_{split}.pkl")
            if self.save_skipped_examples:
                all_skipped_samples.to_json(full_path / f"{split}_skipped.json", orient="records", indent=4)
            time_stop = time.perf_counter()
            print(f"Finished processing {split} split in {time_stop - time_start:.2f} seconds")

        example_prompt_path = full_path / "example_prompt.pt"
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


class JsonProcessor(DataProcessor):
    def __init__(self, file_path: str, *args, **kwargs):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Json file {file_path} does not exist")
        with open(file_path, "r") as f:
            self.json_data = json.load(f)
        super().__init__(*args, **kwargs)

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        return sample["prompt"]

    def get_chosen(self, sample: Dict[str, Any]) -> str:
        return sample["chosen"]

    def get_rejected(self, sample: Dict[str, Any]) -> str:
        return sample["rejected"]
