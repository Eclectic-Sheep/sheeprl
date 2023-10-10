from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


def list_to_tensor(list_item: List[int], dtype=torch.int64):
    return torch.tensor(list_item, dtype=dtype)


class SFTCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids, targets = [], []
        for item in batch:
            prompt_len = item["prompt_len"]
            input_ids.append(list_to_tensor(item["chosen_input_ids"]))
            target = list_to_tensor([self.ignore_index] * prompt_len + item["chosen_input_ids"][prompt_len:])
            targets.append(target)

        # Use PyTorch's pad_sequence function
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_value)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.ignore_index)
        attention_mask = input_ids.ne(self.pad_value).type(torch.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
        }


class CompareCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        chosen_input_ids, chosen_targets = [], []
        rejected_input_ids, rejected_targets = [], []
        for item in batch:
            prompt_len = item["prompt_len"]
            chosen_input_ids.append(list_to_tensor(item["chosen_input_ids"]))
            rejected_input_ids.append(list_to_tensor(item["rejected_input_ids"]))
            chosen_target = list_to_tensor([self.ignore_index] * prompt_len + item["chosen_input_ids"][prompt_len:])
            chosen_targets.append(chosen_target)
            rejected_targets.append(
                list_to_tensor([self.ignore_index] * prompt_len + item["rejected_input_ids"][prompt_len:])
            )
        input_ids = chosen_input_ids + rejected_input_ids
        targets = chosen_targets + rejected_targets

        # Use PyTorch's pad_sequence function
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_value)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.ignore_index)
        attention_mask = input_ids.ne(self.pad_value).type(torch.int64)

        return {
            "chosen_input_ids": input_ids[: len(batch)],
            "rejected_input_ids": input_ids[len(batch) :],
            "chosen_attention_mask": attention_mask[: len(batch)],
            "rejected_attention_mask": attention_mask[len(batch) :],
            "chosen_targets": targets[: len(batch)],
            "rejected_targets": targets[len(batch) :],
        }


class LeftPadCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = [list_to_tensor(item["chosen_input_ids"])[: item["prompt_len"]] for item in batch]
        # Use PyTorch's pad_sequence function
        # convert into left padding
        reversed_input_ids = [i.flip(dims=[0]) for i in input_ids]
        input_ids = pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.pad_value).flip(dims=[1])
        attention_mask = input_ids.ne(self.pad_value).type(torch.int64)

        return {
            "prompt_input_ids": input_ids,
            "prompt_attention_mask": attention_mask,
        }


class EvaluateCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids_list = []
        prompt_input_ids_list = []
        targets_list = []
        attention_mask = []
        prompt_len_list = []
        for item in batch:
            input_ids = list_to_tensor(item["chosen_input_ids"])
            input_ids_list.append(input_ids)
            prompt_len = item["prompt_len"]
            prompt_input_ids_list.append(input_ids[:prompt_len])
            target = list_to_tensor([self.ignore_index] * prompt_len + item["chosen_input_ids"][prompt_len:])
            targets_list.append(target)
            prompt_len_list.append(prompt_len)

        # Use PyTorch's pad_sequence function
        padded_input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_value)
        padded_targets_list = pad_sequence(targets_list, batch_first=True, padding_value=self.ignore_index)
        attention_mask = padded_input_ids_list.ne(self.pad_value).type(torch.int64)

        reversed_input_ids_list = [sample.flip(dims=[0]) for sample in prompt_input_ids_list]
        padded_reversed_input_ids_list = pad_sequence(
            reversed_input_ids_list, batch_first=True, padding_value=self.pad_value
        ).flip(dims=[1])
        prompt_attention_mask = padded_reversed_input_ids_list.ne(self.pad_value).type(torch.int64)

        return {
            "input_ids": padded_input_ids_list,
            "attention_mask": attention_mask,
            "prompt_input_ids": padded_reversed_input_ids_list,
            "prompt_attention_mask": prompt_attention_mask,
            "targets": padded_targets_list,
            "prompt_len": prompt_len_list,
        }
