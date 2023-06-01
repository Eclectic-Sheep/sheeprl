import torch
from torch.nn.utils.rnn import pad_sequence


class SFTCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = [item["input_ids"].type(torch.int64) for item in batch]
        targets = [item["targets"].type(torch.int64) for item in batch]

        # Use PyTorch's pad_sequence function
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_value)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.ignore_index)
        attention_mask = input_ids.ne(self.pad_value).type(torch.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
        }


class RMCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = [item["chosen_input_ids"].type(torch.int64) for item in batch]
        input_ids += [item["rejected_input_ids"].type(torch.int64) for item in batch]
        # Use PyTorch's pad_sequence function
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_value)
        attention_mask = input_ids.ne(self.pad_value).type(torch.int64)

        return {
            "chosen_input_ids": input_ids[: len(batch)],
            "rejected_input_ids": input_ids[len(batch) :],
            "chosen_attention_mask": attention_mask[: len(batch)],
            "rejected_attention_mask": attention_mask[len(batch) :],
        }


class LeftPadCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = [item["prompt_input_ids"].type(torch.int64) for item in batch]

        # Use PyTorch's pad_sequence function
        reversed_input_ids = [i.flip(dims=[0]) for i in input_ids]
        input_ids = pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.pad_value).flip(dims=[1])
        attention_mask = input_ids.ne(self.pad_value).type(torch.int64)

        return {
            "prompt_input_ids": input_ids,
            "prompt_attention_mask": attention_mask,
        }
