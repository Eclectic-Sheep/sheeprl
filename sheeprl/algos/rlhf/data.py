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

        # attention_mask
        attention_mask = [torch.ones_like(x) for x in input_ids]

        # Use PyTorch's pad_sequence function
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_value)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.ignore_index)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
        }
