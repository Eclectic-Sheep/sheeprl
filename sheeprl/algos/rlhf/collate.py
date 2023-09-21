import torch
from torch.nn.utils.rnn import pad_sequence


class SFTCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = [item["chosen_input_ids"].type(torch.int64) for item in batch]
        targets = [item["chosen_masked_targets"].type(torch.int64) for item in batch]

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
        input_ids = [item["chosen_input_ids"].type(torch.int64) for item in batch]
        input_ids += [item["rejected_input_ids"].type(torch.int64) for item in batch]
        targets = [item["chosen_masked_targets"].type(torch.int64) for item in batch]
        targets += [item["rejected_masked_targets"].type(torch.int64) for item in batch]
        # Use PyTorch's pad_sequence function
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_value)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.ignore_index)
        attention_mask = input_ids.ne(self.pad_value).type(torch.int64)

        return {
            "chosen_input_ids": input_ids[: len(batch)],
            "rejected_input_ids": input_ids[len(batch) :],
            "chosen_attention_mask": attention_mask[: len(batch)],
            "rejected_attention_mask": attention_mask[len(batch) :],
            "chosen_masked_targets": targets[: len(batch)],
            "rejected_masked_targets": targets[len(batch) :],
        }


class LeftPadCollate:
    def __init__(self, dim=1, pad_value=0, ignore_index=-1):
        self.dim = dim
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = [item["chosen_input_ids"].type(torch.int64)[: item["prompt_len"]] for item in batch]
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
            input_ids = item["chosen_input_ids"].type(torch.int64)
            input_ids_list.append(input_ids)
            prompt_len = item["prompt_len"]
            prompt_input_ids_list.append(input_ids[:prompt_len])
            targets_list.append(item["targets"].type(torch.int64))
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
