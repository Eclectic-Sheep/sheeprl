from typing import Any, Dict

from sheeprl.algos.rlhf.data.base import DataProcessor


class HelpfulHarmlessData(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        return sample["prompt"]

    def get_chosen(self, sample: Dict[str, Any]) -> str:
        return sample["chosen"]

    def get_rejected(self, sample: Dict[str, Any]) -> str:
        return sample["rejected"]

    def wrap_prompt(self, prompt: str) -> str:
        return "\n\nHuman: " + prompt + "\n\nAssistant: "

    def get_example_prompt(self) -> str:
        return "How does the computer work?"
