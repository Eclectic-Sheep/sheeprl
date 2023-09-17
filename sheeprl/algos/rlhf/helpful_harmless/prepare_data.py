import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from sheeprl.algos.rlhf.args import TextDataArgs
from sheeprl.algos.rlhf.data import DataProcessor
from sheeprl.utils.parser import HfArgumentParser

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


class HHDataProcessor(DataProcessor):
    def __init__(self, dataset_name: str = "Dahoas/full-hh-rlhf", *args, **kwargs):
        super().__init__(*args, dataset_name=dataset_name, **kwargs)

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


if __name__ == "__main__":
    parser = HfArgumentParser([TextDataArgs])
    dataclasses = parser.parse_args_into_dataclasses()
    data_args: TextDataArgs = dataclasses[0]
    data_processor = HHDataProcessor(**asdict(data_args))
    data_processor.process()
    with open(Path(data_args.destination_dir) / "args.json", "w") as f:
        json.dump(asdict(data_args), f, indent=4)
