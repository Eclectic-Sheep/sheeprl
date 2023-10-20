from typing import Any, Dict

from sheeprl.algos.rlhf.data import DataProcessor


class SummarizationData(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        return sample["prompt"] + "\nTL;DR: "

    def get_chosen(self, sample: Dict[str, Any]) -> str:
        return sample["chosen"][8:]  # remove "TL;DR: "

    def get_rejected(self, sample: Dict[str, Any]) -> str:
        return sample["rejected"][8:]  # remove "TL;DR: "

    def wrap_prompt(self, prompt: str, **kwargs) -> str:
        subreddit = kwargs["subreddit"] if "subreddit" in kwargs else "TechSupport"
        title = kwargs["title"] if "title" in kwargs else "How to fix my laptop?"
        return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {prompt}\nTL;DR: "

    def get_example_prompt(self) -> str:
        prompt = "Hello everyone, I've been having some trouble with my laptop and I was hoping someone could " \
        "help me out. I've had this laptop for about 2 years. Recently, it has been running really slow " \
        "and it takes forever to load anything. I've tried running virus scans and deleting unnecessary files, " \
        "but nothing seems to be working. Sometimes, it freezes completely and I have to restart it. " \
        "One thing that I have noticed is that the fan on my laptop seems to be running constantly and sometimes " \
        "it's quite loud, even when I'm not doing anything particularly demanding on the laptop. I'm not sure if " \
        "this is related to the performance issues, but it's something that I thought might be worth mentioning. " \
        "I'm really hoping that someone can help me figure out what's causing these problems and what I can do to " \
        "fix them. I don't have a lot of experience with troubleshooting hardware issues, so any advice or guidance " \
        "would be greatly appreciated! Does anyone have any ideas for what I can do to fix this?"
        return prompt
