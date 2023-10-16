# Reinforcement Learning with Human Feedback

<p align="center">
  <img src="../assets/images/rlhf_overview.png" style="width:80%">
</p>

Reinforcement Learning with Human Feedback (RLHF) is a technique that combines traditional reinforcement learning (RL) with human decisions to train more effective and safe policies. Instead of solely relying on reward signals obtained from the environment, RLHF integrates feedback from humans to guide the learning process. With RLHF, we can have approximated reward signals that are not crafted manually, but rather learned from human judgments.

Moreover, we have implemented Direct Policy Optimization for aligning models to human preferences without actual reward modelling.

In this algorithm section, we are using transformer based large language models from [transformers](https://github.com/huggingface/transformers) library and implement each different algorithm with using [fabric](https://lightning.ai/docs/fabric/stable/) and [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/).

> **Note**
>
> This algorithm section of SheepRL requires optional dependencies for RLHF. To install all the optional dependencies one can run `pip install .[rlhf]`.

## Details

- Every experiment script creates its own folder and saves all the checkpoints, logs, and metrics to that folder. All configuration parameters are saved as YAML file with Hydra.
- Datasets provided in this repository might not be the original ones with correct train/validation/test splits. They are provided for quick experimenting. Users can implement their own DataProcessor classes and use them with the scripts. The DataProcessor class helps you to customize settings for the data and saves the data in a format that is compatible with the training script.
- According to the GPU capacity, the user can change the batch size and activate gradient accumulation. For example passing `algo.mini_batch_size=4 algo.micro_batch_size=2` as parameter to CLI will result in 2 samples per device and 2 gradient accumulation steps per device. The total batch size will be 4 samples per device.

## Datasets

Currently we have two example datasets suitable for RLHF:

- Helpful - Harmless
- Summarization

> **Note**
>
> For setting up all data related configurations, please check `sheeprl/algo/rlhf/config_store/data.py` file.

### Helpful - Harmless

This dataset contains human-assistant dialogue samples for training a helpful and harmless agents. The dataset contains chosen and rejected pairs. Chosen pairs are the ones that are considered helpful and harmless by human annotators. Rejected pairs are the ones that are considered harmful or not helpful by human annotators. 

### Summarization

For summarization task, we are using [CarperAI/openai_summarize_comparisons](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons). This dataset is cleaned and prepared version of original OpenAI summarization dataset. Here is an example script to download and process the dataset.


## Algorithms

> **Note**
>
> By default all scripts use OPT-350m model with helpful harmless dataset.
model can be updated using `model=<model_config_name>` argument and dataset can be updated using `data=<dataset_name>` argument from CLI.


### Supervised Fine-Tuning (SFT)

SFT is a supervised learning approach to train a language model to learn generating similar responses given prompts in the dataset. The model is trained with cross entropy loss. The algorithm uses `chosen` texts to finetune the model. To run SFT training:

```bash
python sheeprl.py exp=sft data=helpful_harmless
```

### Reward Modeling (RM)

Reward modeling method learns a reward function from human choices. The model compares `chosen` and `rejected` responses given the same prompt and learns to predict which one is more likely to be chosen by human. The RM algorithm can be run as follows:

```bash
python sheeprl.py exp=rlhf_rm data=helpful_harmless
```

### Reinforcement Learning (RL/PPO)

Last stage of RLHF is reinforcement learning. The model is trained with Proximal Policy Optimization (PPO) algorithm. The reward signal is approximated with RM model. This training requires already completed SFT and RM training stages and their final checkpoints. The PPO algorithm training can be instantiated as follows:

```bash
python sheeprl.py exp=rlhf_ppo data=helpful_harmless \
algo.sft_experiment_dir=<path_to_sft_experiment_dir> \
algo.rm_experiment_dir=<path_to_rm_experiment_dir>
```

### DPO (Direct Policy Optimization)

This algorithm works without requiring reward modelling stage. It simplifies the process of fine-tuning. The dataset structure for DPO algorithm is similar to other RLHF algorithms and DPO requires SFT stage completed for working properly. Models are fine-tuned to maximize the likelihood of generating chosen completions and minimize the likelihood of generating rejected ones. The DPO algorithm can be run as follows:

```bash
python sheeprl.py exp=rlhf_dpo data=helpful_harmless \
algo.sft_experiment_dir=<path_to_sft_experiment_dir>
```

## Frequently Asked Questions


### How to add a new dataset?

Currently, only way to add new dataset is extending `DataProcessor` class defined under `sheeprl.algos.rlhf.data.data.DataProcessor`.

### Can we integrate Lit-GPT models?

Currently we are using transformers library for language models. We are planning to integrate Lit-GPT models in the future. Here are few things have to be done for integrating Lit-GPT models:
- Lightning and Pytorch 2.1.x support
- Models have to have `generate` method for now. Otherwise, we have to implement `generate` method for each model.

### How to add a new model architecture?

All models are defined as `dataclasses` under `config_store/model.py` file. The OPT-350m model is used as default if no model is specified. To add a new model, one can add a new `dataclass` to the file. For example, to add Phi 1.5B model, one can add the following code:

```python
@dataclass
class PhiConfig(ModelConfig):
    name: str = "microsoft/phi-1_5"
    # this model has to download the source code from their page.
    library_cfg: HuggingFaceConfig = HuggingFaceConfig(trust_remote_code=True)
    lora_cfg: Optional[LORAConfig] = LORAConfig(targets="('Wqkv','out_proj')")
    # This model cannot use attention mask during the training
    # https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_mixformer_sequential.py#L756
    use_attention_mask: bool = False
```

Please pay attention to the `use_attention_mask` parameter. This kind of extra configurations can be extended to other models as well.


Now you need to register the config:

```python
def register_model_configs(cs: ConfigStore) -> None:
   ...
    cs.store(
        group="model",
        name="phi",
        node=PhiConfig,
    )
```


### How to use LoRA(Low-Rank Adaptation)

LoRA is a method to adapt a pretrained language model to a new task with updating a fraction of the parameters. We have adapted [MinLora](https://github.com/cccntu/minlora) in to our training script. In this way, we can add multiple LORA heads to pretrained language model for PPO or DPO algorithms. It can be enabled as follows:

```bash
python sheeprl.py exp=<any_rlhf_experiment> model.finetune_mode=LORA
```

Optionally LoRA parameters can be set as follows:

```bash
python sheeprl.py exp=<any_rlhf_experiment> model.finetune_mode=LORA \
model.lora_cfg.rank=32 \
model.lora_cfg.dropout=0.1
```

## Acknowledgements

This work and the code developed for the task is a long educational and experimental journey. Please ask us about anything you need or not clear on GitHub. It will be even more then welcomed if you like to contribute. We would like to thank the following works for their contributions to the field and inspiring us to develop this work.

### Libraries

- [TRL](https://github.com/lvwerra/trl)
- [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md)
- [TRLX](https://github.com/CarperAI/trlx)
- [Lit-Llama](https://github.com/Lightning-AI/lit-llama)
- [Lit-GPT](https://github.com/Lightning-AI/lit-gpt)
- [MinLora](https://github.com/cccntu/minlora)
- [MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)
- [Original DPO](https://github.com/eric-mitchell/direct-preference-optimization)

### Blog Posts

- [StackLLaMa](https://huggingface.co/blog/stackllama)
- [Implementing RLHF: Learning to Summarize with trlX](https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-trlX--VmlldzozMzAwODM2)
- [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)
- [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl)

### Research Articles

- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [Training language models to follow instructions](https://arxiv.org/abs/2203.02155)
- [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales](https://arxiv.org/abs/2308.01320)
- [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964)
- [LLAMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)