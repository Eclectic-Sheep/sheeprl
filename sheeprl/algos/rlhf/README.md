# Reinforcement Learning with Human Feedback

In this folder, we have implementation of multiple algorithms for reinforcement learning with human feedback. The algorithms are:

- Supervised Fine-Tuning (SFT)
- Reward Modelling (RM)
- Proximal Policy Optimization (PPO)
- Direct Policy Optimization (DPO)

Currently, we are using transformer based large language models from [transformers](https://github.com/huggingface/transformers) library.

> **Note**
>
> This algorithm section of SheepRL requires optional dependencies for RLHF. To install all the optional dependencies one can run `pip install .[rlhf]`.

For more detailed explanation, please refer to our how-to guide [here](https://github.com/Eclectic-Sheep/sheeprl/blob/main/howto/rlhf.md).