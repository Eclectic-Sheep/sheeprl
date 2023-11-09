# P2E Dv3
In this folder the [Plan2Explore algorithm](https://arxiv.org/abs/2005.05960) based on Dreamer V3 is implemented.

As P2E Dv1 and P2E Dv2, there is the possibility to perform *zero-shot* learning and *few-shot* learning by specifing the `exploration_steps` and `total_steps` arguments (from the configs). In particular, if `exploration_steps == total_steps` then only *zero-shot* learning is performed; otherwise, if `exploration_steps < total_steps`, then *zero-shot* learning and *few-shot* learning are performed.

In P2E_DV3 we added the possibility to use more critics for the exploration:
* The exploration critics are defined in the `algo.critics_exploration` config.
* It consists of a python dictionary that contains a pair key-critic_configs.
* Each critic_config has to contain: the weight to give to the advantages (if zero, then the critic is ignored), the reward to use (`intrinsic` or `task`).

> **Note**
>
> There must be at least one intrinsic critic (the reward type must be `intrinsic`)

The following example shows a possible configuration for the exploration critics:
```yaml
critics_exploration:
  intr:
    weight: 0.1
    reward_type: intrinsic
  extr:
    weight: 1.0
    reward_type: task
```