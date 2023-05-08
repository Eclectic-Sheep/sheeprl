# Learning with CNNs
Atari games are harder to solve then basic gym environments, since they rely on visual information. This implies using a CNN model instead of an MLP model. Despite this difference, the rest of the training reimains the same.

In this section, we will learn how to train an agent using a CNN model on Atari games.

The code for this section is available in `algos/ppo/ppo_atari.py`.

## Step by step
We start from `ppo_decoupled.py` and copy its code in the new `ppo_atari.py` file.

