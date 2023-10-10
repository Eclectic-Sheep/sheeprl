# PPO recurrent

We developed a recurrent version of the PPO algorithm, which is able to solve partially observable environments.

The algorithm is implemented in the `ppo_recurrent.py` file, while the agent is implemented in the `agent.py` file. The `utils.py` file contains some useful functions for the algorithm.

# Algorithm

The algorithm is inspired by the [Generalization, Mayhems and Limits in Recurrent Proximal Policy
Optimization](https://arxiv.org/abs/2205.11104) paper. In particular, the steps in the algorithm pseudo-code are the following:

1. Initialize recurrent states to zero and retrieve the first observation from the environment
2. Collect rollout experiences for `T` steps. When a done is encountered the recurrent states are reset to zero
3. Compute returns and advantages
4. Split collected data into multiple episodes, where a new episode starts when the `done` or `terminated` flag is true
5. Split every episode into chunks of length at most `per_rank_sequence_length`
6. Zero-pad every sequence to be all of the same length
7. Optimize PPO loss for `algo.update_epochs` epochs, creating `per_rank_num_batches` random mini-batches of sequences
8. Repeat from 2. until convergence
## Agent

We needed the creation of the agent in a separate class to make things easier. Indeed, in the forward, the agent needs to process data sequentially (since it uses a recurrent neural network). It was therefore impractical to implement the actor and critic `torch.nn.Module` classes using the utility building blocks as we did for `sheeprl/algos/ppo/ppo.py`.

The agent is also equipped with some utility methods that make it easier to only call the actor or the critic network's forward method.

## Algorithm

For the rest, the algorithm is the same as in `sheeprl/algos/ppo/ppo.py`.