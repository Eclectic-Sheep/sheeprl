# PPO recurrent
We developed a recurrent version of the PPO algorithm, which is able to solve partially observable environments. The algorithm is based on the [PPO algorithm](https://arxiv.org/abs/1707.06347) and its [implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

The algorithm is implemented in the `ppo_recurrent.py` file, while the agent is implemented in the `agent.py` file. The `utils.py` file contains some useful functions for the algorithm.

## Agent
We needed the creation of the agent in a separate class to make things easier. Indeed, in the forward, the agent needs to process data sequentially (since it uses a recurrent neural network). It was therefore impractical to implement the actor and critic `torch.nn.Module` classes using the utility building blocks as we did for `fabricrl/algos/ppo/ppo.py`.

The agent is also equipped with some utility methods that make it easier to only call the actor or the critic network's forward method.

## Algorithm
For the rest, the algorithm is the same as in `fabricrl/algos/ppo/ppo.py`.