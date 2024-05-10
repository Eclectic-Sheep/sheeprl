# Work with steps
In this document, we want to discuss the hyper-parameters that refer to the concept of step.
There are various ways to interpret it, so it is necessary to clearly specify how to interpret it.

## Policy steps
We start from the concept of *policy step*: a policy step is the particular step in which the policy selects the action to perform in the environment, given an observation received by it.

> [!NOTE]
>
> The environment step is the step performed by the environment: the environment takes in input an action and computes the next observation and the next reward. This means that the environment steps are taking into consideration also the **action repeat**, which is a value greater or equal to 0 that specifies how many times an action has to be played (repeated by the environment) independently by the observations received. 

Now that we have introduced the concept of *policy step*, it is necessary to clarify some aspects:

1. When there are multiple parallel environments, the policy step is proportional to the number of parallel environments. E.g., if there are $m$ environments, then the actor has to choose $m$ actions and each environment performs an environment step: this means that $\bold{m}$ **policy steps** are performed.
2. When there are multiple parallel processes (i.e. the script has been run with `python sheeprl fabric.devices>=2 ...`), the policy step it is proportional to the number of parallel processes. E.g., let us assume that there are $n$ processes each one containing one single environment: the $n$ actors select an action, and a (per-process) step in the environment is performed. In this case $\bold{n}$ **policy steps** are performed.

In general, if we have $n$ parallel processes, each one with $m$ independent environments, the policy step increases **globally** by $n \cdot m$ at each iteration.

The hyper-parameters that refer to the *policy steps* are:

* `total_steps`: the total number of policy steps to perform in an experiment. Effectively, this number will be divided in each process by $n \cdot m$ to obtain the number of iteration steps to be performed by each of them.
* `exploration_steps`: the number of policy steps in which the agent explores the environment in the P2E algorithms.
* `max_episode_steps`: the maximum number of policy steps an episode can last (`max_steps`); when this number is reached a `truncated=True` is returned by the environment. This means that if you decide to have an action repeat greater than one (`action_repeat > 1`), then the environment performs a maximum number of steps equal to: `env_steps = max_steps * action_repeat`$.
* `learning_starts`: how many policy steps the agent has to perform before starting the training. During the first `learning_starts` steps the buffer is pre-filled with random actions sampled by the environment.

## Gradient steps
A *gradient step* consists of an update of the parameters of the agent, i.e., a call of the *train* function. The gradient step is proportional to the number of parallel processes, indeed, if there are $n$ parallel processes, `n * per_rank_gradient_steps` calls to the *train* method will be executed.

The hyper-parameters which refer to the *gradient steps* are:
* `algo.per_rank_pretrain_steps`: the number of gradient steps per rank to perform in the first iteration.
* `algo.replay_ratio`: the `replay-ratio` is the ratio between the gradient steps and the policy steps played by the agent. The higher the replay-ratio the more sample-efficient the agent should be. The replay-ratio is a global hyper-parameters that affects only the off-policy algorithms like SAC or Dreamer and must be a float greater than zero. For example, a replay-ratio of 0.5 means that the agent will train itself for 1 gradient step every 2 policy steps. The **replay ratio does not account for both the environment's action-repeat and the `algo.learning_starts`** 