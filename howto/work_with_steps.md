# Work with steps
In this document we want to discuss about the hyper-parameters which refer to the concept of step.
There are various ways to interpret it, so it is necessary to clearly specify which iterpretation we give to that concept.

## Policy steps
We start from the concept of *policy steps*: a policy step is the selection of an action to perform an environment step. In other words, it is called *policy step* when the actor takes in input an observation and choose an action and then the agent performs an environment step.

> **Note**
>
> The environment step is the step performed by the environment: the environment takes in input an action and computes the next observation and the next reward.

Now we have introduced the concept of policy step, it is necessary to clarify some aspects:
1. When there are more parallel environments, the policy step is proportional with the number of parallel environments. E.g, if there are $m$ environments, then the agent has to choose $m$ actions and each environment performs an environment step: this mean that **m policy steps** are performed.
2. When there are more parallel processes, the policy step it is proportional with the number of parallel processes. E.g, let us assume that there are $n$ processes each one containing one single environment: the $n$ actors select an action and perform a step in the environment, so, also in this case **n policy steps** are performed.

To generalize, in a case with $n$ processes, each one with $m$ environments, the policy steps increase by $n \cdot m$ at each iteration.

The hyper-parameters which refer to the *policy steps* are:
* `total_steps`: the total number of policy steps to perform in an experiment.
* `exploration_steps`: the number of policy steps in which the agent explores the environment in the P2E algorithms.
* `max_episode_steps`: the maximum number of policy steps an episode can last ($\text{max\_steps}$). This means that if you decide to have an action repeat greater than one ($\text{act\_repeat} > 1$), then you can perform a numeber of environment steps equal to: $\text{env\_steps} = \text{max\_steps} \cdot \text{act\_repeat}$.
* `learning_starts`: how many policy steps the agent has to perform before starting the training.
* `train_every`: how many policy steps the agent has to perform between one training and the next.

## Gradient steps
A *gradient step* consists of an update of the parameters of the agent, i.e., a call of the *train* function. The gradient step is proportional to the number of parallel processes, indeed, if there are $n$ parallel processes, the call of the *train* method will increase by $n$ the gradient step.

The hyper-parameters which refer to the *policy steps* are:
* `algo.per_rank_gradient_steps`: the number of gradient steps per rank to perform in a single iteration.
* `algo.per_rank_pretrain_steps`: the number of gradient steps per rank to perform in the first iteration.