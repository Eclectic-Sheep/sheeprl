# Work with steps
In this document, we want to discuss the hyper-parameters that refer to the concept of step.
There are various ways to interpret it, so it is necessary to clearly specify how to interpret it.

## Policy steps
We start from the concept of *policy step*: a policy step is the particular step in which the policy selects the action to perform in the environment, given an observation received by it.

> [!NOTE]
>
> The environment step is the step performed by the environment: the environment takes in input an action and computes the next observation and the next reward. This means that the environment steps are taking into consideration also the **action repeat**, which is an integer greater or equal to 1 (`env.action_repeat`, values lower than 1 are set to 1) that specifies how many times an action has to be played (repeated by the environment) independently by the observations received. 

Now that we have introduced the concept of *policy step*, it is necessary to clarify some aspects:

1. When there are multiple parallel environments, the policy step is proportional to the number of parallel environments. E.g., if there are $m$ environments, then the actor has to choose $m$ actions and each environment performs an environment step: this means that $\bold{m}$ **policy steps** are performed.
2. When there are multiple parallel processes (i.e. the script has been run with `python sheeprl fabric.devices>=2 ...`), the policy step it is proportional to the number of parallel processes. E.g., let us assume that there are $n$ processes each one containing one single environment: the $n$ actors select an action, and a (per-process) step in the environment is performed. In this case $\bold{n}$ **policy steps** are performed.

In general, if we have $n$ parallel processes, each one with $m$ independent environments, the policy step increases **globally** by $n \cdot m$ at each step of the environments.

## Iterations
Every algorithm is trained by the same training loop (`sheeprl.core.loop.run`), which repeats an *iteration* until the end of the experiment: every environment plays $s$ steps, then the agent is trained, then the metrics are logged and a checkpoint is saved, if it is time to. The steps $s$ played by every environment in an iteration are:

* `algo.rollout_steps` for the on-policy algorithms (A2C, PPO and PPO Recurrent);
* $1$ for the off-policy algorithms (SAC, DroQ, SAC-AE, Dreamer-V1, Dreamer-V2, Dreamer-V3 and the Plan2Explore ones).

So, an iteration increases the policy step by $n \cdot m \cdot s$. The iterations, the policy steps and the gradient steps of an experiment are counted by the `TrainSchedule` class (`sheeprl/core/schedule.py`).

The hyper-parameters that refer to the *policy steps* are:

* `algo.total_steps`: the total number of policy steps to perform in an experiment. Effectively, this number will be divided by $n \cdot m \cdot s$ (rounded down) to obtain the number of iterations to be performed by each process.
* `env.max_episode_steps`: the maximum number of policy steps an episode can last (`max_steps`); when this number is reached a `truncated=True` is returned by the environment. This means that if you decide to have an action repeat greater than one (`action_repeat > 1`), then the environment performs a maximum number of steps equal to: `env_steps = max_steps * action_repeat`. If it is `null` or not greater than 0, SheepRL does not add any limit to the episodes (the environment can still have its own).
* `algo.learning_starts`: how many policy steps the agent of an off-policy algorithm has to perform before starting the training. During the first `learning_starts` steps (rounded down to whole iterations) the buffer is pre-filled with random actions sampled by the environment, and the training starts at the end of the last of these iterations. The Dreamer and Plan2Explore agents, which train on sequences of `algo.per_rank_sequence_length` steps of every environment, start training only when every environment has played that many steps, also when `learning_starts` comes earlier (a warning says so; a dry run plays until then), and the replay buffer of every environment (`buffer.size // (env.num_envs * world_size)`) must hold a sequence. They do not play random actions in the MineDojo environments, and the Plan2Explore finetuning plays with the actor selected by `algo.player.actor_type` (by default the exploration one) instead of random actions.
* `metric.log_every` and `checkpoint.every`: the policy steps between two consecutive logging and checkpointing operations, respectively. Since they happen at the end of an iteration, if they are not a multiple of $n \cdot m \cdot s$, the metrics are logged and the checkpoints are saved at the nearest greater multiple of it. Check the [logs and checkpoints howto](./logs_and_checkpoints.md) for more information.

> [!NOTE]
>
> In the Plan2Explore algorithms, the exploration (e.g. `exp=p2e_dv3_exploration`) and the finetuning (e.g. `exp=p2e_dv3_finetuning`) are two different experiments, each one with its own `algo.total_steps`.

## Gradient steps
A *gradient step* consists of an update of the parameters of the agent, i.e., a call of the `train_step` method of the algorithm on a batch of data. The gradient steps are counted per process (`per_rank_gradient_steps`), indeed, if there are $n$ parallel processes, `n * per_rank_gradient_steps` calls to the `train_step` method will be executed.

The gradient steps of an iteration of the on-policy algorithms are decided by the algorithm: e.g., PPO trains for `algo.update_epochs` epochs over the minibatches of `algo.per_rank_batch_size` steps of the rollout, whereas A2C performs one gradient step per iteration.

The hyper-parameters which refer to the *gradient steps* of the off-policy algorithms are:
* `algo.replay_ratio`: the `replay-ratio` is the ratio between the gradient steps and the policy steps played by the agent. The higher the replay-ratio the more sample-efficient the agent should be. The replay-ratio is a global hyper-parameters that affects only the off-policy algorithms like SAC or Dreamer and must be a float greater than zero. For example, a replay-ratio of 0.5 means that the agent will train itself for 1 gradient step every 2 policy steps. With $n$ processes, every process performs `replay_ratio` gradient steps for each of its policy steps, so the global ratio is the same (it is logged as `Params/replay_ratio`). The **replay ratio does not account for both the environment's action-repeat and the `algo.learning_starts`**: it counts the policy steps from the start of the iteration in which the training starts.
* `algo.per_rank_pretrain_steps`: the gradient steps per process that the first training iteration performs in addition to the ones of the replay ratio, to pretrain the agent on the data collected during the first `algo.learning_starts` policy steps (the `pretrain` hyper-parameter of DreamerV1 and DreamerV2). A dry run does not perform them.

## Resume an experiment
When an experiment is resumed from a checkpoint (`checkpoint.resume_from=/path/to/checkpoint.ckpt`), it continues from the iteration that follows the one in which the checkpoint was saved (`iter_num` in the checkpoint): the policy steps continue from the ones of the checkpoint and the gradient steps from the `per_rank_gradient_steps` saved in it. The `algo.total_steps` and `algo.learning_starts` of the resumed experiment are the ones of the command used to resume it.

An off-policy algorithm whose replay buffer is saved in the checkpoint (`buffer.checkpoint=True`) continues as the stopped experiment: it does not play random actions again (it does only if it was stopped before `algo.learning_starts`) and the replay ratio continues from the one saved in the checkpoint (`ratio`). If the replay buffer is not saved in the checkpoint (`buffer.checkpoint=False`), the resumed experiment fills a new one, playing its policy (not random actions) for `algo.learning_starts` policy steps (rounded down to whole iterations), and then it trains as a new experiment does: the replay ratio counts the policy steps from the start of the iteration in which the training starts again.
