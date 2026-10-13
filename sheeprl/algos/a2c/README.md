# A2C algorithm
Advantage-Actor-Critic (A2C) is an on-policy algorithm that uses the standard policy gradient algorithm, scaled by the advantages, to update the policy.

From the interaction with the environment, it collects trajectories of *observations*, *actions*, *rewards*, *values*, *logprobs* and *dones*. These trajectories are stored in a buffer, and used to train the policy and value networks.

The algorithm is the `A2C` class in the `a2c.py` file, run by the training loop shared by all the algorithms (`sheeprl.core.loop.run`); the agent is the one of PPO (`sheeprl/algos/ppo/agent.py`). Every iteration plays `algo.rollout_steps` steps in every environment, then does a single optimizer step on the whole rollout: the rollout is split in minibatches of `algo.per_rank_batch_size` steps, the gradients of the *policy loss*, *value loss* and *entropy loss* of every minibatch are accumulated, averaged across the processes and applied once. By default (`algo.loss_reduction=sum`) the losses are summed over the steps, so the gradient is the one of the sum over the whole rollout. The learning rate is not annealed: `algo.anneal_lr` has no effect.

From the rewards, *returns* and *advantages* are estimated with GAE. The *returns* and the values from the updated critic are used to compute the *value loss*: A2C uses the value loss of PPO (`sheeprl/algos/ppo/loss.py`) without clipping, i.e. the squared error of the values from the returns.

```python
def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
    reduction: str = "mean",
) -> Tensor:
    if not clip_vloss:
        return F.mse_loss(new_values, returns, reduction=reduction)
    ...
```

Advantages and the logprobs from the updated model are used to compute the *policy loss*.

```python
def policy_loss(
    logprobs: Tensor,
    advantages: Tensor,
    reduction: str = "mean",
) -> Tensor:
    pg_loss = -(logprobs * advantages)
    reduction = reduction.lower()
    if reduction == "none":
        return pg_loss
    elif reduction == "mean":
        return pg_loss.mean()
    elif reduction == "sum":
        return pg_loss.sum()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")
```

