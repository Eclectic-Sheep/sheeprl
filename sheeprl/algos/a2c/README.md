# A2C algorithm
Advantage-Actor-Critic (A2C) is an on-policy algorithm that uses the standard policy gradient algorithm, scaled by the advantages, to update the policy.

From the interaction with the environment, it collects trajectories of *observations*, *actions*, *rewards*, *values*, *logprobs* and *dones*. These trajectories are stored in a buffer, and used to train the policy and value networks.

Indeed, the training loop consists in sampling a batch of trajectories from the buffer, and computing the *policy loss*, *value loss* and *entropy loss*, while accumulating the gradients over the trajectories collected during the interaction with the environment. By deafult it will sum the gradients over multiple batches, averaging them across all replicas on the last batch seen.

From the rewards, *returns* and *advantages* are estimated. The *returns*, together with the values stored in the buffer and the values from the updated critic, are used to compute the *value loss*. 

```python
def value_loss(
    values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
) -> Tensor:
    return mse_loss(values, returns)
```

Advantages and logprobs are used to compute the *policy loss*, using also the logprobs from the updated model.

```python
def policy_loss(logprobs: Tensor, advantages: Tensor) -> Tensor:
    pg_loss = -logprobs * advantages.detach()
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

