# PPO algorithm
PPO is an on-policy algorithm that uses a clipped objective function to update the policy.

From the interaction with the environment, it collects trajectories of *observations*, *actions*, *rewards*, *values*, *logprobs* and *dones*. These trajectories are stored in a buffer, and used to train the policy and value networks.

Indeed, the training loop consists in sampling a batch of trajectories from the buffer, and computing the *policy loss*, *value loss* and *entropy loss*.

From the rewards, *returns* and *advantages* are estimated. The *returns*, together with the values stored in the buffer and the values from the updated critic, are used to compute the *value loss*. 

```python
def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
) -> Tensor:
    if not clip_vloss:
        values_pred = new_values
    else:
        values_pred = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    return mse_loss(values_pred, returns)
```

Advantages and logprobs are used to compute the *policy loss*, using also the logprobs from the updated model.

```python
def policy_loss(dist: torch.distributions.Distribution, batch: TensorDict, clip_coef: float) -> Tensor:
    new_logprobs = dist.log_prob(batch["actions"])
    logratio = new_logprobs - batch["logprobs"]
    ratio = logratio.exp()
    advantages: Tensor = batch["advantages"]

    pg_loss1 = advantages * ratio
    pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.min(pg_loss1, pg_loss2).mean()
    return pg_loss
```

