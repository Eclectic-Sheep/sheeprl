# PPO algorithm
PPO is an on-policy algorithm that uses a clipped objective function to update the policy.

From the interaction with the environment, it collects trajectories of *observations*, *actions*, *rewards*, *values*, *logprobs* and *dones*. These trajectories are stored in a buffer, and used to train the policy and value networks.

The algorithm is the `PPO` class in the `ppo.py` file, run by the training loop shared by all the algorithms (`sheeprl.core.loop.run`). Every iteration plays `algo.rollout_steps` steps in every environment, then trains for `algo.update_epochs` epochs: in every epoch the rollout is split in random minibatches of `algo.per_rank_batch_size` steps, and every minibatch gives one optimizer step on the *policy loss*, *value loss* and *entropy loss*.

From the rewards, *returns* and *advantages* are estimated with GAE. The *returns*, together with the values stored in the buffer and the values from the updated critic, are used to compute the *value loss*. 

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
    v_loss_unclipped = (new_values - returns) ** 2
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    v_loss_clipped = (v_clipped - returns) ** 2
    return reduce_loss(torch.max(v_loss_unclipped, v_loss_clipped), reduction)
```

Both are the squared error without a factor ½, as in Stable-Baselines3: `algo.vf_coef` weighs this scale (CleanRL halves
the squared error, so its `vf_coef` is twice ours).

Advantages and logprobs are used to compute the *policy loss*, using also the logprobs from the updated model.

```python
def policy_loss(
    new_logprobs: Tensor,
    logprobs: Tensor,
    advantages: Tensor,
    clip_coef: float,
    reduction: str = "mean",
) -> Tensor:
    logratio = new_logprobs - logprobs
    ratio = logratio.exp()

    pg_loss1 = advantages * ratio
    pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = -torch.min(pg_loss1, pg_loss2)
    return reduce_loss(pg_loss, reduction)
```

The losses are reduced as set by `algo.loss_reduction`. The learning rate, the clipping coefficient and the entropy coefficient can be linearly annealed to zero during the training (`algo.anneal_lr`, `algo.anneal_clip_coef` and `algo.anneal_ent_coef`).
