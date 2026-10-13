# SAC
Soft Actor Critic (SAC) is an off policy algorithm developed by Haarnoja et al. in [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905). It is an actor-critic algorithm that uses a stochastic policy, and a critic that estimates the state-action value function. The critic is trained to minimize the mean squared error between the estimated state-action value function and the target state-action value function. The policy is trained to maximize the expected return, while also maximizing the entropy of the policy. The entropy is maximized to encourage exploration.

The algorithm is the `SAC` class in the `sac.py` file, run by the training loop shared by all the algorithms (`sheeprl.core.loop.run`). Every iteration plays one step in every environment (with random actions until `algo.learning_starts`) and stores it in a replay buffer, then does `algo.replay_ratio` gradient steps per policy step, each one on its own batch sampled from the buffer: the critics, the actor and the entropy coefficient are updated in this order.

Losses are very simple to compute, and are implemented in the `loss.py` file. The policy loss is computed as follows:

```python
def policy_loss(alpha: Number, logprobs: Tensor, qf_values: Tensor) -> Tensor:
    return ((alpha * logprobs) - qf_values).mean()
```
where the logprobs are computed from the policy network, and the qf values are the smallest of the values computed by the critic networks. The critic loss is computed as follows:

```python
def critic_loss(qf_values: Tensor, next_qf_value: Tensor, num_critics: int) -> Tensor:
    qf_loss = sum(
        F.mse_loss(qf_values[..., qf_value_idx].unsqueeze(-1), next_qf_value) for qf_value_idx in range(num_critics)
    )
    return qf_loss
```
here `qf_values` are the values computed by the critic networks, and `next_qf_value` is the target value of the next state computed with the target critic networks. 

Finally, the entropy loss is computed as follows:

```python
def entropy_loss(log_alpha: Tensor, logprobs: Tensor, target_entropy: Tensor) -> Tensor:
    alpha_loss = (-log_alpha * (logprobs + target_entropy)).mean()
    return alpha_loss
```
this loss is used to update the temperature parameter of the entropy in the policy's objective function.

## Agent
For SAC, we decided to create an agent class, which is implemented in the `agent.py` file. This was made to make it easier for the user to interact with the numerous models that the SAC's algorithm uses: a policy network, the critic networks (`algo.critic.n`, two by default), as many target critic networks, and the temperature parameter of the entropy.
