# DreamerV2
A reinforcement learning repository cannot be without DreamerV2, a model-base algorithm developed by Hafner et al. in [Mastering Atari with discrete world models](https://doi.org/10.48550/arXiv.1912.01603). We implemented DreamerV2 in PyTorch for various reasons: first of all, it is the SOTA algorithm; second, there are no easy-to-understand implementations in PyTorch. We aim to provide a clear implementation which faithfully respects the paper, we started from the first version and then with the intention of moving toward later versions.

The agent uses a world model to learn a latent representation of the environment, this representation is used by the actor and the critic to select the actions and to predict the state values respectively. The world model is the most complex part of DreamerV2 and it is composed by:

*  An **encoder** which encodes the pixel observations provided by the environment.
*  An **RSSM** ([Learning Latent Dynamics for Planning from Pixels](https://doi.org/10.48550/arXiv.1811.04551)) which is responsible to to encode the history and dynamics of the environments.
*  A **representation model** which models the posterior probabilities over stochastic states. The representation model computes the stochastic states given the previous history and the current observation received by the environment.
*  A **transition model** which models the prior probabilities over stochastic states given just the history encoded by the RSSM. The transition model is trained to match the representation model by minimizing the KL divergence w.r.t. the representation model.
*  An **observation model** that tries to reconstruct the observations from the latent state.
*  A **reward model** which predicts the reward for a given state.
*  An optional **continue model** that estimates the discount factor for the computation of cumulative reward.

Two big differences between Dreamer-V1 and Dreamer-V2 in the world model are:

1. The stochastic states are no more represented as Gaussian distributions, but as a 32x32 One-Hot Categoricals. The reasons why One-Hot Categoricals are preferred w.r.t. Gaussians are well explained in the paper:
   1. A categorical prior can perfectly fit the aggregate posterior, because a mixture of categoricals is again a categorical. In contrast, a Gaussian prior cannot match a mixture of Gaussian posteriors, which could make it difficult to predict multi-modal changes between one image and the next
   2. The level of sparsity enforced by a vector of categorical latent variables could be beneficial for generalization
   3. Despite common intuition, categorical variables may be easier to optimize than Gaussian variables, possibly because the straight-through gradient estimator ignores a term that would otherwise scale the gradient
   4. Categorical variables could be a better inductive bias than unimodal continuous latent variables for modeling the non-smooth aspects of Atari games, such as when entering a new room, or when collected items or defeated enemies disappear from the image
2. Instead of directly minimizing the KL divergence between the representation distribution (Q) and the transition one (P),  a balance between the two is used: $\alpha \text{KL}(\text{sg}(\text{approx-posterior}), \text{prior}) + (1 - \alpha) \text{KL}(\text{approx-posterior}, \text{sg}(\text{prior}))$, with $\alpha=0.8$. Called **KL-balancing**, it works because by scaling up the prior cross entropy relative to the posterior entropy, the world model is encouraged to minimize the KL by improving its prior dynamics toward the more informed posteriors, as opposed to reducing the KL by increasing the posterior entropy. 

The actor and the critic are two MLP models which take in input the latent states and produce in output the actions and the predicted values respectively. The great advantage of DreamerV2 consists of learning long-horizon behaviours by leveraging the latent dynamics, so the actor and the critic are learned in the latent space. The learning process consists of two parts:

*  **Dynamic Learning**: the agent learns the latent representations of the states.
*  **Behaviour Learning**: the agent leverages the world model to imagine tajectories (without the use of observations) and learn the actor and the critic entirely in the latent dynamics.

The three losses of DreamerV2 are implemented in the `loss.py` file. The *reconstruction loss* is the most complicated and it is composed by four different parts:

1.  **KL loss**: the kl divergence balanced between the posterior and prior states.
2.  **Observation loss**: the logprob of the distribution produced by the observation model on the observations.
3.  **Reward loss**: the logprob of the distribution computed by the reward model on the rewards.
4.  **Continue loss** *(optional)*: the logprob of the distribution produced by the continue model on the dones.

The reconstruction loss is computed as follows:
```python
def reconstruction_loss(
    po: Distribution,
    observations: Tensor,
    pr: Distribution,
    rewards: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    kl_balancing_alpha: float = 0.8,
    kl_free_nats: float = 0.0,
    kl_free_avg: bool = True,
    kl_regularizer: float = 1.0,
    pc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = observations.device
    observation_loss = -po.log_prob(observations).mean()
    reward_loss = -pr.log_prob(rewards).mean()
    # KL balancing
    lhs = kl_divergence(
        OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()),
        OneHotCategoricalStraightThrough(logits=priors_logits),
    )
    rhs = kl_divergence(
        OneHotCategoricalStraightThrough(logits=posteriors_logits),
        OneHotCategoricalStraightThrough(logits=priors_logits.detach()),
    )
    kl_free_nats = torch.tensor([kl_free_nats], device=lhs.device)
    if kl_free_avg:
        loss_lhs = torch.maximum(lhs.mean(), kl_free_nats)
        loss_rhs = torch.maximum(rhs.mean(), kl_free_nats)
    else:
        loss_lhs = torch.maximum(lhs, kl_free_nats).mean()
        loss_rhs = torch.maximum(rhs, kl_free_nats).mean()
    kl_loss = kl_balancing_alpha * loss_lhs + (1 - kl_balancing_alpha) * loss_rhs
    continue_loss = torch.tensor(0, device=device)
    if pc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continue_targets).mean()
    reconstruction_loss = kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss
    return reconstruction_loss, kl_loss, reward_loss, observation_loss, continue_loss
```
Here it is necessary to define some hyper-parameters, such as *(i)* the `kl_free_nats`, which is the minimum value of the *KL loss* (default to 0); or *(ii)* the `kl_regularizer` parameter to scale the *KL loss*; *(iii)* wheter to compute or not the *continue loss*; *(iv)* `continue_scale_factor`, the parameter to scale the *continue loss*.

Another difference between Dreamer-V1 and Dreamer-V2 is in the *actor loss*, which mixes standard on-policy Reinforce with Dynamic backprop, while adding the policy entropy to favor the exploration of the environment.

$$
    \mathcal{L}(\psi) = \text{E}_{p_{\phi},p_{\psi}}\left[\sum^{H-1}_{t=1}(-\rho\ln p_{\psi}(\hat{a}_t|\hat{z}_t)\text{sg}(V^{\lambda}_t - v_{\xi}))-(1-\rho)V^{\lambda}_t-\eta\text{H}[a_t|\hat{z}_t]\right]
$$
here the `lambda_values` are the discounted lambda targets computed in the latent dynamics by a **target critic** which is updated every 100 gradient steps.

Finally, the critic loss is computed as follows:
```python
def critic_loss(qv: Distribution, lambda_values: Tensor, discount: Tensor) -> Tensor:
    return -torch.mean(discount * qv.log_prob(lambda_values))
```
where `discount` is the discount to apply to the returns to compute the cumulative return, whereas `qv` is the distribution of the values computed by the critic.

## Agent
The models of the DreamerV2 agent are defined in the `agent.py` file in order to have a clearer definition of the components of the agent. Our implementation of DreamerV2 assumes that the observations are images of shape `(3, 64, 64)` and that the recurrent model of the RSSM is composed by a linear layer followed by a ELU activation function and a GRU layer. Finally, the agent can work with continuous or discrete contorl.

## Packages
In order to use a broader set of environments of provided by [Gymnasium](https://gymnasium.farama.org/) it is necessary to install optional packages:

*  Mujoco environments: `pip install gymnasium[mujoco]`
*  Atari environments: `pip install gymnasium[atari]` and `pip install gymnasium[accept-rom-license]`

## Hyper-parameters
For DreamerV2, we decided to fix the number of environments to `1`, in order to have a clearer and more understandable management of the environment interaction. In addition, we would like to recommend the value of the `per_rank_batch_size` hyper-parameter to the users: the recommended batch size for the DreamerV2 agent is 50 for single-process training, if you want to use distributed training, we recommend to divide the batch size by the number of processes and to set the `per_rank_batch_size` hyper-parameter accordingly.

## Atari environments
There are two versions for most Atari environments: one version uses the *frame skip* property by default, whereas the second does not implement it. If the first version is selected, then the value of the `action_repeat` hyper-parameter must be `1`; instead, to select an environment without *frame skip*, it is necessary to insert `NoFrameskip` in the environment id and remove the prefix `ALE/` from it. For instance, the environment `ALE/AirRaid-v5` must be instantiated with `action_repeat=1`, whereas its version without *frame skip* is `AirRaidNoFrameskip-v4` and can be istanziated with any value of `action_repeat` greater than zero.
For more information see the official documentation of [Gymnasium Atari environments](https://gymnasium.farama.org/environments/atari/).

The standard hyperparameters to learn in the Atari environments are:

```bash
lightning run model --devices=1 sheeprl.py dreamer_v3 \
--env_id=AssaultNoFrameskip-v0 \
--capture_video \
--action_repeat=4 \
--clip_rewards=True \
--total_steps=200000000 \
--learning_starts=200000 \
--pretrain_steps=1 \
--train_every=4 \
--gamma=0.995 \
--kl_regularizer=0.1 \
--continue_scale_factor=5.0 \
--actor_ent_coef=1e-3 \
--actor_lr=4e-5 \
--critic_lr=1e-4 \
--world_lr=2e-4 \
--use_continues \
--hidden_size=600 \
--recurrent_state_size=600 \
--buffer_size=2000000 \
--memmap_buffer \
--kl_free_nats=0.0 \
--max_episode_steps=27000 \
--per_rank_batch_size=50 \
--checkpoint_every=100000 \
--buffer_type=episode \
--prioritize_ends=True 
```

## DMC environments
It is possible to use the environments provided by the [DeepMind Control suite](https://www.deepmind.com/open-source/deepmind-control-suite). To use such environments it is necessary to specify the "dmc" domain in the `env_id` hyper-parameter, e.g., `env_id = dmc_walker_walk` will create an instance of the walker walk environment. For more information about all the environments, check their [paper](https://arxiv.org/abs/1801.00690).

When running DreamerV2 in a DMC environment on a server (or a PC without a video terminal) it could be necessary to add two variables to the command to launch the script: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa <command>`. For instance, to run walker walk with DreamerV2 on two gpus (0 and 1) it is necessary to runthe following command: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="2,3" lightning run model --devices=2 --accelerator=gpu sheeprl.py dreamer_v1 --env_id=dmc_walker_walk --action_repeat=2 --capture_video --checkpoint_every=80000 --seed=1`. 
Other possibitities for the variable `MUJOCO_GL` are: `GLFW` for rendering to an X11 window or and `EGL` for hardware accelerated headless. (For more information, click [here](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)).
Moreover, it could be necessary to decomment two rows in the `sheeprl.algos.dreamer_v1.dreamer_v1.py` file.

The standard hyperparameters used for the DMC environment are the following:

```bash
PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa lightning run model --devices=1 sheeprl.py dreamer_v3 \
--env_id=dmc_walker_walk \
--capture_video \
--action_repeat=2 \
--clip_rewards=False \
--total_steps=200000000 \
--learning_starts=2000 \
--pretrain_steps=200 \
--train_every=5 \
--gamma=0.99 \
--kl_regularizer=1.0 \
--actor_ent_coef=1e-4 \
--actor_lr=8e-5 \
--critic_lr=8e-5 \
--world_lr=3e-4 \
--use_continues=False \
--hidden_size=200 \
--recurrent_state_size=200 \
--buffer_size=2000000 \
--memmap_buffer \
--kl_free_nats=1.0 \
--max_episode_steps=27000 \
--per_rank_batch_size=50 \
--checkpoint_every=100000 \
--buffer_type=episode \
--prioritize_ends=True 
```

## Recommendations
Since DreamerV2 requires a huge number of steps and consequently a large buffer size, we recommend keeping the buffer on cpu and not moving it to cuda. Furthermore, in order to limit memory usage, we recommend to store the observations in `uint8` format and to normalize the observations just before starting the training one batch at a time. In addition, it is recommended to set the `memmap_buffer` argment to `True` to map the buffer to disk and avoid having it all in RAM. Finally, it is important to remind the user that DreamerV2 works only with observations in pixel form, therefore, only environments with observation space that is an instance of `gym.spaces.Box` can be selected when used with gymnasium.