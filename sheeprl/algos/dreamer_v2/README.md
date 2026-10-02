# DreamerV2
A reinforcement learning repository cannot be without DreamerV2, a model-base algorithm developed by Hafner et al. in [Mastering Atari with discrete world models](https://doi.org/10.48550/arXiv.2010.02193). We implemented DreamerV2 in PyTorch for various reasons: first of all, it is the SOTA algorithm; second, there are no easy-to-understand implementations in PyTorch. We aim to provide a clear implementation which faithfully respects the paper, we started from the first version and then with the intention of moving toward later versions.

The agent uses a world model to learn a latent representation of the environment, this representation is used by the actor and the critic to select the actions and to predict the state values respectively. The world model is the most complex part of DreamerV2 and it is composed by:

*  An **encoder** which encodes the observations provided by the environment: the images with a convolutional network, the vectors with an MLP.
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

The *reconstruction loss* of DreamerV2 is implemented in the `loss.py` file, while the actor and critic losses are computed in the `behaviour_learning` function of the `dreamer_v2.py` file. The *reconstruction loss* is the most complicated and it is composed by four different parts:

1.  **KL loss**: the kl divergence balanced between the posterior and prior states.
2.  **Observation loss**: the logprob of the distribution produced by the observation model on the observations.
3.  **Reward loss**: the logprob of the distribution computed by the reward model on the rewards.
4.  **Continue loss** *(optional)*: the logprob of the distribution produced by the continue model on the continue targets, which are the discount factor `algo.gamma` for the non-terminal steps and $0$ for the terminal ones (`terminated`).

The reconstruction loss is computed as follows:
```python
def reconstruction_loss(
    po: Dict[str, Distribution],
    observations: Dict[str, Tensor],
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
    discount_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    observation_loss = -sum([po[k].log_prob(observations[k]).mean() for k in po.keys()])
    reward_loss = -pr.log_prob(rewards).mean()
    # KL balancing
    lhs = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    rhs = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    if kl_free_avg:
        lhs = lhs.mean()
        rhs = rhs.mean()
        free_nats = torch.full_like(lhs, kl_free_nats)
        loss_lhs = torch.maximum(lhs, free_nats)
        loss_rhs = torch.maximum(rhs, free_nats)
    else:
        free_nats = torch.full_like(lhs, kl_free_nats)
        loss_lhs = torch.maximum(lhs, kl_free_nats).mean()
        loss_rhs = torch.maximum(rhs, kl_free_nats).mean()
    kl_loss = kl_balancing_alpha * loss_lhs + (1 - kl_balancing_alpha) * loss_rhs
    if pc is not None and continue_targets is not None:
        continue_loss = discount_scale_factor * -pc.log_prob(continue_targets).mean()
    else:
        continue_loss = torch.zeros_like(reward_loss)
    reconstruction_loss = kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss
    return reconstruction_loss, kl, kl_loss, reward_loss, observation_loss, continue_loss
```
Here it is necessary to define some hyper-parameters, such as *(i)* the `kl_free_nats`, which is the minimum value of the *KL loss* (`algo.world_model.kl_free_nats`, default to 1); or *(ii)* the `kl_regularizer` parameter to scale the *KL loss* (`algo.world_model.kl_regularizer`); *(iii)* wheter to compute or not the *continue loss* (`algo.world_model.use_continues`, default to `False`); *(iv)* `discount_scale_factor`, the parameter to scale the *continue loss* (`algo.world_model.discount_scale_factor`).

Another difference between Dreamer-V1 and Dreamer-V2 is in the *actor loss*, which mixes standard on-policy Reinforce with Dynamic backprop, while adding the policy entropy to favor the exploration of the environment:

```math
\mathcal{L}(\psi) = \text{E}{p_{\phi},p_{\psi}}\left[\sum^{H-1}_{t=1}(-\rho\ln p_{\psi}(\hat{a}_t|\hat{z}_t) \cdot \text{sg}(V^{\lambda}_t - v_{\xi}))-(1-\rho)V^{\lambda}_t-\eta\text{H}[a_t|\hat{z}_t]\right]
```

here the `lambda_values` $`V^{\lambda}_t`$ are the discounted lambda targets computed in the latent dynamics by a **target critic**, a copy of the critic updated every `algo.critic.per_rank_target_network_update_freq` gradient steps (100 by default). The mix $`\rho`$ is `algo.actor.objective_mix` and the entropy coefficient $`\eta`$ is `algo.actor.ent_coef`.

Finally, the critic loss is computed as follows:
```python
qv = Independent(Normal(critic(imagined_trajectories.detach()[:-1]), 1), 1)
value_loss = -torch.mean(discount[:-1, ..., 0] * qv.log_prob(lambda_values.detach()))
```
where `discount` is the cumulative product of the discounts predicted along the imagined trajectories (the continue model if `algo.world_model.use_continues=True`, otherwise `algo.gamma`), so that the imagined steps that are likely to end the episode count less, whereas `qv` is the distribution of the values computed by the critic. The actor loss is weighted by the same `discount`.

## Implementation
The algorithm is the `DreamerV2` class in the `dreamer_v2.py` file, run by the training loop shared by all the algorithms (`sheeprl.core.loop.run`). Every iteration plays one step in every environment (the `SequencePlayer` class) and stores it in the replay buffer, then does `algo.replay_ratio` gradient steps per policy step, each one on its own batch of sequences of `algo.per_rank_sequence_length` steps: the world model is updated by the `world_model_learning` function (dynamic learning), then the actor and the critic by the `behaviour_learning` function (behaviour learning). Until `algo.learning_starts` the actions are uniformly random (except with MineDojo, whose actions must respect the action masks). The models are created by the `build_models` function of the `agent.py` file.

The replay buffer also stores `is_first`, which marks the first step of every episode: there the RSSM restarts from the zero state, both when the agent plays and when the world model learns. The replay buffer is selected by `buffer.type`: `sequential` (one buffer of steps per environment) or `episode` (a buffer of whole episodes, whose ends can be sampled more often with `buffer.prioritize_ends=True`).

Continuous actions are in $[-1, 1]$ for the agent: with `algo.normalize_actions=True` (the default), the `NormalizeAction` wrapper (`sheeprl/envs/wrappers.py`) rescales them to the bounds of the action space of the environment.

## Agent
The models of the DreamerV2 agent are defined in the `agent.py` file in order to have a clearer definition of the components of the agent. Our implementation of DreamerV2 resizes the images to 64x64 pixels (`env.screen_size` is set to `64` and `env.frame_stack` to `1`), and the recurrent model of the RSSM is composed by a linear layer followed by a ELU activation function (with layer normalization when `algo.world_model.recurrent_model.layer_norm=True`, the default) and a GRU cell with layer normalization. Finally, the agent can work with continuous or discrete control.

## Packages
In order to use a broader set of environments of provided by [Gymnasium](https://gymnasium.farama.org/) it is necessary to install optional packages:

*  Mujoco environments: `pip install -e .[mujoco]`
*  Atari environments: `pip install -e .[atari]` (the ROMs are included)
*  DMC environments: `pip install -e .[dmc]`

For more information, check the [Atari](../../../howto/learn_in_atari.md) and the [DMC and MuJoCo](../../../howto/learn_in_dmc.md) how-tos.

## Hyper-parameters
For DreamerV2, the number of environments of every process is set by `env.num_envs`: the replay buffer keeps the steps of every environment separate. In addition, we would like to recommend the value of the `per_rank_batch_size` hyper-parameter to the users: the recommended batch size for the DreamerV2 agent is 50 for single-process training (the standard hyper-parameters below use it, while `exp=dreamer_v2` sets `16`), if you want to use distributed training, we recommend to divide the batch size by the number of processes and to set the `per_rank_batch_size` hyper-parameter accordingly.

## Atari environments
There are two versions for most Atari environments: one version uses the *frame skip* property by default, whereas the second does not implement it. If the first version is selected, then the value of the `env.action_repeat` hyper-parameter must be `1`; instead, to select an environment without *frame skip*, it is necessary to insert `NoFrameskip` in the environment id and remove the prefix `ALE/` from it. For instance, the environment `ALE/AirRaid-v5` must be instantiated with `env.action_repeat=1`, whereas its version without *frame skip* is `AirRaidNoFrameskip-v4` and can be istanziated with any value of `env.action_repeat` greater than zero.
For more information see the official documentation of [Gymnasium Atari environments](https://gymnasium.farama.org/environments/atari/).

The standard hyperparameters to learn in the Atari environments are the following (`algo.replay_ratio` is the number of gradient steps per policy step: `0.25` is one gradient step every $4$ policy steps):

```bash
python sheeprl.py \
exp=dreamer_v2 \
fabric.devices=1 \
env=atari \
env.id=AssaultNoFrameskip-v0 \
env.capture_video=True \
env.action_repeat=4 \
env.clip_rewards=True \
env.max_episode_steps=27000 \
env.num_envs=1 \
algo.total_steps=200000000 \
algo.learning_starts=200000 \
algo.per_rank_pretrain_steps=1 \
algo.replay_ratio=0.25 \
algo.gamma=0.995 \
algo.world_model.kl_regularizer=0.1 \
algo.world_model.discount_scale_factor=5.0 \
algo.actor.ent_coef=1e-3 \
algo.actor.optimizer.lr=4e-5 \
algo.critic.optimizer.lr=1e-4 \
algo.world_model.optimizer.lr=2e-4 \
algo.world_model.use_continues=True \
algo.world_model.representation_model.hidden_size=600 \
algo.world_model.transition_model.hidden_size=600 \
algo.world_model.recurrent_model.recurrent_state_size=600 \
algo.world_model.kl_free_nats=0.0 \
algo.per_rank_batch_size=50 \
algo.cnn_keys.encoder=[rgb] \
buffer.size=2000000 \
buffer.memmap=True \
buffer.type=episode \
buffer.prioritize_ends=True \
checkpoint.every=100000
```

## DMC environments
It is possible to use the environments provided by the [DeepMind Control suite](https://www.deepmind.com/open-source/deepmind-control-suite). To use such environments it is necessary to specify "dmc", the domain and the task of the environment in the `env`, `env.wrapper.domain_name` and `env.wrapper.task_name` hyper-parameters respectively, e.g., `env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk` will create an instance of the walker walk environment. For more information about all the environments, check their [paper](https://arxiv.org/abs/1801.00690).

When running DreamerV2 in a DMC environment on a server (or a PC without a video terminal) it could be necessary to add two variables to the command to launch the script: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa <command>`. For instance, to run walker walk with DreamerV2 on two gpus (0 and 1) it is necessary to run the following command: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="0,1" python sheeprl.py exp=dreamer_v2 fabric.devices=2 fabric.accelerator=gpu env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk env.action_repeat=2 env.capture_video=True checkpoint.every=80000 algo.cnn_keys.encoder=[rgb]`. 
Other possibitities for the variable `MUJOCO_GL` are: `GLFW` for rendering to an X11 window or and `EGL` for hardware accelerated headless. (For more information, click [here](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)).
Moreover, it could be necessary to decomment two rows in the `sheeprl.algos.dreamer_v2.dreamer_v2.py` file.

The standard hyperparameters used for the DMC environment are the following:

```bash
PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa python sheeprl.py \
exp=dreamer_v2 \
fabric.devices=1 \
env=dmc \
env.wrapper.domain_name=walker \
env.wrapper.task_name=walk \
env.capture_video=True \
env.action_repeat=2 \
env.clip_rewards=False \
env.max_episode_steps=1000 \
env.num_envs=1 \
algo.cnn_keys.encoder=[rgb] \
algo.total_steps=5000000 \
algo.learning_starts=1000 \
algo.per_rank_pretrain_steps=100 \
algo.replay_ratio=0.2 \
algo.gamma=0.99 \
algo.world_model.kl_regularizer=1.0 \
algo.actor.ent_coef=1e-4 \
algo.actor.optimizer.lr=8e-5 \
algo.critic.optimizer.lr=8e-5 \
algo.world_model.optimizer.lr=3e-4 \
algo.world_model.use_continues=False \
algo.world_model.representation_model.hidden_size=200 \
algo.world_model.transition_model.hidden_size=200 \
algo.world_model.recurrent_model.recurrent_state_size=200 \
algo.per_rank_batch_size=50 \
algo.world_model.kl_free_nats=1.0 \
algo.actor.objective_mix=0.0 \
buffer.size=5000000 \
buffer.memmap=True \
buffer.type=episode \
buffer.prioritize_ends=False \
checkpoint.every=100000
```

## Recommendations
Since DreamerV2 requires a huge number of steps and consequently a large buffer size, we recommend keeping the buffer on cpu and not moving it to cuda. Furthermore, in order to limit memory usage, we recommend to store the observations in `uint8` format and to normalize the observations just before starting the training one batch at a time. In addition, it is recommended to set the `buffer.memmap` argment to `True` to map the buffer to disk and avoid having it all in RAM. Finally, it is important to remind the user that DreamerV2 works with image observations (`algo.cnn_keys.encoder`), vector observations (`algo.mlp_keys.encoder`) or both, and that the keys to reconstruct (`algo.cnn_keys.decoder` and `algo.mlp_keys.decoder`) must be among the encoded ones.
