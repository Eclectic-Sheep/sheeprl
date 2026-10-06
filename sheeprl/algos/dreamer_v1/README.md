# DreamerV1
A reinforcement learning repository cannot be without DreamerV1, a model-base algorithm developed by Hafner et al. in [Dream to Control: Learning Behaviors by Latent Imagination](https://doi.org/10.48550/arXiv.1912.01603). We implemented DreamerV1 in PyTorch for various reasons: first of all, it is the SOTA algorithm; second, there are no easy-to-understand implementations in PyTorch. We aim to provide a clear implementation which faithfully respects the paper, we started from the first version and then with the intention of moving toward later versions.

The agent uses a world model to learn a latent representation of the environment, this representation is used by the actor and the critic to select the actions and to predict the state values respectively. The world model is the most complex part of DreamerV1, it is composed by:

*  An **encoder** which encodes the observations provided by the environment: the images with a convolutional network, the vectors with an MLP.
*  An **RSSM** ([Learning Latent Dynamics for Planning from Pixels](https://doi.org/10.48550/arXiv.1811.04551)) which is responsible to generate the latent states.
*  An **observation model** that tries to reconstruct the observations from the latent state.
*  A **reward model** which predicts the reward for a given state.
*  An optional **continue model** that estimates the discount factor for the computation of cumulative reward.

The actor and the critic are two MLP models which take in input the latent states and produce in output the actions and the predicted values respectively. The great advantage of DreamerV1 consists of learning long-horizon behaviours by leveraging the latent dynamics, so the actor and the critic are learned in the latent space. The learning process consists of two parts:

*  **Dynamic Learning**: the agent learns the latent representations of the states.
*  **Behaviour Learning**: the agent leverages the world model to imagine tajectories (without the use of observations) and learn the actor and the critic entirely in the latent dynamics.

The three losses of DreamerV1 are implemented in the `loss.py` file. The *reconstruction loss* is the most complicated and it is composed by four different parts:

1.  **State loss**: the kl divergence between the posterior (computed from the observations) and the prior (predicted) stochastic states computed by the RSSM.
2.  **Observation loss**: the logprob of the distribution produced by the observation model on the observations.
3.  **Reward loss**: the logprob of the distribution computed by the reward model on the rewards.
4.  **Continue loss** *(optional)*: the logprob of the distribution produced by the continue model on the continue targets, which are the discount factor `algo.gamma` for the non-terminal steps and $0$ for the terminal ones (`terminated`).

The reconstruction loss is computed as follows:
```python
def reconstruction_loss(
    qo: Distribution,
    observations: Dict[str, Tensor],
    qr: Distribution,
    rewards: Tensor,
    posteriors_dist: Distribution,
    priors_dist: Distribution,
    kl_free_nats: float = 3.0,
    kl_regularizer: float = 1.0,
    qc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 10.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    observation_loss = -sum([qo[k].log_prob(observations[k]).mean() for k in qo.keys()])
    reward_loss = -qr.log_prob(rewards).mean()
    kl = kl_divergence(posteriors_dist, priors_dist).mean()
    free_nats = torch.full_like(kl, kl_free_nats)
    state_loss = torch.max(kl, free_nats)
    if qc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -qc.log_prob(continue_targets).mean()
    else:
        continue_loss = torch.zeros_like(reward_loss)
    reconstruction_loss = kl_regularizer * state_loss + observation_loss + reward_loss + continue_loss
    return reconstruction_loss, kl, state_loss, reward_loss, observation_loss, continue_loss
```
Here it is necessary to define some hyper-parameters, such as *(i)* the `kl_free_nats`, which is the minimum value of the *state loss* (`algo.world_model.kl_free_nats`, default to 3); or *(ii)* the `kl_regularizer` parameter to scale the *state loss* (`algo.world_model.kl_regularizer`); *(iii)* wheter to compute or not the *continue loss* (`algo.world_model.use_continues`, default to `False`); *(iv)* `continue_scale_factor`, the parameter to scale the *continue loss* (`algo.world_model.continue_scale_factor`, default to 1).

The *actor loss* aims to maximize the lambda targets computed in the latent dynamics, and it is computed as follows:
```python
def actor_loss(lambda_values: Tensor) -> Tensor:
    return -torch.mean(lambda_values)
```
here the `lambda_values` are the discounted lambda targets computed in the latent dynamics.

Finally, the critic loss is computed as follows:
```python
def critic_loss(qv: Distribution, lambda_values: Tensor, discount: Tensor) -> Tensor:
    return -torch.mean(discount * qv.log_prob(lambda_values))
```
where `discount` is the cumulative product of the discounts predicted along the imagined trajectories (the continue model if `algo.world_model.use_continues=True`, otherwise `algo.gamma`), so that the imagined steps that are likely to end the episode count less, whereas `qv` is the distribution of the values computed by the critic. The actor loss receives the lambda targets multiplied by the same `discount`.

## Implementation
The algorithm is the `DreamerV1` class in the `dreamer_v1.py` file, run by the training loop shared by all the algorithms (`sheeprl.core.loop.run`). Every iteration plays one step in every environment with the policy (the `DreamerV1Policy` class) and stores it in the replay buffer (the `DreamerV1Writer` class), then does `algo.replay_ratio` gradient steps per policy step, each one on its own batch of sequences of `algo.per_rank_sequence_length` steps: the world model is updated by the `world_model_learning` function (dynamic learning), then the actor and the critic by the `behaviour_learning` function (behaviour learning). Until `algo.learning_starts` the actions are uniformly random (except with MineDojo, whose actions must respect the action masks). The models are created by the `build_agent` function of the `agent.py` file.

The replay buffer also stores `is_first`, which marks the first step of every episode: there the RSSM restarts from the zero state, both when the agent plays and when the world model learns. The first step of every sequence sampled from the buffer is treated as the first one of an episode.

The actions played in the environments get an exploration noise, whose amount is `max(expl_amount * 0.5 ** (step / expl_decay), expl_min)`, where `step` is the number of policy steps (`algo.actor.expl_amount`, `algo.actor.expl_decay` and `algo.actor.expl_min`; the amount does not decay when `algo.actor.expl_decay=0`, the default). Continuous actions are perturbed with a Gaussian noise whose standard deviation is the amount, then clipped to $[-1, 1]$; for every discrete action, every environment plays a uniformly random one with a probability equal to the amount.

Continuous actions are in $[-1, 1]$ for the agent: with `algo.normalize_actions=True` (the default), the `NormalizeAction` wrapper (`sheeprl/envs/wrappers.py`) rescales them to the bounds of the action space of the environment.

## Agent
The models of the DreamerV1 agent are defined in the `agent.py` file in order to have a clearer definition of the components of the agent. Our implementation of DreamerV1 resizes the images to 64x64 pixels (`env.screen_size` is set to `64` and `env.frame_stack` to `1`), and the recurrent model of the RSSM is composed by a linear layer followed by a ELU activation function and a GRU layer. Finally, the agent can work with continuous or discrete control.

## Packages
In order to use a broader set of environments of provided by [Gymnasium](https://gymnasium.farama.org/) it is necessary to install optional packages:

*  Mujoco environments: `pip install -e .[mujoco]`
*  Atari environments: `pip install -e .[atari]` (the ROMs are included)
*  DMC environments: `pip install -e .[dmc]`

For more information, check the [Atari](../../../howto/learn_in_atari.md) and the [DMC and MuJoCo](../../../howto/learn_in_dmc.md) how-tos.

## Hyper-parameters
For DreamerV1, the number of environments of every process is set by `env.num_envs`: the replay buffer keeps the steps of every environment separate. In addition, we would like to recommend the value of the `per_rank_batch_size` hyper-parameter to the users: the recommended batch size for the DreamerV1 agent is 50 for single-process training, if you want to use distributed training, we recommend to divide the batch size by the number of processes and to set the `per_rank_batch_size` hyper-parameter accordingly.

## Atari environments
There are two versions for most Atari environments: one version uses the *frame skip* property by default, whereas the second does not implement it. If the first version is selected, then the value of the `action_repeat` hyper-parameter must be `1`; instead, to select an environment without *frame skip*, it is necessary to insert `NoFrameskip` in the environment id and remove the prefix `ALE/` from it. For instance, the environment `ALE/AirRaid-v5` must be instantiated with `action_repeat=1`, whereas its version without *frame skip* is `AirRaidNoFrameskip-v4` and can be istanziated with any value of `action_repeat` greater than zero.
For more information see the official documentation of [Gymnasium Atari environments](https://gymnasium.farama.org/environments/atari/).

## DMC environments
It is possible to use the environments provided by the [DeepMind Control suite](https://www.deepmind.com/open-source/deepmind-control-suite). To use such environments it is necessary to specify "dmc", the domain and the task of the environment in the `env`, `env.wrapper.domain_name` and `env.wrapper.task_name` hyper-parameters respectively, e.g., `env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk` will create an instance of the walker walk environment. For more information about all the environments, check their [paper](https://arxiv.org/abs/1801.00690).

When running DreamerV1 in a DMC environment on a server (or a PC without a video terminal) it could be necessary to add two variables to the command to launch the script: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa <command>`. For instance, to run walker walk with DreamerV1 on two gpus (0 and 1) it is necessary to run the following command: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa python sheeprl.py exp=dreamer_v1 fabric.devices=2 fabric.accelerator=gpu env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk env.action_repeat=2 env.capture_video=True checkpoint.every=100000 algo.cnn_keys.encoder=[rgb]`. 
Other possibitities for the variable `MUJOCO_GL` are: `GLFW` for rendering to an X11 window or and `EGL` for hardware accelerated headless. (For more information, click [here](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)).
Moreover, it could be necessary to decomment two rows in the `sheeprl.algos.dreamer_v1.dreamer_v1.py` file.

## Recommendations
Since DreamerV1 requires a huge number of steps and consequently a large buffer size, we recommend keeping the buffer on cpu and not moving it to cuda. Furthermore, in order to limit memory usage, we recommend to store the observations in `uint8` format and to normalize the observations just before starting the training one batch at a time. In addition, it is recommended to set the `buffer.memmap` argment to `True` to map the buffer to disk and avoid having it all in RAM. Finally, it is important to remind the user that DreamerV1 works with image observations (`algo.cnn_keys.encoder`), vector observations (`algo.mlp_keys.encoder`) or both, and that the keys to reconstruct (`algo.cnn_keys.decoder` and `algo.mlp_keys.decoder`) must be among the encoded ones.