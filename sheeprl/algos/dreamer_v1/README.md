# DreamerV1
A reinforcement learning repository cannot be without DreamerV1, a model-base algorithm developed by Hafner et al. in [Dream to Control: Learning Behaviors by Latent Imagination](https://doi.org/10.48550/arXiv.1912.01603). We implemented DreamerV1 in PyTorch for various reasons: first of all, it is the SOTA algorithm; second, there are no easy-to-understand implementations in PyTorch. We aim to provide a clear implementation which faithfully respects the paper, we started from the first version and then with the intention of moving toward later versions.

The agent uses a world model to learn a latent representation of the environment, this representation is used by the actor and the critic to select the actions and to predict the state values respectively. The world model is the most complex part of DreamerV1, it is composed by:

*  An **encoder** which encodes the observations in pixel form provided by the environment.
*  An **RSSM** ([Learning Latent Dynamics for Planning from Pixels](https://doi.org/10.48550/arXiv.1811.04551)) which is responsible to generate the latent states.
*  An **observation model** that tries to reconstruct the observations from the latent state.
*  A **reward model** which predicts the reward for a given state.
*  An optional **continue model** that estimates the discount factor for the computation of cumulative reward.

The actor and the critic are two MLP models which take in input the latent states and produce in output the actions and the predicted values respectively. The great advantage of DreamerV1 consists of learning long-horizon behaviours by leveraging the latent dynamics, so the actor and the critic are learned in the latent space. The learning process consists of two parts:

*  **Dynamic Learning**: the agent learns the latent representations of the states.
*  **Behaviour Learning**: the agent leverages the world model to imagine tajectories (without the use of observations) and learn the actor and the critic entirely in the latent dynamics.

The three losses of DreamerV1 are implemented in the `loss.py` file. The *reconstruction loss* is the most complicated and it is composed by four different parts:

1.  **State loss**: the kl divergence between the real and predicted latent states computed by the RSSM.
2.  **Observation loss**: the logprob of the distribution produced by the observation model on the observations.
3.  **Reward loss**: the logprob of the distribution computed by the reward model on the rewards.
4.  **Continue loss** *(optional)*: the logprob of the distribution produced by the continue model on the dones.

The reconstruction loss is computed as follows:
```python
def reconstruction_loss(
    qo: Distribution,
    observations: Tensor,
    qr: Distribution,
    rewards: Tensor,
    p: Distribution,
    q: Distribution,
    kl_free_nats: float = 3.0,
    kl_regularizer: float = 1.0,
    qc: Optional[Distribution] = None,
    dones: Optional[Tensor] = None,
    continue_scale_factor: float = 10.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

    device = observations.device
    continue_loss = torch.tensor(0, device=device)
    observation_loss = -qo.log_prob(observations).mean()
    reward_loss = -qr.log_prob(rewards).mean()
    state_loss = torch.max(torch.tensor(kl_free_nats, device=device), kl_divergence(p, q).mean())
    if qc is not None and dones is not None:
        continue_loss = continue_scale_factor * F.binary_cross_entropy(qc.probs, dones)
    reconstruction_loss = kl_regularizer * state_loss + observation_loss + reward_loss + continue_loss
    return reconstruction_loss, state_loss, reward_loss, observation_loss, continue_loss
```
Here it is necessary to define some hyper-parameters, such as *(i)* the `kl_free_nats`, which is the minimum value of the *state loss* (default to 3); or *(ii)* the `kl_regularizer` parameter to scale the *state loss*; *(iii)* wheter to compute or not the *continue loss*; *(iv)* `continue_scale_factor`, the parameter to scale the *continue loss*.

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
where `discount` is the discount to apply to the returns to compute the cumulative return, whereas `qv` is the distribution of the values computed by the critic.

## Agent
The models of the DreamerV1 agent are defined in the `agent.py` file in order to have a clearer definition of the components of the agent. Our implementation of DreamerV1 assumes that the observations are images of shape `(3, 64, 64)` and that the recurrent model of the RSSM is composed by a linear layer followed by a ELU activation function and a GRU layer. Finally, the agent can work with continuous or discrete contorl.

## Packages
In order to use a broader set of environments of provided by [Gymnasium](https://gymnasium.farama.org/) it is necessary to install optional packages:

*  Mujoco environments: `pip install gymnasium[mujoco]`
*  Atari environments: `pip install gymnasium[atari]` and `pip install gymnasium[accept-rom-license]`

## Hyper-parameters
For DreamerV1, we decided to fix the number of environments to `1`, in order to have a clearer and more understandable management of the environment interaction. In addition, we would like to recommend the value of the `per_rank_batch_size` hyper-parameter to the users: the recommended batch size for the DreamerV1 agent is 50 for single-process training, if you want to use distributed training, we recommend to divide the batch size by the number of processes and to set the `per_rank_batch_size` hyper-parameter accordingly.

## Atari environments
There are two versions for most Atari environments: one version uses the *frame skip* property by default, whereas the second does not implement it. If the first version is selected, then the value of the `action_repeat` hyper-parameter must be `1`; instead, to select an environment without *frame skip*, it is necessary to insert `NoFrameskip` in the environment id and remove the prefix `ALE/` from it. For instance, the environment `ALE/AirRaid-v5` must be instantiated with `action_repeat=1`, whereas its version without *frame skip* is `AirRaidNoFrameskip-v4` and can be istanziated with any value of `action_repeat` greater than zero.
For more information see the official documentation of [Gymnasium Atari environments](https://gymnasium.farama.org/environments/atari/).

## DMC environments
It is possible to use the environments provided by the [DeepMind Control suite](https://www.deepmind.com/open-source/deepmind-control-suite). To use such environments it is necessary to specify "dmc", the domain and the task of the environment in the `env`, `env.wrapper.domain_name` and `env.wrapper.task_name` hyper-parameters respectively, e.g., `env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk` will create an instance of the walker walk environment. For more information about all the environments, check their [paper](https://arxiv.org/abs/1801.00690).

When running DreamerV1 in a DMC environment on a server (or a PC without a video terminal) it could be necessary to add two variables to the command to launch the script: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa <command>`. For instance, to run walker walk with DreamerV1 on two gpus (0 and 1) it is necessary to runthe following command: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa python sheeprl.py exp=dreamer_v1 fabric.devices=2 fabric.accelerator=gpu env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk env.action_repeat=2 env.capture_video=True checkpoint.every=100000 algo.cnn_keys.encoder=[rgb]`. 
Other possibitities for the variable `MUJOCO_GL` are: `GLFW` for rendering to an X11 window or and `EGL` for hardware accelerated headless. (For more information, click [here](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)).
Moreover, it could be necessary to decomment two rows in the `sheeprl.algos.dreamer_v1.dreamer_v1.py` file.

## Recommendations
Since DreamerV1 requires a huge number of steps and consequently a large buffer size, we recommend keeping the buffer on cpu and not moving it to cuda. Furthermore, in order to limit memory usage, we recommend to store the observations in `uint8` format and to normalize the observations just before starting the training one batch at a time. In addition, it is recommended to set the `buffer.memmap` argment to `True` to map the buffer to disk and avoid having it all in RAM. Finally, it is important to remind the user that DreamerV1 works only with observations in pixel form, therefore, only environments with observation space that is an instance of `gym.spaces.Box` can be selected when used with gymnasium.