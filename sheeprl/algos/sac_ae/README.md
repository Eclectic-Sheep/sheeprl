# SAC-AutoEncoder (SAC-AE)
Images are everywhere, thus having effective RL approaches that can utilize pixels as input would potentially enable solutions for a wide range of real world applications, for example robotics and videogames. In [SAC-AutoEncoder](https://arxiv.org/abs/1910.01741) the standard [SAC](https://arxiv.org/abs/1801.01290) agent is enriched with a convolutional encoder, which encodes images into features, shared between the actor and critic. Also, to improve the quality of the extracted features, a convolutional decoder is used to reconstruct the input images from the features, effectively creating the Encoder-Decoder architecture.

The architecture is depicted in the following figure:

![](https://eclecticsheep.ai/assets/images/sac_ae.png)

Since learning directly from images can be cumbersome, as the authors have found out, some tricks must be taken into account: i.e.:

1. Deterministic autoencoder: the encoder-decoder architecture is a standard deterministic one, which means that we are not going to learn a distribution over the extracted features conditioned on the input images
2. The encoder will receive the gradients from the critic but not from the actor: receiving the gradients from the actor changes also the Q-function during the actor update, since the encoder is shared between the actor and the critic. 
3. To overcame the slowdown in the encoder update due to 2., the convolutional weights of the target Q-function are updated faster than the rest of the network’s parameters (effectively using a $\tau_{\text{enc}} > \tau_{\text{Q}}$)

The models are created by the `build_agent` function of the `agent.py` file. The critics encode the observations with the encoder, while the actor encodes them with a copy of it: the agent ties the convolutional layers of the CNN encoder and the layers of the MLP encoder of the actor to the ones of the critics.

```python
encoder = MultiEncoder(cnn_encoder, mlp_encoder)
...
decoder = MultiDecoder(cnn_decoder, mlp_decoder)

# Setup actor and critic. Those will initialize with orthogonal weights
# both the actor and critic
actor = SACAEContinuousActor(
    encoder=copy.deepcopy(encoder),
    action_dim=act_dim,
    distribution_cfg=cfg.distribution,
    hidden_size=cfg.algo.actor.hidden_size,
    action_low=action_space.low,
    action_high=action_space.high,
)
qfs = [
    SACAEQFunction(
        input_dim=encoder.output_dim, action_dim=act_dim, hidden_size=cfg.algo.critic.hidden_size, output_dim=1
    )
    for _ in range(cfg.algo.critic.n)
]
critic = SACAECritic(encoder=encoder, qfs=qfs)

# The agent will tied convolutional and linear weights between the encoder actor and critic
agent = SACAEAgent(
    actor,
    critic,
    target_entropy,
    alpha=cfg.algo.alpha.alpha,
    tau=cfg.algo.tau,
    encoder_tau=cfg.algo.encoder.tau,
    device=device,
)
return agent, encoder, decoder
```

The algorithm is the `SACAE` class in the `sac_ae.py` file, run by the training loop shared by all the algorithms (`sheeprl.core.loop.run`). Its `build` method moves the models to the device with `sheeprl.core.setup_module`: the models are not wrapped in `DistributedDataParallel`, and with several processes `sheeprl.core.update` averages the gradients across them before every optimizer step.

```python
# The modules set up on the device of the process (`setup_module`), and the policy to play with, which shares the
# weights of the actor
agent, encoder, decoder, policy = build_agent(fabric, cfg, obs_space, action_space)
```

Every iteration plays one step in every environment (with random actions until `algo.learning_starts`) and stores it in a replay buffer, then does `algo.replay_ratio` gradient steps per policy step, each one on its own batch sampled from the buffer.
The three losses of SAC-AE are the same ones used for SAC, implemented in the `sheeprl/algos/sac/loss.py` file.
To account for the points 2. and 3. above, a gradient step (the `train_step` method) is the following:

```python
cfg = self.cfg.algo
agent = state.agent
cnn_keys = cfg.cnn_keys.encoder
obs = {k: batch[k] / 255.0 if k in cnn_keys else batch[k] for k in cnn_keys + cfg.mlp_keys.encoder}
next_obs = {
    k: batch[f"next_{k}"] / 255.0 if k in cnn_keys else batch[f"next_{k}"]
    for k in cnn_keys + cfg.mlp_keys.encoder
}

# Critics (and the encoder): regress the soft Q-values towards the one-step target of the target critics
with autocast(self.fabric):
    target_qf_values = agent.get_next_target_q_values(
        next_obs, batch["rewards"], batch["terminated"], cfg.gamma
    )
    qf_values = agent.get_q_values(obs, batch["actions"])
    qf_loss = critic_loss(qf_values, target_qf_values, agent.num_critics)
update(self.fabric, qf_loss, state.qf_optimizer)
metrics = {"Loss/value_loss": qf_loss.detach()}
if step % cfg.critic.per_rank_target_network_update_freq == 0:
    agent.critic_target_ema()
    agent.critic_encoder_target_ema()

# Actor: maximize the smallest Q-value of its actions plus their entropy, on the features of the encoder
# (not trained by this loss)
if step % cfg.actor.per_rank_update_freq == 0:
    with autocast(self.fabric):
        actions, logprobs = agent.get_actions_and_log_probs(obs, detach_encoder_features=True)
        qf_values = agent.get_q_values(obs, actions, detach_encoder_features=True)
        min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
        actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
    update(self.fabric, actor_loss, state.actor_optimizer)

    # Entropy coefficient: towards the target entropy
    alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
    update(self.fabric, alpha_loss, state.alpha_optimizer)
    metrics["Loss/policy_loss"] = actor_loss.detach()
    metrics["Loss/alpha_loss"] = alpha_loss.detach()

# Encoder and decoder: reconstruct the observations (the images with 5 bits per channel), with an L2 penalty
# on the features, once for every decoded key
if step % cfg.decoder.per_rank_update_freq == 0:
    with autocast(self.fabric):
        hidden = state.encoder(obs)
        reconstruction = state.decoder(hidden)
        reconstruction_loss = 0
        for k in cfg.cnn_keys.decoder + cfg.mlp_keys.decoder:
            target = preprocess_obs(batch[k], bits=5) if k in cfg.cnn_keys.decoder else batch[k]
            reconstruction_loss += (
                F.mse_loss(target, reconstruction[k])
                + cfg.decoder.l2_lambda * (0.5 * hidden.pow(2).sum(1)).mean()
            )
    # One backward pass for both, then the step of the encoder and the one of the decoder
    state.decoder_optimizer.zero_grad(set_to_none=True)
    params = [*state.encoder.parameters(), *state.decoder.parameters()]
    update(self.fabric, reconstruction_loss, state.encoder_optimizer, params=params)
    state.decoder_optimizer.step()
    metrics["Loss/reconstruction_loss"] = reconstruction_loss.detach()
return metrics
```

`step` counts the gradient steps of the process: by default the target networks and the actor are updated every 2 gradient steps (`algo.critic.per_rank_target_network_update_freq` and `algo.actor.per_rank_update_freq`), the encoder and the decoder at every gradient step (`algo.decoder.per_rank_update_freq`). The encoder of the critics is the `state.encoder` module, so the reconstruction loss also updates the features used by the critics and, through the tied layers, by the actor.

## Agent
The models of the SAC-AE agent are defined in the `agent.py` file in order to have a clearer definition of the components of the agent. Our implementation of SAC-AE resizes the images to 64x64 pixels (`env.screen_size` is set to `64`), with the stacked frames (`env.frame_stack`, `3` in `exp=sac_ae`) concatenated on the channels, and encodes the vector observations (`algo.mlp_keys.encoder`) with an MLP, while both the encoder and decoder of the images are fixed as specified in the paper. The actions must be continuous (`gym.spaces.Box`).

## Packages
In order to use a broader set of environments of provided by [Gymnasium](https://gymnasium.farama.org/) it is necessary to install optional packages:

*  Mujoco environments: `pip install -e .[mujoco]`
*  DMC environments: `pip install -e .[dmc]`

For more information, check the [DMC and MuJoCo how-to](../../../howto/learn_in_dmc.md).

## Hyper-parameters
For SAC-AE, the number of environments of every process is set by `env.num_envs`. In addition, we would like to recommend the value of the `per_rank_batch_size` hyper-parameter to the users: the recommended batch size for the SAC-AE agent is 128 for single-process training (the default of `exp=sac_ae`), if you want to use distributed training, we recommend to divide the batch size by the number of processes and to set the `per_rank_batch_size` hyper-parameter accordingly.

## Atari environments
SAC-AE works only with continuous actions, so it cannot be used with the Atari environments, whose actions are discrete.

## DMC environments
It is possible to use the environments provided by the [DeepMind Control suite](https://www.deepmind.com/open-source/deepmind-control-suite). To use such environments it is necessary to specify "dmc", the domain and the task of the environment in the `env`, the `env.wrapper.domain_name` and `env.wrapper.task_name` hyper-parameters respectively, e.g., `env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk` will create an instance of the walker walk environment. For more information about all the environments, check their [paper](https://arxiv.org/abs/1801.00690).

When running SAC-AE in a DMC environment on a server (or a PC without a video terminal) it could be necessary to add two variables to the command to launch the script: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa <command>`. For instance, to run walker walk with SAC-AE on two gpus (0 and 1) it is necessary to run the following command: `PYOPENGL_PLATFORM="" MUJOCO_GL=osmesa python sheeprl.py exp=sac_ae fabric.devices=2 fabric.accelerator=gpu env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk env.action_repeat=2 env.capture_video=True checkpoint.every=80000 algo.cnn_keys.encoder=[rgb]`. 
Other possibitities for the variable `MUJOCO_GL` are: `GLFW` for rendering to an X11 window or and `EGL` for hardware accelerated headless. (For more information, click [here](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)).

## Recommendations
Since SAC-AE requires a huge number of steps and consequently a large buffer size, we recommend keeping the buffer on cpu and not moving it to cuda, while mapping it to disk by setting the flag `buffer.memmap=True` (the default) when launching the script. Furthermore, in order to limit memory usage, we recommend to store the observations in `uint8` format and to normalize the observations just before starting the training one batch at a time. Finally, it is important to remind the user that SAC-AE works with image observations (`algo.cnn_keys.encoder`), vector observations (`algo.mlp_keys.encoder`) or both, and that the keys to reconstruct (`algo.cnn_keys.decoder` and `algo.mlp_keys.decoder`) must be among the encoded ones.
