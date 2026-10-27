# Migrate from sheeprl 0.x to 1.0

sheeprl 1.0 trains every algorithm with one shared core (`sheeprl.core`) and one replay buffer read by samplers (`sheeprl.data`), and reads the observations through declared inputs and transforms (`algo.inputs`, `sheeprl.transforms`). The commands are the same; the changes concern the keys of the observations in the configurations, the checkpoints, the reproducibility of a run, and the code that imports sheeprl.

## Command line and configurations

The commands (`sheeprl`, `sheeprl-eval`, `sheeprl-registration`) are the ones of 0.8.3. The configurations are too, except for the observations and the rewards (next section) and the keys of the players, which play no more:

| 0.8.3 | 1.0 |
|---|---|
| `algo.player.actor_type` (Plan2Explore) | `algo.policy.actor_type` |
| `algo.player.discrete_size` (DreamerV2, DreamerV3) | removed: it was `algo.world_model.discrete_size` |

### Observations and rewards

The observations an algorithm reads are declared as inputs, each a group of observation keys with the transforms of every key and one network; the algorithms with a decoder declare their outputs the same way. Check the [inputs and outputs howto](./inputs_and_outputs.md).

| 0.8.3 | 1.0 |
|---|---|
| `algo.cnn_keys.encoder` | `algo.inputs.image.keys` |
| `algo.mlp_keys.encoder` | `algo.inputs.vector.keys` |
| `algo.cnn_keys.decoder` | `algo.outputs.image.keys` (by default the keys of the image input) |
| `algo.mlp_keys.decoder` | `algo.outputs.vector.keys` (by default the keys of the vector input) |
| `env.frame_stack`, `env.frame_stack_dilation` | the transform `sheeprl.transforms.FrameStack` (`n`, `dilation`) in `algo.inputs.image.transforms` |
| `env.clip_rewards: True` | the transform `sheeprl.transforms.Tanh` in `algo.rewards.transforms` |
| `algo.world_model.encoder`, `algo.world_model.observation_model` (Dreamers, Plan2Explore) | `algo.inputs.*.net`, `algo.outputs.*.net`; the size presets set `algo.inputs.image.net.channels_multiplier` |
| `algo.world_model.encoder.symlog_inputs` (DreamerV3.5) | the transform `sheeprl.transforms.Symlog` in `algo.inputs.vector.transforms` |
| the observation of an environment that returns a bare array, under the cnn or mlp key you set | the key `state` |
| the render added as an observation when a cnn key was not an observation | `env.render_key`, e.g. `env.render_key=rgb algo.inputs.image.keys=[rgb]` |
| the action masks of MineDojo, the keys starting with `mask` | `algo.extra_keys` |

The standardization of the observations and the scaling of the rewards by the standard deviation of the discounted return (the `VecNormalize` of Stable-Baselines3, for the on-policy algorithms) are transforms too: `sheeprl.transforms.Standardize` in the transforms of an input and `sheeprl.transforms.ScaleByReturnStd` in `algo.rewards.transforms`.

The Dreamers and Plan2Explore no longer force `env.screen_size=64` and `env.frame_stack=-1`: their image inputs resize the images to `env.screen_size` and don't stack them.

### New keys

Every other key keeps its default. 1.0 adds:

| Key | Default | |
|---|---|---|
| `algo.compile` | `enabled: False`, `mode: reduce-overhead`, `policy: True` | in `algo/default.yaml`, for every algorithm: compiles the losses (and the step of the policy) with `torch.compile` |
| `buffer.prefetch` | `False` | samples the next batches in a thread while the training uses the current ones |
| `buffer.on_device` | `False` | keeps the replay buffer in the memory of the device of the training |
| `buffer.curious` | `enabled: False`, `c: 10000`, `beta: 0.7`, `alpha: 0.7`, `epsilon: 0.01`, `initial_priority: 100000` | DreamerV3 and DreamerV3.5: draws the sequences with the priorities of Curious Replay |
| `algo.refresh_recurrent_states` | `False` | PPO-recurrent: recomputes the recurrent states of the rollout at every epoch after the first one |
| `algo.inputs`, `algo.outputs`, `algo.encoder._target_`, `algo.extra_keys` | per algorithm | the observations the agent reads and reconstructs (see above) |
| `algo.rewards.transforms` | `[]` | the transforms of the rewards, applied before they are stored |
| `env.render_key` | `null` | the key under which the render of the environment is added to its observations |

## Checkpoints

The checkpoints of 0.x cannot be resumed, finetuned from, evaluated or registered with 1.0: the training state and the replay buffers are saved in other formats. Use sheeprl 0.8.3 for them (`pip install "sheeprl<1"`).

The checkpoints and the replay buffers saved by the release candidates of 1.0 (`1.0.0rc1` to `1.0.0rc6`) cannot be loaded either: the observations are stored by key as the inputs read them (e.g. SAC no longer concatenates them in the buffer), the encoders and decoders are other modules, and the state of the environments in the checkpoints (`checkpoint["env"]`) is the one of the transforms (`{"transforms": ..., "rewards": ...}`).

The checkpoints now record the version of sheeprl that saved them (`sheeprl_version`): loading a checkpoint of another major version (or of 0.x, which didn't record it) raises an error that says so, instead of failing while unpickling it. Within a major version the checkpoints stay compatible.

## Reproducibility

A run of 1.0 with the seed of a run of 0.x doesn't play and train the same: e.g. the samplers draw the batches with their own random number generators, in another order.

## Python API

This concerns only the code that imports sheeprl, e.g. an external algorithm ([howto/register_external_algorithm.md](register_external_algorithm.md)).

### Algorithms

Every algorithm is a `TrainState` (its models and optimizers, whose `state_dict` is the checkpoint) and an `Algorithm` (`build`, `policy`, `writer`, `batches`, `train_step`, optionally `end_iteration`) trained by `sheeprl.core.run(fabric, cfg, algorithm)`: the loops of the algorithms (`main`, `train`) and their helpers are gone. The entry point registered with `@register_algorithm()` still receives `(fabric, cfg)`, and calls `run`. See [howto/register_new_algorithm.md](register_new_algorithm.md).

The data is collected by `sheeprl.core.Collector`, the same for every algorithm: it plays the `Policy` of the algorithm (`policy`) in the environments and gives every step to its `Writer` (`writer`), which writes it in the replay buffer or the rollout. The players of the algorithms are their policies, renamed: `PPOPlayer` is `PPOPolicy`, `SACPlayer` is `SACPolicy`, `SACAEPlayer` is `SACAEPolicy`, `RecurrentPPOPlayer` is `RecurrentPPOPolicy`, and `PlayerDV1`, `PlayerDV2` and `PlayerDV3` are `DreamerV1Policy`, `DreamerV2Policy` and `DreamerV3Policy`. They implement `sheeprl.core.Policy`: `act(obs, greedy)` chooses the actions for the observations of the environments (the most likely ones with `greedy`, e.g. in the test episode, `sheeprl.core.run_test`, which replaces the `test` of every algorithm), `random(env)` plays random actions, `init_states(num_envs)` creates the memory of the policy (e.g. the recurrent state) of `num_envs` environments before their first step, `reset_state(env_idxes)` forgets it for the environments that start a new episode.

The modules are not wrapped in `DistributedDataParallel`: `sheeprl.core.update` averages the gradients of the weights of every optimizer step over the processes.

The observations reach the agents through their inputs (`sheeprl.models.io`): `build_encoder(cfg, obs_space)` builds the encoder of `algo.inputs` (a `MultiEncoder` of `Input`s, each with the `batch` transforms of its keys, one concatenation and one network), whose forward takes the observations as they are stored and whose `keys` are the keys it reads; `MultiDecoder.from_config(cfg.algo.outputs, encoder, latent_size)` builds the decoder, with the targets of its reconstructions (`targets`) and their log-likelihoods (`log_probs`). They replace the `MultiEncoder` and `MultiDecoder` of `sheeprl.models.models`, which took a cnn and an mlp encoder, and the `prepare_obs` and `normalize_obs` helpers of the algorithms: a policy moves the observations of its `keys` to the device as they are (`sheeprl.models.io.to_device`), and the inputs scale them. The networks follow one contract: an encoder takes `input_shape` and exposes `output_dim`, a decoder takes `latent_size` and `output_shape`. The policies and the writers read `keys` instead of `cnn_keys` and `mlp_keys`.

The actions go through an `ActionCodec` (`sheeprl.core.actions`), made from the action space (`ActionCodec.from_space`), which the `build_agent` functions take instead of `actions_dim` and `is_continuous`: its `dims`, `continuous`, the actions for the environments (`to_env`) and for the store (`to_store`), and the stored actions split per action (`split`).

The transforms of the observations and of the rewards are in `sheeprl.transforms`: write your own as in [howto/add_transform.md](add_transform.md).

### Replay buffers

| 0.x | 1.0 |
|---|---|
| `ReplayBuffer`, `SequentialReplayBuffer`, `EnvIndependentReplayBuffer` | `ReplayBuffer`: one storage `[buffer_size, n_envs, ...]` with a write position per environment (`add(data, env_idxes)`), read by a sampler |
| `EpisodeBuffer` | a `ReplayBuffer` read by an `EpisodeSampler` |
| `rb.sample(...)`, `rb.sample_tensors(...)` | `ReplayStore(storage, sampler).sample(batch_size, n_samples)`, on the device, or `sampler.sample(storage, batch_size, n_samples)` |
| transitions, sequences | `TransitionSampler`, `SequenceSampler` (`online` for the online queue of DreamerV3); the replay buffers are `ReplayStore`s built by `sheeprl.core.transition_store` and `sheeprl.core.sequence_store` |
| the minibatches of an on-policy update | `EpochSampler`; the rollout is a `ReplayStore` built by `sheeprl.core.rollout_store` (`read()`, `minibatches(data, epochs)`) |

All of them are in `sheeprl.data`.
