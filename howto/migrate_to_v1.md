# Migrate from sheeprl 0.x to 1.0

sheeprl 1.0 trains every algorithm with one shared core (`sheeprl.core`) and one replay buffer read by samplers (`sheeprl.data`). What you run from the command line works as before: the changes concern the checkpoints, the reproducibility of a run, and the code that imports sheeprl.

## Command line and configurations

The commands (`sheeprl`, `sheeprl-eval`, `sheeprl-registration`) and the configurations are the ones of 0.8.3: every key keeps its default. 1.0 adds:

| Key | Default | |
|---|---|---|
| `algo.compile` | `enabled: False`, `mode: reduce-overhead`, `policy: True` | in `algo/default.yaml`, for every algorithm: compiles the losses (and the step of the policy) with `torch.compile` |
| `buffer.prefetch` | `False` | samples the next batches in a thread while the training uses the current ones |
| `buffer.on_device` | `False` | keeps the replay buffer in the memory of the device of the training |
| `algo.refresh_recurrent_states` | `False` | PPO-recurrent: recomputes the recurrent states of the rollout at every epoch after the first one |

## Checkpoints

The checkpoints of 0.x cannot be resumed, finetuned from, evaluated or registered with 1.0: the training state and the replay buffers are saved in other formats. Use sheeprl 0.8.3 for them (`pip install "sheeprl<1"`).

The checkpoints now record the version of sheeprl that saved them (`sheeprl_version`): loading a checkpoint of another major version (or of 0.x, which didn't record it) raises an error that says so, instead of failing while unpickling it. Within a major version the checkpoints stay compatible.

## Reproducibility

A run of 1.0 with the seed of a run of 0.x doesn't play and train the same: e.g. the samplers draw the batches with their own random number generators, in another order.

## Python API

This concerns only the code that imports sheeprl, e.g. an external algorithm ([howto/register_external_algorithm.md](register_external_algorithm.md)).

### Algorithms

Every algorithm is a `TrainState` (its models and optimizers, whose `state_dict` is the checkpoint) and an `Algorithm` (`build`, `policy`, `writer`, `batches`, `train_step`, optionally `end_iteration`) trained by `sheeprl.core.run(fabric, cfg, algorithm)`: the loops of the algorithms (`main`, `train`) and their helpers are gone. The entry point registered with `@register_algorithm()` still receives `(fabric, cfg)`, and calls `run`. See [howto/register_new_algorithm.md](register_new_algorithm.md).

The data is collected by `sheeprl.core.Collector`, the same for every algorithm: it plays the `Policy` of the algorithm (`policy`) in the environments and gives every step to its `Writer` (`writer`), which writes it in the replay buffer or the rollout. The players of the algorithms are their policies, renamed: `PPOPlayer` is `PPOPolicy`, `SACPlayer` is `SACPolicy`, `SACAEPlayer` is `SACAEPolicy`, `RecurrentPPOPlayer` is `RecurrentPPOPolicy`, and `PlayerDV1`, `PlayerDV2` and `PlayerDV3` are `DreamerV1Policy`, `DreamerV2Policy` and `DreamerV3Policy`. They implement `sheeprl.core.Policy`: `act(obs)` chooses the actions for the observations of the environments, `random(env)` plays random actions, `reset(env_idxes)` resets the state of the environments that start a new episode (e.g. the recurrent state).

The modules are not wrapped in `DistributedDataParallel`: `sheeprl.core.update` averages the gradients of the weights of every optimizer step over the processes.

### Replay buffers

| 0.x | 1.0 |
|---|---|
| `ReplayBuffer`, `SequentialReplayBuffer`, `EnvIndependentReplayBuffer` | `ReplayBuffer`: one storage `[buffer_size, n_envs, ...]` with a write position per environment (`add(data, env_idxes)`), read by a sampler |
| `EpisodeBuffer` | a `ReplayBuffer` read by an `EpisodeSampler` |
| `rb.sample(...)`, `rb.sample_tensors(...)` | `ReplayStore(storage, sampler).sample(batch_size, n_samples)`, on the device, or `sampler.sample(storage, batch_size, n_samples)` |
| transitions, sequences | `TransitionSampler`, `SequenceSampler` (`online` for the online queue of DreamerV3) |
| the minibatches of an on-policy update | `EpochSampler`; the rollout is a `ReplayStore` built by `sheeprl.core.rollout_store` (`read()`, `minibatches(data, epochs)`) |

All of them are in `sheeprl.data`.
