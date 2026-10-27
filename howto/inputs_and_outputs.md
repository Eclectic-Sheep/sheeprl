# Inputs and outputs

This page explains how an algorithm reads the observations of the environment (its *inputs*) and, for the algorithms
with a decoder, how it reconstructs them (its *outputs*). It replaces the former `algo.cnn_keys` and `algo.mlp_keys`.

## Vocabulary

- **Observation key**: the name of one observation of the environment, e.g. `rgb`, `state`, `frame`, `fov`. The
  environments return a dictionary of observations; an environment that returns a single array is wrapped so that
  its observation is the key `state`.
- **Transform**: one step of what happens to the data of a key between the environment and the network, e.g. a
  resize, a frame stack, a division by 255, a running standardization. See [how to add a transform](./add_transform.md).
- **Placement** of a transform: where it runs, named after the unit of data it sees:
  - `frame`: in the process of the environment, once per step, on the observation of one environment (NumPy). For
    the steps that change the shape or lose information, done once and stored compactly (resize, frame stack,
    flatten, one-hot).
  - `envs`: in the main process, on the observations of all the environments of the process, one row each (NumPy).
    For the steps with running statistics (e.g. `Standardize`), which are synced across the processes and saved in
    the checkpoints.
  - `batch`: on the device, inside the forward of the input, on a tensor of any leading shape. For the cheap
    elementwise steps that must not inflate the storage (to float, divide by 255, symlog) and the learned ones.
- **Pipeline**: the transforms of one key, always in the order `frame`, then `envs`, then `batch`. The store (rollout
  or replay buffer) holds the data after the `envs` stage, the network is sized from the data after the `batch` stage.
- **Phase**: `act` when the policy chooses the actions in the environments, `train` for a training batch. A `batch`
  transform runs in the phases it lists (e.g. a data augmentation only in `train`).
- **Input**: a group of keys encoded together: their `batch` transforms, one concatenation, one network.
- **Output**: a group of keys reconstructed together from the latent state: one network, its output split per key,
  and optionally a *likelihood* that scores the reconstruction of every key.

```
environment process          main process                              device
raw obs --frame stage--> obs --envs stage--> store --batch stage--> concatenation --> net --> features
```

## The configuration

Every algorithm declares its inputs under `algo.inputs`; the algorithms with a decoder (SAC-AE, the Dreamers and
Plan2Explore) declare their outputs under `algo.outputs`. Every algorithm ships two inputs, named `image` and `vector`,
with no keys: the experiment (or the command line) sets the keys. An input with no keys is skipped.

```yaml
algo:
  inputs:
    image:
      keys: []                 # the observation keys of the input
      transforms:              # the pipeline of every key, Hydra objects as `env.wrapper`
        - {_target_: sheeprl.transforms.Image, size: "${env.screen_size}", grayscale: "${env.grayscale}"}
        - {_target_: sheeprl.transforms.Scale, divisor: 255, shift: 0.5}
      net: {_target_: sheeprl.models.NatureCNN, features_dim: 512}
      # cat_dim: -3            # optional: the axis of the concatenation of the keys
    vector:
      keys: []
      transforms: []
      net: {_target_: sheeprl.models.MLPEncoder, features_dim: 64}
  outputs: {}                  # the algorithms with a decoder, see below
  extra_keys: []               # keys stored but not encoded, e.g. the action masks of MineDojo
  rewards:
    transforms: []             # the transforms of the rewards, see below
```

The fields of an input:

- `keys`: the observation keys, in order. Every key must be an observation of the environment: the error lists the
  keys the environment has.
- `transforms`: the transforms of every key of the input. A transform runs at its default placement, or at the one
  you give with `where: ...` among the ones it supports.
- `net`: the network, built with the shape of one sample after the concatenation (`input_shape`). Its arguments are
  the ones you give plus `input_shape`.
- `cat_dim`: the axis along which the keys are concatenated. By default it is `-3` (the channels) when the keys are
  3D after the `batch` stage, and `-1` when they are 1D; mixing 3D and 1D keys in one input is an error (flatten the
  3D ones first, with `sheeprl.transforms.Flatten`).

The features of the inputs are concatenated, in the order of the configuration, into the features the agent reads.
Every key is stored once, even if two inputs list it: its `frame` and `envs` stages must then be identical in both,
only the `batch` stage may differ.

The keys that no input and no `algo.extra_keys` list are dropped by the environment: they are neither transformed nor
stored.

### Select the observations

Train PPO on CartPole from its vector observation (a bare `Box`, so its key is `state`):

```bash
python sheeprl.py exp=ppo env=gym env.id=CartPole-v1 algo.inputs.vector.keys=[state]
```

Train it from the pixels of the render: `env.render_key` names the key under which the render is added to the
observations.

```bash
python sheeprl.py exp=ppo env=gym env.id=CartPole-v1 env.render_key=rgb algo.inputs.image.keys=[rgb]
```

Read the image and two vectors of a DIAMBRA game:

```bash
diambra run python sheeprl.py exp=ppo env=diambra env.id=doapp env.num_envs=1 \
  algo.inputs.image.keys=[frame] algo.inputs.vector.keys=[opp_health,own_health]
```

The observations of an environment, as the environment returns them, after the `frame` stage and after the `envs`
stage (what the store holds), are printed by:

```bash
python examples/observation_space.py agent=dreamer_v3 env=atari env.id=MsPacmanNoFrameskip-v4 algo.inputs.image.keys=[state]
```

### Frame stacking and standardization

They are transforms. The frames of an image input are stacked by `FrameStack`, a `frame` transform that concatenates
the last `n` frames along the channel axis. A list in an experiment replaces the whole list of the defaults, so an
experiment that stacks the frames lists all the image transforms:

```yaml
algo:
  inputs:
    image:
      keys: [rgb]
      transforms:
        - {_target_: sheeprl.transforms.Image, size: 64, grayscale: false}
        - {_target_: sheeprl.transforms.FrameStack, n: 4}
        - {_target_: sheeprl.transforms.Scale, divisor: 255, shift: 0.5}
```

One element of a list can be changed from the command line by its index, e.g. `algo.inputs.image.transforms.1.n=2`.

The vector observations are standardized with their running mean and variance by `Standardize`, an `envs` transform
whose statistics are synced across the processes and saved in the checkpoints. It learns from the data, so only the
on-policy algorithms accept it: the replay buffer of an off-policy algorithm would keep observations standardized
with statistics that change.

```bash
python sheeprl.py exp=ppo env=gym env.id=CartPole-v1 algo.inputs.vector.keys=[state] \
  "algo.inputs.vector.transforms=[{_target_:sheeprl.transforms.Standardize,clip:10.0}]"
```

### A new input

An input other than `image` and `vector` is added with `+` from the command line, or in an experiment file. For
example, the grid of 0 and 1 of [issue #237](https://github.com/Eclectic-Sheep/sheeprl/issues/237), `fov` of space
`Box(0, 1, (5, 5, 5), float32)`, read as it is by small convolutions, and a 2D plane flattened into the vector input:

```yaml
algo:
  inputs:
    image:
      keys: []
    grid:
      keys: [fov]
      transforms: []        # no resize, no grayscale, no division by 255
      net: {_target_: sheeprl.models.ConvEncoder, channels: [32, 64], kernel_size: 3, stride: 1, padding: 1}
    vector:
      keys: [plane]
      transforms: [{_target_: sheeprl.transforms.Flatten}]
```

### The rewards

The rewards have their own pipeline, `algo.rewards.transforms`, of `envs` transforms only, applied before the rewards
are stored. It replaces `env.clip_rewards` (now `sheeprl.transforms.Tanh`). The rewards divided by the running
standard deviation of the discounted return (the reward normalization of the `VecNormalize` of Stable-Baselines3),
then squashed by the `tanh`, are, in this order:

```yaml
algo:
  rewards:
    transforms:
      - {_target_: sheeprl.transforms.ScaleByReturnStd, gamma: "${algo.gamma}", clip: 10.0, epsilon: 1e-8}
      - {_target_: sheeprl.transforms.Tanh}
```

`Sign` (the rewards of the Atari games as in DQN) and `Clip(low, high)` are the other ones shipped. `ScaleByReturnStd`
learns from the data, so only the on-policy algorithms accept it.

## Outputs

The algorithms with a decoder reconstruct the keys of their outputs from the latent state. An output has:

- `keys`: the keys it reconstructs. A reconstructed key must be read by an input: its shape is the one the input gives
  it after the `batch` stage, and its target is the key in that same domain (e.g. the images in [-0.5, 0.5] when the
  input scales them so).
- `net`: the network, built with `latent_size` (the size of the latent state) and `output_shape` (the shape of the
  concatenation of the keys).
- `likelihood` (the Dreamers and Plan2Explore): a partial Hydra object called as `likelihood(reconstruction, dims)`
  that returns the distribution of the reconstruction of a key over its last `dims` dimensions; the reconstruction
  loss is the sum over the keys of the log-probability of their targets.
- `transforms` (optional): `batch` transforms of the targets of the output, instead of the ones of the input, e.g.
  DreamerV3.5 reconstructs the images in [0, 1] while its encoder reads them in [-0.5, 0.5].
- `cat_dim` (optional): as for the inputs.

By default the outputs reconstruct the keys of the inputs of the same name:

```yaml
algo:
  outputs:
    image:
      keys: ${algo.inputs.image.keys}
      net: {_target_: sheeprl.algos.dreamer_v3.agent.CNNDecoder, channels_multiplier: 96, stages: 4}
      likelihood: {_target_: sheeprl.utils.distribution.MSEDistribution, _partial_: true}
    vector:
      keys: ${algo.inputs.vector.keys}
      net: {_target_: sheeprl.algos.dreamer_v3.agent.MLPDecoder}
      likelihood: {_target_: sheeprl.utils.distribution.MSEDistribution, _partial_: true, tol: 1e-8}
```

To reconstruct only some of the keys, set the keys of the output, which must be among the ones of the inputs:

```bash
python sheeprl.py exp=dreamer_v3 env=minerl env.id=custom_navigate \
  algo.inputs.vector.keys=[life_stats,inventory,max_inventory] algo.outputs.vector.keys=[life_stats,inventory]
```

The grid of the example above, reconstructed by an MLP whose 125 outputs are reshaped to the grid and read as the
logits of its cells:

```yaml
algo:
  outputs:
    grid:
      keys: [fov]
      net: {_target_: sheeprl.algos.dreamer_v3.agent.MLPDecoder, mlp_layers: 2, dense_units: 256}
      likelihood: {_target_: sheeprl.utils.distribution.BernoulliLogits, _partial_: true}
```

SAC-AE has no likelihood: it scores its reconstructions with a mean squared error. Its output `image` declares its own
target transform, `sheeprl.transforms.Dequantize`: the pixels of the store reduced to 5 bits, with a uniform noise,
in [-0.5, 0.5]. Its other outputs reconstruct the keys as its encoder reads them.

## The algorithms

| Algorithm | Inputs | Outputs |
| --- | --- | --- |
| A2C, PPO, PPO Recurrent | any | none |
| SAC, DroQ | vectors only: the network of the input must have no parameters (`sheeprl.models.IdentityEncoder`); a 2D or 3D key is flattened with `Flatten` | none |
| SAC-AE | any | the keys of its inputs, by default |
| Dreamer V1, V2, V3, V3.5, Plan2Explore | any | the keys of its inputs, by default, each with a likelihood |

The Dreamers have no frame stacking (their recurrent model remembers the past) and their vector input applies
`Symlog` (DreamerV3 and V3.5). The image networks of DreamerV3 and V3.5 take images whose sides are divisible by
`2 ** stages`; the image decoders of DreamerV1 and V2 reconstruct 64 x 64 images.

## Networks

A network is any `nn.Module` with this contract, so that it can be referenced from the configuration by its import
path:

- An **encoder** takes `input_shape` (the shape of one sample after the concatenation, without the batch dimensions)
  and exposes `output_dim: int`, the size of its features. Its forward takes any number of leading dimensions (the
  Dreamers pass `[sequence, batch, ...]`; `sheeprl.utils.model.cnn_forward` is a helper for the convolutions).
- A **decoder** takes `latent_size` and `output_shape` and returns tensors of shape `[..., *output_shape]`.

Shipped encoders: `sheeprl.models.NatureCNN`, `sheeprl.models.MLPEncoder`, `sheeprl.models.ConvEncoder` (small
convolutions for grids and small images, no resize) and `sheeprl.models.IdentityEncoder` (the flattened input), plus
the ones of every algorithm (e.g. `sheeprl.algos.dreamer_v3.agent.CNNEncoder`).

The checks of `sheeprl.transforms` build and run what you write:

```python
from sheeprl.transforms import check_input, check_net

check_net(MyEncoder(input_shape=(5, 5, 5)), (5, 5, 5))  # `output_dim` and the leading dimensions
check_input(cfg.algo.inputs.grid, store_space, compile=True)  # the input runs in both phases, in one compiled graph
```

The encoder of the inputs and the decoder of the outputs are classes too, set by `algo.encoder._target_` and
`algo.decoder._target_`. By default `sheeprl.models.io.MultiEncoder` concatenates the features of the inputs and
`sheeprl.models.io.MultiDecoder` reconstructs every output with its own network. Another class must subclass
`sheeprl.models.io.Encoder` or `sheeprl.models.io.Decoder`, the contracts the algorithms rely on:

- an encoder has `keys` (what the policies read and the writers store), `output_dim`, `forward(obs, phase)`,
  `targets(obs, keys)`, `store_space(key)`, `batch_space(key)`, and `named_keys()` and `named_nets()`, the keys and
  the network of every input by name;
- a decoder has `keys`, `forward(latent)`, which returns the reconstruction of every key, `targets(obs, encoder)`,
  `log_probs(reconstructions, targets)`, and `named_keys()` and `named_nets()`.

Both are built by their `from_config` class method, from `algo.inputs` and the spaces of the store, or from
`algo.outputs`, the encoder and the size of the latent state.

## Checkpoints

The checkpoint of a run holds what was learned from the data: the statistics of the `envs` transforms of every key
and of the rewards (`checkpoint["env"]`, restored on resume and applied, frozen, by the evaluation) and the parameters
of the `batch` transforms (with the agent). The `frame` stage is not saved: its state (e.g. the stacked frames) is the
one of an episode, restarted on resume.
