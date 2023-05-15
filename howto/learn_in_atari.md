# Learning with CNNs
Atari games are harder to solve then basic gym environments, since they rely on visual information. This implies using a CNN model instead of an MLP model. Despite this difference, the rest of the training reimains the same.

In this section, we will learn how to train an agent using a CNN model on Atari games.

The code for this section is available in `algos/ppo/ppo_atari.py`.

## Install Atari environments
First we should install the Atari environments with:

```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```

For more information: https://gymnasium.farama.org/environments/atari/ 

## Step by step
We start from `ppo_decoupled.py` and copy its code in the new `ppo_atari.py` file.

We decide to add a `feature_extractor` model that takes in input the observations as images and outputs a feature vector. To do this, we need to define a `torch.nn.Module` that uses `torch.nn.Conv2d` and `torch.nn.Linear` layers.

We also need to define a `forward` method that takes in input the observations and returns the feature vector.

```python
import torch.nn.functional as F
from torch import Tensor, nn


class NatureCNN(nn.Module):
    def __init__(self, in_channels: int, features_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, features_dim)

    def forward(self, x: Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.flatten(1)))  # flatten but keep batch dimension 
        x = self.fc2(x)
        return x
```

The output of the `feature_extractor` is then fed to the `actor` and `critic` models. This happens both in the `player` and in the `trainer` functions.

```diff
+features_dim = 512
+feature_extractor = NatureCNN(in_channels=4, features_dim=features_dim)  # '4' is the number of skipped frames by default by the AtariPreprocessing wrapper

actor = MLP(
- 	input_dims=envs.single_observation_space.shape[0],
+   input_dims=features_dim,
    output_dim=envs.single_action_space.n,
    hidden_sizes=(64, 64),
    activation=torch.nn.ReLU,
).to(device)
critic = MLP(
- 	input_dims=envs.single_observation_space.shape[0],
+   input_dims=features_dim,
    output_dim=1,
    hidden_sizes=(64, 64),
    activation=torch.nn.ReLU,
).to(device)
```

We need to remember to add these parameters to the list of parameters to be optimized and shared by Fabric

```diff
+    all_parameters = list(feature_extractor.parameters()) + list(actor.parameters()) + list(critic.parameters())
    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.
- 			parameters_to_vector(list(actor.parameters()) + list(critic.parameters()))        
+        parameters_to_vector(all_parameters), device=device
    )
```

and to extract the features every time before calling the actor or the critic

```python
features = feature_extractor(next_obs)
actions_logits = actor(features)
values = critic(features)
```

We also need to assure that we correctly pre-process the observation coming from the environment, pratically redefining the `make_env` function, wrapping the environment with the `gymnasium.wrappers.atari_preprocess.AtariPreprocess` wrapper:

```python
def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    prefix: str = "",
    vector_env_idx: int = 0
):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if vector_env_idx == 0 and idx == 0:
                env = gym.experimental.wrappers.RecordVideoV0(
                    env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
                )
        env = AtariPreprocessing(env, grayscale_obs=True, grayscale_newaxis=False, scale_obs=True)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
```

Once this is done, we are all set.

We can train the model by running:

```bash
lightning run model --accelerator=cpu --strategy=ddp --devices=2 main.py ppo_atari --env_id PongNoFrameskip-v0
```