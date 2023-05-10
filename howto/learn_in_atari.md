# Learning with CNNs
Atari games are harder to solve then basic gym environments, since they rely on visual information. This implies using a CNN model instead of an MLP model. Despite this difference, the rest of the training reimains the same.

In this section, we will learn how to train an agent using a CNN model on Atari games.

The code for this section is available in `algos/ppo/ppo_atari.py`.

## Step by step
We start from `ppo_decoupled.py` and copy its code in the new `ppo_atari.py` file.

We decide to add a `feature_extractor` model that takes in input the observations as images and outputs a feature vector. To do this, we need to define a `torch.nn.Module` that uses `torch.nn.Conv2d` and `torch.nn.Linear` layers.

We also need to define a `forward` method that takes in input the observations and returns the feature vector.

```python
import torch.nn
from torch.nn.modules import Conv2d, ReLU


class CnnNet(torch.nn.Module):
    def __init__(self, num_input_layers: int, features_length: int):
        super().__init__()
        self.conv1 = Conv2d(num_input_layers, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(7 * 7 * 64, 512)
        self.fc2 = torch.nn.Linear(512, features_length)
        self.activation = ReLU()

    def forward(self, x: torch.Tensor):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.fc1(x.view(x.size(0), -1)))  # flatten but keep batch dimension 
        x = self.fc2(x)
        return x
```

The output of the `feature_extractor` is then fed to the `actor` and `critic` models. This happens bot in the `player` and in the `trainer` functions.

```diff
+features_length = 512
+feature_extractor = CnnNet(num_input_layers=envs.single_observation_space.shape[0], features_length=features_length).to(device)

actor = MLP(
- 			input_dims=envs.single_observation_space.shape[0],
+        input_dims=features_length,
        output_dim=envs.single_action_space.n,
        hidden_sizes=(64, 64),
        activation=torch.nn.ReLU,
    ).to(device)
    critic = MLP(
- 			input_dims=envs.single_observation_space.shape[0],
+        input_dims=features_length,
        output_dim=1,
        hidden_sizes=(64, 64),
        activation=torch.nn.ReLU,
    ).to(device)
```

We need to remember to add these parameters to the list of parameters to be optimized and shared by Fabric.

```diff
+    all_parameters = list(feature_extractor.parameters()) + list(actor.parameters()) + list(critic.parameters())
    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.
- 			parameters_to_vector(list(actor.parameters()) + list(critic.parameters()))        
+        parameters_to_vector(all_parameters), device=device
    )
```

Once this is done, we are all set.

We can train the model by running:

```bash
lightning run model --accelerator=cpu --strategy=ddp --devices=2 main.py ppo_atari --env_id Pong-v4
```

:warning: **Note**: remember to install the Atari environments by running 
```bash
poetry add gymnasium[atari]
poetry add gymnasium[accept-rom-license]
```
