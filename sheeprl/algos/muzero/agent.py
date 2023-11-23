from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn

from sheeprl.models.models import MLP
from sheeprl.utils.model import ModuleType

# class CNNEncoder(nn.Module):
#     """The Dreamer-V3 image encoder. This is composed of 4 `nn.Conv2d` with
#     kernel_size=3, stride=2 and padding=1. No bias is used if a `nn.LayerNorm`
#     is used after the convolution. This 4-stages model assumes that the image
#     is a 64x64 and it ends with a resolution of 4x4. If more than one image is to be encoded, then those will
#     be concatenated on the channel dimension and fed to the encoder.
#
#     Args:
#         keys (Sequence[str]): the keys representing the image observations to encode.
#         input_channels (Sequence[int]): the input channels, one for each image observation to encode.
#         image_size (Tuple[int, int]): the image size as (Height,Width).
#         channels_multiplier (int): the multiplier for the output channels. Given the 4 stages, the 4 output channels
#             will be [1, 2, 4, 8] * `channels_multiplier`.
#         layer_norm (bool, optional): whether to apply the layer normalization.
#             Defaults to True.
#         activation (ModuleType, optional): the activation function.
#             Defaults to nn.SiLU.
#         stages (int, optional): how many stages for the CNN.
#     """
#
#     def __init__(
#         self,
#         keys: Sequence[str],
#         input_channels: Sequence[int],
#         image_size: Tuple[int, int],
#     ) -> None:
#         super().__init__()
#         self.keys = keys
#         self.input_dim = (sum(input_channels), *image_size)
#         self.model = nn.Sequential(
#             nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
#             ResidualBlock(128, 128),
#             ResidualBlock(128, 128),
#             nn.Conv2d(input_channels, 256, kernel_size=3, stride=2, padding=1, bias=False),
#             ResidualBlock(256, 256),
#             ResidualBlock(256, 256),
#             ResidualBlock(256, 256),
#             nn.AvgPool2d(3, stride=2),
#             ResidualBlock(256, 256),
#             ResidualBlock(256, 256),
#             ResidualBlock(256, 256),
#             nn.AvgPool2d(3, stride=2),
#
#             nn.Flatten(-3, -1),
#         )
#         with torch.no_grad():
#             self.output_dim = self.model(torch.zeros(1, *self.input_dim)).shape[-1]
#
#     def forward(self, obs: Dict[str, Tensor]) -> Tensor:
#         x = torch.cat([obs[k] for k in self.keys], -3)  # channels dimension
#         return cnn_forward(self.model, x, x.shape[-3:], (-1,))


class MLPEncoder(nn.Module):
    """The Dreamer-V3 vector encoder. This is composed of N `nn.Linear` layers, where
    N is specified by `mlp_layers`. No bias is used if a `nn.LayerNorm` is used after the linear layer.
    If more than one vector is to be encoded, then those will concatenated on the last
    dimension before being fed to the encoder.

    Args:
        keys (Sequence[str]): the keys representing the vector observations to encode.
        input_dims (Sequence[int]): the dimensions of every vector to encode.
        mlp_layers (int, optional): how many mlp layers.
            Defaults to 4.
        dense_units (int, optional): the dimension of every mlp.
            Defaults to 512.
        layer_norm (bool, optional): whether to apply the layer normalization.
            Defaults to True.
        activation (ModuleType, optional): the activation function after every layer.
            Defaults to nn.ELU.

    """

    def __init__(
        self,
        keys: Sequence[str],
        input_dims: Sequence[int],
        output_dim: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        layer_norm: bool = False,
        activation: ModuleType = nn.ELU,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = sum(input_dims)
        self.model = MLP(
            self.input_dim,
            output_dim,
            [dense_units] * mlp_layers,
            activation=activation,
            layer_args={"bias": not layer_norm},
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units, "eps": 1e-3} for _ in range(mlp_layers)]
            if layer_norm
            else None,
        )
        self.output_dim = dense_units

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], -1)
        return self.model(x)


class MuzeroAgent(torch.nn.Module):
    """
    Muzero agent.

    Args:
        representation: torch.nn.Module
            Representation function, map the observation to a latent space (hidden_state).
            Used to start the imagination process during MCTS.
        prediction: torch.nn.Module
            Prediction function, map the hidden state to policy logits and value.
            Used to predict the policy and value during MCTS, based on the latent space representation of observations.
        dynamics: torch.nn.Module
            Dynamics function, map an action and a hidden state to the imagined reward and next hidden state.
            Used to imagine the response (evolution) of the environment during MCTS.
    """

    def __init__(
        self,
        cfg,
        input_dims: Sequence[int],
        num_actions: int,
        full_support_size: int,
    ):
        super().__init__()
        model_cfg = cfg.algo.model
        representation_cfg = model_cfg.encoder
        prediction_cfg = model_cfg.prediction
        dynamics_cfg = model_cfg.dynamics
        self.representation = MLPEncoder(
            keys=cfg.mlp_keys,
            input_dims=input_dims,
            output_dim=model_cfg.hidden_state_size,
            mlp_layers=representation_cfg.mlp_layers,
            dense_units=representation_cfg.dense_units,
            layer_norm=representation_cfg.layer_norm,
            activation=representation_cfg.activation,
        )
        self.prediction = Predictor(
            embedding_size=model_cfg.hidden_state_size,
            policy_hidden_sizes=prediction_cfg.policy_hidden_sizes,
            value_hidden_sizes=prediction_cfg.value_hidden_sizes,
            num_actions=num_actions,
            full_support_size=full_support_size,
        )
        self.dynamics = Dynamics(
            num_actions=num_actions,
            embedding_size=model_cfg.hidden_state_size,
            state_hidden_sizes=dynamics_cfg.state_hidden_sizes,
            rew_hidden_sizes=dynamics_cfg.reward_hidden_sizes,
            full_support_size=full_support_size,
        )

    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the initial hidden state and the initial prediction (policy and value) from the observation.
        To run to initialize MCTS.
        """
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state.unsqueeze(0), policy_logits, value

    def recurrent_inference(
        self, action: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and the next prediction (policy and value)
        from the action and the previous hidden state.
        To run at each simulation of MCTS.
        """
        reward, next_hidden_state = self.dynamics(action, hidden_state)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value

    def forward(self, action, hidden_state):
        return self.recurrent_inference(action, hidden_state)

    @torch.no_grad()
    def gradient_norm(self):
        """Compute the norm of the parameters' gradients."""
        total_norm = 0
        p_with_grads = [p for p in self.parameters() if p.grad is not None]
        if not p_with_grads:
            raise RuntimeError("No parameters have gradients. Run the backward method first.")
        for p in p_with_grads:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm**0.5


# class GruMlpDynamics(torch.nn.Module):
#     def __init__(self, num_actions, embedding_size, mlp_hidden_sizes, full_support_size):
#         super().__init__()
#         self.num_actions = num_actions
#         self.gru = torch.nn.GRU(input_size=num_actions, hidden_size=embedding_size)
#         self.mlp = MLP(
#             input_dims=embedding_size,
#             hidden_sizes=mlp_hidden_sizes,
#             activation=torch.nn.ELU,
#             output_dim=full_support_size,
#         )
#
#     def forward(self, x, h):
#         one_hot_action = torch.nn.functional.one_hot(x.long(), num_classes=self.num_actions).float().transpose(0, 1)
#         y, h_next = self.gru(one_hot_action, h)
#         y = self.mlp(y)
#         return y, h_next


class Dynamics(torch.nn.Module):
    def __init__(
        self,
        num_actions,
        embedding_size,
        state_hidden_sizes,
        rew_hidden_sizes,
        full_support_size,
        state_activation_fn=torch.nn.ELU,
        rew_activation_fn=torch.nn.ELU,
    ):
        super().__init__()
        self.hstate = MLP(
            input_dims=embedding_size + int(num_actions),
            hidden_sizes=state_hidden_sizes,
            output_dim=embedding_size,
            activation=state_activation_fn,
        )
        self.mlp = MLP(
            input_dims=embedding_size + int(num_actions),
            hidden_sizes=rew_hidden_sizes,
            activation=rew_activation_fn,
            output_dim=full_support_size,
        )
        self.num_actions = num_actions

    def forward(self, x, h0):
        one_hot_action = torch.nn.functional.one_hot(x.long(), num_classes=self.num_actions).float().transpose(0, 1)
        cat_input = torch.cat([one_hot_action, h0], dim=-1)
        h1 = self.hstate(cat_input)
        y = self.mlp(cat_input)
        return y, h1


class Predictor(torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        policy_hidden_sizes,
        value_hidden_sizes,
        num_actions,
        full_support_size,
    ):
        super().__init__()
        self.mlp_actor = MLP(
            input_dims=embedding_size, hidden_sizes=policy_hidden_sizes, activation=torch.nn.ELU, output_dim=num_actions
        )

        self.mlp_value = MLP(
            input_dims=embedding_size,
            hidden_sizes=value_hidden_sizes,
            activation=torch.nn.ELU,
            output_dim=full_support_size,
        )

    def forward(self, x):
        policy = self.mlp_actor(x)
        value = self.mlp_value(x)
        return policy, value


# class RecurrentMuzero(MuzeroAgent):
#     def __init__(self, hidden_state_size=256, num_actions=4):
#         super().__init__()
#         self.representation = NatureCNN(in_channels=3, features_dim=hidden_state_size)
#         self.dynamics: torch.nn.Module = GruMlpDynamics(mlp_hidden_sizes=hidden_state_size)
#         self.prediction: torch.nn.Module = Predictor(policy_hidden_sizes=hidden_state_size, num_actions=num_actions)


# if __name__ == "__main__":
#     batch_size = 32
#     sequence_len = 5
#     agent = RecurrentMuzero()
#     # Player:
#     print("Player:")
#     observation = torch.rand(1, 3, 64, 64)
#     hidden_state, policy_logits, value = agent.initial_inference(observation)
#     print("Initial inference:")
#     print(hidden_state.shape)
#     print(policy_logits.shape)
#     print(value.shape)

# action = torch.rand(1, 1, 1)
# hidden_state, policy_logits, reward, value = agent.recurrent_inference(action, hidden_state)
# print("Recurrent inference:")
# print(hidden_state.shape)
# print(policy_logits.shape)
# print(reward.shape)
# print(value.shape)
#
# ## Trainer:
# print("Trainer:")
# observation = torch.rand(batch_size, 3, 64, 64)
# hidden_state, policy_logits, value = agent.initial_inference(observation)
# print("Initial inference:")
# print(hidden_state.shape)
# print(policy_logits.shape)
# print(value.shape)
#
# action = torch.rand(1, batch_size, 1)
# hidden_state, policy_logits, reward, value = agent.recurrent_inference(action, hidden_state)
# print("Recurrent inference:")
# print(hidden_state.shape)
# print(policy_logits.shape)
# print(reward.shape)
# print(value.shape)
# action2 = torch.randint(0, 4, (1, 1)).to(torch.float32)
# last_hidden_state, reward, policy_logits, value = agent.recurrent_inference(action2, next_hidden_state)
# print(last_hidden_state.shape)
# print(reward.shape)
# print(policy_logits.shape)
# print(value.shape)
