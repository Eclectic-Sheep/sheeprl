import copy
from math import prod
from typing import Sequence, SupportsFloat, Tuple, Union

import torch
import torch.nn as nn
from lightning.fabric.wrappers import _FabricModule
from numpy.typing import NDArray
from torch import Size, Tensor

from sheeprl.models.models import CNN, MLP, DeCNN

LOG_STD_MAX = 2
LOG_STD_MIN = -10


class Encoder(CNN):
    """Encoder network from https://arxiv.org/abs/1910.01741

    Args:
        in_channels (int): the input channels to the first convolutional layer
        features_dim (int): the features dimension in output from the last convolutional layer
        screen_size (int, optional): the dimension of the input image as a single integer.
            Needed to extract the features and compute the output dimension after all the
            convolutional layers.
            Defaults to 64.
    """

    def __init__(self, in_channels: int, features_dim: int, screen_size: int = 64):
        super().__init__(
            in_channels,
            [32, 32, 32, 32],
            layer_args=[
                {"kernel_size": 3, "stride": 2},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
            ],
        )

        with torch.no_grad():
            x: Tensor = self.model(
                torch.rand(1, in_channels, screen_size, screen_size, device=self.model[0].weight.device)
            )
            self._conv_output_shape = x.shape[1:]
            flattened_conv_output_dim = x.flatten(1).shape[1]
        self.fc = MLP(
            input_dims=flattened_conv_output_dim,
            hidden_sizes=(features_dim,),
            activation=nn.Tanh,
            norm_layer=nn.LayerNorm,
            norm_args={"normalized_shape": features_dim},
        )
        self._output_dim = features_dim

    @property
    def conv_output_shape(self) -> Size:
        return self._conv_output_shape

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = self.fc(x.flatten(1))
        return x


class Decoder(DeCNN):
    """Decoder network from https://arxiv.org/abs/1910.01741

    Args:
        encoder_conv_output_shape (Size): the output shape of the encoder convolutional layer
        features_dim (int): the features dimension in output from the last convolutional layer
        out_channels (int, optional): the number of channels of the generated image.
            Defaults to 3
        screen_size (int, optional): the dimension of the input image as a single integer.
            Needed to extract the features and compute the output dimension after all the
            convolutional layers.
            Defaults to 64
    """

    def __init__(
        self, encoder_conv_output_shape: Size, features_dim: int, out_channels: int = 3, screen_size: int = 64
    ):
        super().__init__(
            32,
            [32, 32, 32],
            layer_args=[
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
            ],
        )
        self.fc = MLP(input_dims=features_dim, hidden_sizes=(prod(encoder_conv_output_shape),))
        self.to_obs = nn.ConvTranspose2d(
            super().output_dim, out_channels=out_channels, kernel_size=3, stride=2, output_padding=1
        )
        self._output_dim = Size([out_channels, screen_size, screen_size])
        self._encoder_conv_output_shape = encoder_conv_output_shape

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x).view(-1, *self._encoder_conv_output_shape)
        x = self.model(x)
        x = self.to_obs(x)
        return x


class SACPixelCritic(nn.Module):
    def __init__(
        self,
        encoder: Union[Encoder, _FabricModule],
        action_dim: int,
        hidden_dim: int = 256,
        num_critics: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.model = MLP(
            input_dims=encoder.output_dim + action_dim,
            output_dim=num_critics,
            hidden_sizes=(hidden_dim, hidden_dim),
            activation=nn.ReLU,
            flatten_dim=None,
        )

    def forward(self, obs: Tensor, action: Tensor, detach_encoder_features: bool = False) -> Tensor:
        features = self.encoder(obs)
        if detach_encoder_features:
            features = features.detach()
        x = torch.cat([features, action], -1)
        return self.model(x)


class SACPixelContinuousActor(nn.Module):
    def __init__(
        self,
        encoder: Union[Encoder, _FabricModule],
        action_dim: int,
        hidden_dim: int = 256,
        action_low: Union[SupportsFloat, NDArray] = -1.0,
        action_high: Union[SupportsFloat, NDArray] = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.model = MLP(input_dims=encoder.output_dim, hidden_sizes=(hidden_dim, hidden_dim), flatten_dim=None)
        self.fc_mean = nn.Linear(self.model.output_dim, action_dim)
        self.fc_logstd = nn.Linear(self.model.output_dim, action_dim)

        # Action rescaling buffers
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def forward(self, obs: Tensor, detach_encoder_features: bool = False) -> Tuple[Tensor, Tensor]:
        """Given an observation, it returns a tanh-squashed
        sampled action (correctly rescaled to the environment action bounds) and its
        log-prob (as defined in Eq. 26 of https://arxiv.org/abs/1812.05905)

        Args:
            obs (Tensor): the observation tensor

        Returns:
            tanh-squashed action, rescaled to the environment action bounds
            action log-prob
        """
        features = self.encoder(obs)
        if detach_encoder_features:
            features = features.detach()
        x = self.model(features)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        return self.get_actions_and_log_probs(mean, std)

    def get_actions_and_log_probs(self, mean: Tensor, std: Tensor):
        """Given the mean and the std of a Normal distribution, it returns a tanh-squashed
        sampled action (correctly rescaled to the environment action bounds) and its
        log-prob (as defined in Eq. 26 of https://arxiv.org/abs/1812.05905)

        Args:
            mean (Tensor): the mean of the distribution
            std (Tensor): the standard deviation of the distribution

        Returns:
            tanh-squashed action, rescaled to the environment action bounds
            action log-prob
        """
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()

        # Squash sample
        y_t = torch.tanh(x_t)

        # Action sampled from a Tanh transformed Gaussian distribution
        action = y_t * self.action_scale + self.action_bias

        # Change of variable for probability distributions
        # Eq. 26 of https://arxiv.org/abs/1812.05905
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)

        # Log-prob of independent actions is the sum of the log-probs
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def get_greedy_actions(self, obs: Tensor) -> Tensor:
        """Get the action given the input observation greedily

        Args:
            obs (Tensor): input observation

        Returns:
            action
        """
        features = self.feature_extractor(obs)
        x = self.after_conv(features)
        x = self.model(x)
        mean = self.fc_mean(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean


class SACPixelAgent(nn.Module):
    def __init__(
        self,
        actor: Union[SACPixelContinuousActor, _FabricModule],
        critics: Sequence[Union[SACPixelCritic, _FabricModule]],
        target_entropy: float,
        alpha: float = 1.0,
        tau: float = 0.01,
        encoder_tau: float = 0.05,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        # Tie encoder weights between actor and first critic
        for actor_layer, critic_layer in zip(actor.encoder.modules(), critics[0].encoder.modules()):
            if isinstance(actor_layer, nn.Conv2d) and isinstance(actor_layer, nn.Conv2d):
                actor_layer.weight = critic_layer.weight
                actor_layer.bias = critic_layer.bias
        for i in range(1, len(critics)):
            for c0_layer, ci_layer in zip(critics[0].encoder.modules(), critics[i].encoder.modules()):
                if isinstance(c0_layer, nn.Conv2d) and isinstance(ci_layer, nn.Conv2d):
                    ci_layer.weight = c0_layer.weight
                    ci_layer.bias = c0_layer.bias

        # Actor and critics
        self._num_critics = len(critics)
        self._actor = actor
        self._qfs = nn.ModuleList(critics)

        # Create target critic unwrapping the DDP module from the critics to prevent
        # `RuntimeError: DDP Pickling/Unpickling are only supported when using DDP with the default process group.
        # That is, when you have called init_process_group and have not passed process_group argument to DDP constructor`.
        # This happens when we're using the decoupled version of SACPixel for example
        qfs_unwrapped_modules = []
        for critic in critics:
            if getattr(critic, "module"):
                critic_module = critic.module
            else:
                critic_module = critic
            qfs_unwrapped_modules.append(critic_module)
        self._qfs_unwrapped = nn.ModuleList(qfs_unwrapped_modules)
        self._qfs_target = copy.deepcopy(self._qfs_unwrapped)
        for p in self._qfs_target.parameters():
            p.requires_grad = False

        # Automatic entropy tuning
        self._target_entropy = torch.tensor(target_entropy, device=device)
        self._log_alpha = torch.nn.Parameter(torch.log(torch.tensor([alpha], device=device)), requires_grad=True)

        # EMA tau
        self._tau = tau
        self._encoder_tau = encoder_tau

    @property
    def num_critics(self) -> int:
        return self._num_critics

    @property
    def qfs(self) -> nn.ModuleList:
        return self._qfs

    @property
    def qfs_unwrapped(self) -> nn.ModuleList:
        return self._qfs_unwrapped

    @property
    def actor(self) -> Union[SACPixelContinuousActor, _FabricModule]:
        return self._actor

    @property
    def qfs_target(self) -> nn.ModuleList:
        return self._qfs_target

    @property
    def alpha(self) -> float:
        return self._log_alpha.exp().item()

    @property
    def target_entropy(self) -> Tensor:
        return self._target_entropy

    @property
    def log_alpha(self) -> Tensor:
        return self._log_alpha

    def get_actions_and_log_probs(self, obs: Tensor, detach_encoder_features: bool = False) -> Tuple[Tensor, Tensor]:
        return self.actor(obs, detach_encoder_features)

    def get_greedy_actions(self, obs: Tensor) -> Tensor:
        return self.actor.get_greedy_actions(obs)

    def get_q_values(self, obs: Tensor, action: Tensor, detach_encoder_features: bool = False) -> Tensor:
        return torch.cat([self.qfs[i](obs, action, detach_encoder_features) for i in range(len(self.qfs))], dim=-1)

    @torch.no_grad()
    def get_target_q_values(self, obs: Tensor, action: Tensor) -> Tensor:
        return torch.cat([self.qfs_target[i](obs, action) for i in range(len(self.qfs))], dim=-1)

    @torch.no_grad()
    def get_next_target_q_values(self, next_obs: Tensor, rewards: Tensor, dones: Tensor, gamma: float):
        # Get q-values for the next observations and actions, estimated by the target q-functions
        next_state_actions, next_state_log_pi = self.get_actions_and_log_probs(next_obs)
        qf_next_target = self.get_target_q_values(next_obs, next_state_actions)
        min_qf_next_target = torch.min(qf_next_target, dim=-1, keepdim=True)[0] - self.alpha * next_state_log_pi
        next_qf_value = rewards + (1 - dones) * gamma * min_qf_next_target
        return next_qf_value

    @torch.no_grad()
    def qfs_target_ema(self) -> None:
        for i in range(self.num_critics):
            for param, target_param in zip(
                self.qfs_unwrapped[i].model.parameters(), self.qfs_target[i].model.parameters()
            ):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    @torch.no_grad()
    def qfs_target_encoder_ema(self) -> None:
        for i in range(self.num_critics):
            for param, target_param in zip(
                self.qfs_unwrapped[i].encoder.parameters(), self.qfs_target[i].encoder.parameters()
            ):
                target_param.data.copy_(self._encoder_tau * param.data + (1 - self._encoder_tau) * target_param.data)
