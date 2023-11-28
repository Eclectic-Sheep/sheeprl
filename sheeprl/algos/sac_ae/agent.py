import copy
from math import prod
from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from numpy.typing import NDArray
from torch import Size, Tensor

from sheeprl.algos.sac_ae.utils import weight_init
from sheeprl.models.models import CNN, MLP, DeCNN, MultiDecoder, MultiEncoder

LOG_STD_MAX = 2
LOG_STD_MIN = -10


class CNNEncoder(CNN):
    """Encoder network from https://arxiv.org/abs/1910.01741

    Args:
        in_channels (int): the input channels to the first convolutional layer
        features_dim (int): the features dimension in output from the last convolutional layer
        screen_size (int, optional): the dimension of the input image as a single integer.
            Needed to extract the features and compute the output dimension after all the
            convolutional layers.
            Defaults to 64.
    """

    def __init__(
        self,
        in_channels: int,
        features_dim: int,
        keys: Sequence[str],
        screen_size: int = 64,
        cnn_channels_multiplier: int = 1,
    ):
        super().__init__(
            in_channels,
            (np.array([32, 32, 32, 32]) * cnn_channels_multiplier).tolist(),
            layer_args=[
                {"kernel_size": 3, "stride": 2},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
            ],
        )
        self.keys = keys
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
        self.input_dim = in_channels

    @property
    def conv_output_shape(self) -> Size:
        return self._conv_output_shape

    def forward(self, obs: Dict[str, Tensor], *, detach_encoder_features: bool = False, **kwargs) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-3)
        x = self.model(x).flatten(1)
        if detach_encoder_features:
            x = x.detach()
        x = self.fc(x)
        return x


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        keys: Sequence[str],
        dense_units: int = 1024,
        mlp_layers: int = 3,
        act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.model = MLP(
            input_dims=input_dim,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=act,
            norm_layer=nn.LayerNorm if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units}] * mlp_layers if layer_norm else None,
        )
        self.output_dim = dense_units
        self.input_dim = input_dim

    def forward(self, obs: Dict[str, Tensor], *args, detach_encoder_features: bool = False, **kwargs) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-1).type(torch.float32)
        x = self.model(x)
        if detach_encoder_features:
            x = x.detach()
        return x


class MLPDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: Sequence[int],
        keys: Sequence[str],
        dense_units: int = 1024,
        mlp_layers: int = 3,
        act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.model = MLP(
            input_dims=input_dim,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=act,
            norm_layer=nn.LayerNorm if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units}] * mlp_layers if layer_norm else None,
        )
        self.heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.output_dims])

    def forward(self, x: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        reconstructed_obs = {}
        x = self.model(x)
        reconstructed_obs.update({k: h(x) for k, h in zip(self.keys, self.heads)})
        return reconstructed_obs


class CNNDecoder(DeCNN):
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
        self,
        encoder_conv_output_shape: Size,
        features_dim: int,
        keys: Sequence[str],
        channels: Sequence[int],
        screen_size: int = 64,
        cnn_channels_multiplier: int = 1,
    ):
        super().__init__(
            32 * cnn_channels_multiplier,
            (np.array([32, 32, 32]) * cnn_channels_multiplier).tolist(),
            layer_args=[
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
            ],
        )
        self.cnn_splits = channels
        out_channels = sum(channels)
        self.keys = keys
        self.fc = MLP(input_dims=features_dim, hidden_sizes=(prod(encoder_conv_output_shape),))
        self.to_obs = nn.ConvTranspose2d(
            super().output_dim, out_channels=out_channels, kernel_size=3, stride=2, output_padding=1
        )
        self._output_dim = Size([out_channels, screen_size, screen_size])
        self._encoder_conv_output_shape = encoder_conv_output_shape

    def forward(self, x: Tensor, *args, **kwargs) -> Dict[str, Tensor]:
        reconstructed_obs = {}
        x = self.fc(x).view(-1, *self._encoder_conv_output_shape)
        x = self.model(x)
        x = self.to_obs(x)
        reconstructed_obs.update({k: rec_obs for k, rec_obs in zip(self.keys, torch.split(x, self.cnn_splits, dim=-3))})
        return reconstructed_obs


class SACAEQFunction(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        output_dim: int = 1,
    ):
        super().__init__()
        self.model = MLP(
            input_dims=input_dim + action_dim,
            output_dim=output_dim,
            hidden_sizes=(hidden_size, hidden_size),
            activation=nn.ReLU,
            flatten_dim=None,
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([obs, action], -1)
        return self.model(x)


class SACAECritic(nn.Module):
    def __init__(self, encoder: Union[MultiEncoder, _FabricModule], qfs: List[SACAEQFunction]) -> None:
        super().__init__()
        self.encoder = encoder
        self.qfs = nn.ModuleList(qfs)

        # Orthogonal init
        self.apply(weight_init)

    def forward(self, obs: Tensor, action: Tensor, detach_encoder_features: bool = False) -> Tensor:
        features = self.encoder(obs, detach_encoder_features=detach_encoder_features)
        return torch.cat([self.qfs[i](features, action) for i in range(len(self.qfs))], dim=-1)


class SACAEContinuousActor(nn.Module):
    def __init__(
        self,
        encoder: Union[MultiEncoder, _FabricModule],
        action_dim: int,
        distribution_cfg: Dict[str, Any],
        hidden_size: int = 1024,
        action_low: Union[SupportsFloat, NDArray] = -1.0,
        action_high: Union[SupportsFloat, NDArray] = 1.0,
    ):
        super().__init__()
        self.distribution_cfg = distribution_cfg

        self.encoder = encoder
        self.model = MLP(input_dims=encoder.output_dim, hidden_sizes=(hidden_size, hidden_size), flatten_dim=None)
        self.fc_mean = nn.Linear(self.model.output_dim, action_dim)
        self.fc_logstd = nn.Linear(self.model.output_dim, action_dim)

        # Action rescaling buffers
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

        # Orthogonal init
        self.apply(weight_init)

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
        features = self.encoder(obs, detach_encoder_features=detach_encoder_features)
        x = self.model(features)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return self.get_actions_and_log_probs(mean, log_std.exp())

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
        normal = torch.distributions.Normal(mean, std, validate_args=self.distribution_cfg.validate_args)

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
        features = self.encoder(obs)
        x = self.model(features)
        mean = self.fc_mean(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean


class SACAEAgent(nn.Module):
    def __init__(
        self,
        actor: Union[SACAEContinuousActor, _FabricModule],
        critic: Union[SACAECritic, _FabricModule],
        target_entropy: float,
        alpha: float = 1.0,
        tau: float = 0.01,
        encoder_tau: float = 0.05,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        # Tie encoder weights between actor and critic
        if actor.encoder.cnn_encoder is not None:
            actor.encoder.cnn_encoder.model = critic.encoder.cnn_encoder.model
        if actor.encoder.mlp_encoder is not None:
            actor.encoder.mlp_encoder.model = critic.encoder.mlp_encoder.model

        # Actor and critics
        self._num_critics = len(critic.qfs)
        self.actor = actor
        self.critic = critic

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
    def critic(self) -> Union[SACAECritic, _FabricModule]:
        return self._critic

    def __setattr__(self, name: str, value: Union[Tensor, nn.Module]) -> None:
        # Taken from https://github.com/pytorch/pytorch/pull/92044
        # Check if a property setter exists. If it does, use it.
        class_attr = getattr(self.__class__, name, None)
        if isinstance(class_attr, property) and class_attr.fset is not None:
            return class_attr.fset(self, value)
        super().__setattr__(name, value)

    @critic.setter
    def critic(self, critic: Union[SACAECritic, _FabricModule]) -> None:
        self._critic = critic

        # Create target critic unwrapping the DDP module from the critics to prevent
        # `RuntimeError: DDP Pickling/Unpickling are only supported when using DDP with the default process group.
        # That is, when you have called init_process_group and have not passed process_group
        # argument to DDP constructor`.
        # This happens when we're using the decoupled version of SACAE for example
        if hasattr(critic, "module"):
            critic_module = critic.module
        else:
            critic_module = critic
        self._critic_unwrapped = critic_module
        self._critic_target = copy.deepcopy(self._critic_unwrapped)
        for p in self._critic_target.parameters():
            p.requires_grad = False
        return

    @property
    def critic_unwrapped(self) -> SACAECritic:
        return self._critic_unwrapped

    @property
    def actor(self) -> Union[SACAEContinuousActor, _FabricModule]:
        return self._actor

    @actor.setter
    def actor(self, actor: Union[SACAEContinuousActor, _FabricModule]) -> None:
        self._actor = actor
        return

    @property
    def critic_target(self) -> SACAECritic:
        return self._critic_target

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
        return self.critic(obs, action, detach_encoder_features)

    @torch.no_grad()
    def get_target_q_values(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.critic_target(obs, action)

    @torch.no_grad()
    def get_next_target_q_values(self, next_obs: Tensor, rewards: Tensor, dones: Tensor, gamma: float):
        # Get q-values for the next observations and actions, estimated by the target q-functions
        next_state_actions, next_state_log_pi = self.get_actions_and_log_probs(next_obs)
        qf_next_target = self.get_target_q_values(next_obs, next_state_actions)
        min_qf_next_target = torch.min(qf_next_target, dim=-1, keepdim=True)[0] - self.alpha * next_state_log_pi
        next_qf_value = rewards + (1 - dones) * gamma * min_qf_next_target
        return next_qf_value

    @torch.no_grad()
    def critic_target_ema(self) -> None:
        for param, target_param in zip(self.critic_unwrapped.qfs.parameters(), self.critic_target.qfs.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    @torch.no_grad()
    def critic_encoder_target_ema(self) -> None:
        for param, target_param in zip(
            self.critic_unwrapped.encoder.parameters(), self.critic_target.encoder.parameters()
        ):
            target_param.data.copy_(self._encoder_tau * param.data + (1 - self._encoder_tau) * target_param.data)


def build_agent(
    fabric: Fabric,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    action_space: gymnasium.spaces.Box,
    agent_state: Optional[Dict[str, Tensor]] = None,
    encoder_state: Optional[Dict[str, Tensor]] = None,
    decoder_sate: Optional[Dict[str, Tensor]] = None,
) -> Tuple[SACAEAgent, _FabricModule, _FabricModule]:
    act_dim = prod(action_space.shape)
    target_entropy = -act_dim

    # Define the encoder and decoder and setup them with fabric.
    # Then we will set the critic encoder and actor decoder as the unwrapped encoder module:
    # we do not need it wrapped with the strategy inside actor and critic
    cnn_channels = [prod(obs_space[k].shape[:-2]) for k in cfg.algo.cnn_keys.encoder]
    mlp_dims = [obs_space[k].shape[0] for k in cfg.algo.mlp_keys.encoder]
    cnn_encoder = (
        CNNEncoder(
            in_channels=sum(cnn_channels),
            features_dim=cfg.algo.encoder.features_dim,
            keys=cfg.algo.cnn_keys.encoder,
            screen_size=cfg.env.screen_size,
            cnn_channels_multiplier=cfg.algo.encoder.cnn_channels_multiplier,
        )
        if cfg.algo.cnn_keys.encoder is not None and len(cfg.algo.cnn_keys.encoder) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(
            sum(mlp_dims),
            cfg.algo.mlp_keys.encoder,
            cfg.algo.encoder.dense_units,
            cfg.algo.encoder.mlp_layers,
            eval(cfg.algo.encoder.dense_act),
            cfg.algo.encoder.layer_norm,
        )
        if cfg.algo.mlp_keys.encoder is not None and len(cfg.algo.mlp_keys.encoder) > 0
        else None
    )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder)
    cnn_decoder = (
        CNNDecoder(
            cnn_encoder.conv_output_shape,
            features_dim=encoder.output_dim,
            keys=cfg.algo.cnn_keys.decoder,
            channels=cnn_channels,
            screen_size=cfg.env.screen_size,
            cnn_channels_multiplier=cfg.algo.decoder.cnn_channels_multiplier,
        )
        if cfg.algo.cnn_keys.decoder is not None and len(cfg.algo.cnn_keys.decoder) > 0
        else None
    )
    mlp_decoder = (
        MLPDecoder(
            encoder.output_dim,
            mlp_dims,
            cfg.algo.mlp_keys.decoder,
            cfg.algo.decoder.dense_units,
            cfg.algo.decoder.mlp_layers,
            eval(cfg.algo.decoder.dense_act),
            cfg.algo.decoder.layer_norm,
        )
        if cfg.algo.mlp_keys.decoder is not None and len(cfg.algo.mlp_keys.decoder) > 0
        else None
    )
    decoder = MultiDecoder(cnn_decoder, mlp_decoder)
    if encoder_state:
        encoder.load_state_dict(encoder_state)
    if decoder_sate:
        decoder.load_state_dict(decoder_sate)

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
        device=fabric.device,
    )

    if agent_state:
        agent.load_state_dict(agent_state)

    encoder = fabric.setup_module(encoder)
    decoder = fabric.setup_module(decoder)
    agent.actor = fabric.setup_module(agent.actor)
    agent.critic = fabric.setup_module(agent.critic)

    return agent, encoder, decoder
