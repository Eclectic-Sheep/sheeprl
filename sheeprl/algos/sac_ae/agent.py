from __future__ import annotations

import copy
from math import prod
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn as nn
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from numpy.typing import NDArray
from torch import Size, Tensor

from sheeprl.algos.sac_ae.utils import weight_init
from sheeprl.core.collector import Act, Policy
from sheeprl.models.io import MultiDecoder, MultiEncoder, build_encoder, to_device
from sheeprl.models.models import CNN, MLP, DeCNN
from sheeprl.utils.compile import compiled_policy
from sheeprl.utils.fabric import get_single_device_fabric, setup_module
from sheeprl.utils.model import cnn_forward, ema_
from sheeprl.utils.utils import dotdict

LOG_STD_MAX = 2
LOG_STD_MIN = -10


def _activation(activation: Union[str, type]) -> type:
    return hydra.utils.get_class(activation) if isinstance(activation, str) else activation


def conv_output_shape(input_shape: Sequence[int], channels_multiplier: int = 1) -> Tuple[int, int, int]:
    """The shape of the feature maps of the convolutions of `CNNEncoder` on images of shape `input_shape`: a 3x3
    convolution with stride 2, then three with stride 1, of 32 * `channels_multiplier` channels each."""
    _, height, width = input_shape
    return (32 * channels_multiplier, (height - 3) // 2 + 1 - 6, (width - 3) // 2 + 1 - 6)


class CNNEncoder(CNN):
    """The encoder of the images of SAC-AE (https://arxiv.org/abs/1910.01741), on images of shape `input_shape`: four
    3x3 convolutions (the first with stride 2) of 32 * `channels_multiplier` channels, then a linear layer to
    `features_dim` features, normalized and squashed by a tanh. With `detach_encoder_features`, the gradient doesn't
    reach the convolutions (the actor's loss)."""

    def __init__(self, input_shape: Sequence[int], features_dim: int = 50, channels_multiplier: int = 1):
        input_shape = tuple(int(d) for d in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"The CNN encoder of SAC-AE takes images (channels, height, width), got {input_shape}")
        super().__init__(
            input_shape[0],
            (np.array([32, 32, 32, 32]) * channels_multiplier).tolist(),
            layer_args=[
                {"kernel_size": 3, "stride": 2},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
            ],
        )
        with torch.no_grad():
            x: Tensor = self.model(torch.rand(1, *input_shape, device=self.model[0].weight.device))
            flattened_conv_output_dim = x.flatten(1).shape[1]
        self.fc = MLP(
            input_dims=flattened_conv_output_dim,
            hidden_sizes=(features_dim,),
            activation=nn.Tanh,
            norm_layer=nn.LayerNorm,
            norm_args={"normalized_shape": features_dim},
        )
        self._output_dim = features_dim

    def tie(self, other: CNNEncoder) -> None:
        """Use the convolutions of `other` (the actor uses the ones of the critic), keeping its own linear layer."""
        self._model = other._model

    def forward(self, x: Tensor, detach_encoder_features: bool = False) -> Tensor:
        x = cnn_forward(self.model, x, x.shape[-3:], (-1,))
        if detach_encoder_features:
            x = x.detach()
        return self.fc(x)


class MLPEncoder(nn.Module):
    """The encoder of the vectors of SAC-AE: `mlp_layers` layers of `dense_units` units on a vector of shape
    `input_shape`. With `detach_encoder_features`, its features carry no gradient (the actor's loss)."""

    def __init__(
        self,
        input_shape: Sequence[int],
        dense_units: int = 1024,
        mlp_layers: int = 3,
        dense_act: Union[str, type] = "torch.nn.ReLU",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        input_shape = tuple(int(d) for d in input_shape)
        if len(input_shape) != 1:
            raise ValueError(f"The MLP encoder of SAC-AE takes vectors, got the shape {input_shape}")
        self.model = MLP(
            input_dims=input_shape[0],
            hidden_sizes=[dense_units] * mlp_layers,
            activation=_activation(dense_act),
            norm_layer=nn.LayerNorm if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units}] * mlp_layers if layer_norm else None,
        )
        self.output_dim = dense_units

    def tie(self, other: MLPEncoder) -> None:
        """Use the layers of `other` (the actor uses the ones of the critic)."""
        self.model = other.model

    def forward(self, x: Tensor, detach_encoder_features: bool = False) -> Tensor:
        x = self.model(x)
        return x.detach() if detach_encoder_features else x


class MLPDecoder(nn.Module):
    """The decoder of the vectors of SAC-AE: `mlp_layers` layers of `dense_units` units from the latent state, then a
    linear layer to the vectors of shape `output_shape`."""

    def __init__(
        self,
        latent_size: int,
        output_shape: Sequence[int],
        dense_units: int = 1024,
        mlp_layers: int = 3,
        dense_act: Union[str, type] = "torch.nn.ReLU",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.model = MLP(
            input_dims=latent_size,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=_activation(dense_act),
            norm_layer=nn.LayerNorm if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units}] * mlp_layers if layer_norm else None,
        )
        self.head = nn.Linear(dense_units, int(output_shape[0]))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.model(x))


class CNNDecoder(DeCNN):
    """The decoder of the images of SAC-AE (https://arxiv.org/abs/1910.01741): a linear layer from the latent state to
    the feature maps of the encoder (`conv_output_shape`), three 3x3 convolutions of 32 * `channels_multiplier`
    channels, then a transposed one with stride 2 to the images of shape `output_shape` (whose sides must be even)."""

    def __init__(self, latent_size: int, output_shape: Sequence[int], channels_multiplier: int = 1):
        super().__init__(
            32 * channels_multiplier,
            (np.array([32, 32, 32]) * channels_multiplier).tolist(),
            layer_args=[
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
                {"kernel_size": 3, "stride": 1},
            ],
        )
        output_shape = tuple(int(d) for d in output_shape)
        encoder_conv_output_shape = conv_output_shape(output_shape, channels_multiplier)
        self.fc = MLP(input_dims=latent_size, hidden_sizes=(prod(encoder_conv_output_shape),))
        self.to_obs = nn.ConvTranspose2d(
            super().output_dim, out_channels=output_shape[0], kernel_size=3, stride=2, output_padding=1
        )
        self._output_dim = Size(output_shape)
        self._encoder_conv_output_shape = encoder_conv_output_shape
        with torch.no_grad():
            reconstructed_size = tuple(self.to_obs(self.model(torch.zeros(1, *encoder_conv_output_shape))).shape[-2:])
        if reconstructed_size != output_shape[1:]:
            raise ValueError(
                f"The decoder reconstructs images of {reconstructed_size[0]}x{reconstructed_size[1]} pixels from the "
                f"ones of {output_shape[1]}x{output_shape[2]}: their sides must be even"
            )

    def forward(self, x: Tensor) -> Tensor:
        leading = x.shape[:-1]
        x = self.fc(x).view(-1, *self._encoder_conv_output_shape)
        x = self.to_obs(self.model(x))
        return x.view(*leading, *x.shape[-3:])


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
        distribution_cfg: dotdict,
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

    def get_actions_and_log_probs(self, mean: Tensor, std: Tensor) -> Tuple[Tensor, Tensor]:
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
        # Tie encoder weights between actor and critic: the network of every input of the actor uses the layers of the
        # critic's (`tie`: the convolutions of the images, with the actor's own linear layer; the layers of the vectors)
        for name, actor_input in actor.encoder.inputs.items():
            net, critic_net = actor_input.net, critic.encoder.inputs[name].net
            if not hasattr(net, "tie"):
                raise TypeError(
                    f"The network of the input `{name}` of SAC-AE must share its layers with the one of the critic "
                    f"(`tie`), as `sheeprl.algos.sac_ae.agent.CNNEncoder` and `MLPEncoder` do: {type(net).__name__} "
                    "doesn't"
                )
            net.tie(critic_net)

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

        # The target critics are copies of the critics without their wrappers (Fabric, DDP)
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

    @critic_target.setter
    def critic_target(self, critic_target: SACAECritic | _FabricModule) -> None:
        self._critic_target = critic_target
        return

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
        # The temperature as a tensor: no synchronization with the device, and no break of a compiled graph
        alpha = self._log_alpha.exp()
        min_qf_next_target = torch.min(qf_next_target, dim=-1, keepdim=True)[0] - alpha * next_state_log_pi
        next_qf_value = rewards + (1 - dones) * gamma * min_qf_next_target
        return next_qf_value

    @torch.no_grad()
    def critic_target_ema(self) -> None:
        ema_(self.critic_target.qfs, self.critic_unwrapped.qfs, self._tau)

    @torch.no_grad()
    def critic_encoder_target_ema(self) -> None:
        ema_(self.critic_target.encoder, self.critic_unwrapped.encoder, self._encoder_tau)


class SACAEPolicy(nn.Module, Policy):
    """The policy of SAC-AE: `forward` samples the actions from the observations as tensors, `act` plays them in the
    environments (`Policy`), from the observations of the keys of the inputs (`keys`) moved to `device`."""

    def __init__(
        self,
        feature_extractor: MultiEncoder,
        fc: nn.Module,
        fc_mean: nn.Module,
        fc_logstd: nn.Module,
        action_low: Union[SupportsFloat, NDArray] = -1.0,
        action_high: Union[SupportsFloat, NDArray] = 1.0,
        device: str | torch.device = "cpu",
        keys: Sequence[str] = (),
    ):
        super().__init__()
        self.encoder = feature_extractor
        self.model = fc
        self.fc_mean = fc_mean
        self.fc_logstd = fc_logstd
        self.device = torch.device(device)
        # The keys of the inputs of the encoder, by default
        self.keys = tuple(keys) or tuple(feature_extractor.keys)

        # Action rescaling buffers
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def act(self, obs: Dict[str, np.ndarray], greedy: bool = False) -> Act:
        torch_obs = to_device({k: obs[k] for k in self.keys}, self.device)
        actions = self.get_actions(torch_obs, greedy=True) if greedy else self(torch_obs)
        return Act(actions.cpu().numpy())

    def forward(self, obs: Tensor, greedy: bool = False) -> Tensor:
        """Given an observation, it returns a tanh-squashed
        sampled action (correctly rescaled to the environment action bounds) and its
        log-prob (as defined in Eq. 26 of https://arxiv.org/abs/1812.05905)

        Args:
            obs (Tensor): the observation tensor

        Returns:
            tanh-squashed action, rescaled to the environment action bounds
            action log-prob
        """
        features = self.encoder(obs, phase="act", detach_encoder_features=False)
        x = self.model(features)
        mean = self.fc_mean(x)
        if greedy:
            return torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            log_std = self.fc_logstd(x)
            # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            log_std = torch.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            normal = torch.distributions.Normal(mean, log_std.exp())
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            actions = y_t * self.action_scale + self.action_bias
            return actions

    def get_actions(self, obs: Tensor, greedy: bool = False) -> Tensor:
        return self(obs, greedy)


def build_agent(
    fabric: Fabric,
    cfg: dotdict,
    obs_space: gymnasium.spaces.Dict,
    action_space: gymnasium.spaces.Box,
    agent_state: Optional[Dict[str, Tensor]] = None,
    encoder_state: Optional[Dict[str, Tensor]] = None,
    decoder_sate: Optional[Dict[str, Tensor]] = None,
) -> Tuple[SACAEAgent, _FabricModule, _FabricModule, SACAEPolicy]:
    """The agent of SAC-AE on the inputs of `algo.inputs`, from the spaces of the store (`obs_space`), with the decoder
    of `algo.outputs`, and its policy, which shares the weights of the actor."""
    act_dim = prod(action_space.shape)
    target_entropy = -act_dim

    # Define the encoder and decoder and setup them with fabric.
    # Then we will set the critic encoder and actor decoder as the unwrapped encoder module:
    # we do not need it wrapped with the strategy inside actor and critic
    encoder = build_encoder(cfg, obs_space)
    decoder = MultiDecoder.from_config(cfg.algo.outputs, encoder, latent_size=encoder.output_dim)
    # The initialization of the official implementation, as the one of the actor and of the critic
    decoder.apply(weight_init)
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

    # Setup policy agent
    policy_fabric = get_single_device_fabric(fabric)
    policy = SACAEPolicy(
        copy.deepcopy(agent.actor.encoder),
        copy.deepcopy(agent.actor.model),
        copy.deepcopy(agent.actor.fc_mean),
        copy.deepcopy(agent.actor.fc_logstd),
        action_low=action_space.low,
        action_high=action_space.high,
        device=policy_fabric.device,
        keys=encoder.keys,
    )

    # The encoder layers of the actor are tied with the ones of the critic (see `SACAEAgent`): they are trained only
    # through the critic and decoder losses (`detach_encoder_features` in the actor loss)
    encoder = setup_module(fabric, encoder)
    decoder = setup_module(fabric, decoder)
    # Setting the critic makes the target critic a copy of it: the one of the checkpoint is kept
    critic_target = agent.critic_target
    agent.actor = setup_module(fabric, agent.actor)
    agent.critic = setup_module(fabric, agent.critic)
    if agent_state:
        agent.critic_target = critic_target

    # Wrap the target critic with a single-device fabric. This lets the target critic
    # to be on the same device as the agent and to run with the same precision
    agent.critic_target = policy_fabric.setup_module(agent.critic_target)

    # Setup policy agent
    policy.encoder = policy_fabric.setup_module(policy.encoder)
    policy.model = policy_fabric.setup_module(policy.model)
    policy.fc_mean = policy_fabric.setup_module(policy.fc_mean)
    policy.fc_logstd = policy_fabric.setup_module(policy.fc_logstd)
    policy.action_scale = policy.action_scale.to(policy_fabric.device)
    policy.action_bias = policy.action_bias.to(policy_fabric.device)

    # Tie weights between the agent and the policy
    for agent_p, policy_p in zip(agent.actor.parameters(), policy.parameters()):
        policy_p.data = agent_p.data
    # The step of the policy, compiled with `algo.compile` (`compiled_policy`)
    policy.forward = compiled_policy(policy.forward, fabric, cfg)
    return agent, encoder, decoder, policy
