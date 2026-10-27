from __future__ import annotations

import copy
from math import log, sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, OneHotCategorical

from sheeprl.core.actions import ActionCodec
from sheeprl.core.collector import Act, Policy
from sheeprl.models.io import MultiEncoder, build_encoder, to_device
from sheeprl.models.models import MLP
from sheeprl.utils.compile import compiled_policy
from sheeprl.utils.fabric import get_single_device_fabric, setup_module
from sheeprl.utils.model import per_layer_ortho_init_weights
from sheeprl.utils.utils import dotdict, safetanh


def tanh_log_abs_det_jacobian(x: Tensor) -> Tensor:
    """log |d tanh(x) / dx| = log(1 - tanh(x)²), summed over the last dimension, from the values `x` before the tanh;
    the form of `torch.distributions.TanhTransform`, finite where the tanh saturates."""
    return (2.0 * (log(2.0) - x - F.softplus(-2.0 * x))).sum(-1)


def ortho_init_linear_layers(module: nn.Module, gain: float, output_gain: Optional[float] = None) -> None:
    """Orthogonal weights with `gain` and zero biases for every linear layer of `module`; with `output_gain`, the last
    linear layer (the output of an MLP) gets that gain instead."""
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        is_output = output_gain is not None and i == len(linear_layers) - 1
        per_layer_ortho_init_weights(layer, gain=output_gain if is_output else gain)


class PPOActor(nn.Module):
    def __init__(
        self,
        actor_backbone: torch.nn.Module,
        actor_heads: torch.nn.ModuleList,
        is_continuous: bool,
        distribution: str = "auto",
    ) -> None:
        super().__init__()
        self.actor_backbone = actor_backbone
        self.actor_heads = actor_heads
        self.is_continuous = is_continuous
        self.distribution = distribution

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.actor_backbone(x)
        return [head(x) for head in self.actor_heads]


class PPOAgent(nn.Module):
    """The feature extractor (the inputs of `algo.inputs`), the actor and the critic of PPO and A2C. The actor plays
    all-categorical or continuous actions (`codec`); with `ortho_init`, the linear layers of the feature extractor get
    orthogonal weights (gain 1) and zero biases."""

    def __init__(
        self,
        codec: ActionCodec,
        feature_extractor: MultiEncoder,
        actor_cfg: dotdict,
        critic_cfg: dotdict,
        distribution_cfg: dotdict,
        ortho_init: bool = False,
    ):
        super().__init__()
        codec.check("The actor of PPO")
        self.codec = codec
        is_continuous = codec.continuous
        actions_dim = codec.dims
        self.is_continuous = is_continuous
        self.distribution_cfg = distribution_cfg
        self.distribution = distribution_cfg.get("type", "auto").lower()
        if self.distribution not in ("auto", "normal", "tanh_normal", "discrete"):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, `normal` and `tanh_normal`. "
                f"Found: {self.distribution}"
            )
        if self.distribution == "discrete" and is_continuous:
            raise ValueError("You have choose a discrete distribution but `is_continuous` is true")
        elif self.distribution not in {"discrete", "auto"} and not is_continuous:
            raise ValueError("You have choose a continuous distribution but `is_continuous` is false")
        if self.distribution == "auto":
            if is_continuous:
                self.distribution = "normal"
            else:
                self.distribution = "discrete"
        self.actions_dim = actions_dim
        self.feature_extractor = feature_extractor
        if ortho_init:
            for layer in self.feature_extractor.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, 1.0)
                    layer.bias.data.zero_()
        features_dim = self.feature_extractor.output_dim
        self.critic = MLP(
            input_dims=features_dim,
            output_dim=1,
            hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
            activation=hydra.utils.get_class(critic_cfg.dense_act),
            norm_layer=[nn.LayerNorm for _ in range(critic_cfg.mlp_layers)] if critic_cfg.layer_norm else None,
            norm_args=(
                [{"normalized_shape": critic_cfg.dense_units} for _ in range(critic_cfg.mlp_layers)]
                if critic_cfg.layer_norm
                else None
            ),
        )
        actor_backbone = (
            MLP(
                input_dims=features_dim,
                output_dim=None,
                hidden_sizes=[actor_cfg.dense_units] * actor_cfg.mlp_layers,
                activation=hydra.utils.get_class(actor_cfg.dense_act),
                flatten_dim=None,
                norm_layer=[nn.LayerNorm] * actor_cfg.mlp_layers if actor_cfg.layer_norm else None,
                norm_args=(
                    [{"normalized_shape": actor_cfg.dense_units} for _ in range(actor_cfg.mlp_layers)]
                    if actor_cfg.layer_norm
                    else None
                ),
            )
            if actor_cfg.mlp_layers > 0
            else nn.Identity()
        )
        if is_continuous:
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, sum(actions_dim) * 2)])
        else:
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, action_dim) for action_dim in actions_dim])
        self.actor = PPOActor(actor_backbone, actor_heads, is_continuous, self.distribution)
        # Orthogonal initialization as in the PPO implementation details (https://iclr-blog-track.github.io/2022/03/25/
        # ppo-implementation-details/): hidden layers with gain sqrt(2), the actor's heads with 0.01 (an almost uniform
        # initial policy) and the critic's output with 1
        if actor_cfg.ortho_init:
            ortho_init_linear_layers(actor_backbone, gain=sqrt(2))
            ortho_init_linear_layers(actor_heads, gain=0.01)
        if critic_cfg.ortho_init:
            ortho_init_linear_layers(self.critic, gain=sqrt(2), output_gain=1.0)

    def _normal(self, actor_out: Tensor, actions: Optional[List[Tensor]] = None) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = torch.chunk(actor_out, chunks=2, dim=-1)
        std = log_std.exp()
        normal = Independent(Normal(mean, std), 1)
        actions = actions[0]
        log_prob = normal.log_prob(actions)
        return actions, log_prob.unsqueeze(dim=-1), normal.entropy().unsqueeze(dim=-1)

    def _tanh_normal(self, actor_out: Tensor, actions: Optional[List[Tensor]] = None) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = torch.chunk(actor_out, chunks=2, dim=-1)
        std = log_std.exp()
        normal = Independent(Normal(mean, std), 1)
        # The actions played are stored before the tanh (`PPOPolicy.env_actions`): their log-probability is computed
        # from them, as the policy did, also where the tanh saturates
        actions = actions[0].float()
        log_prob = normal.log_prob(actions) - tanh_log_abs_det_jacobian(actions)
        return actions, log_prob.unsqueeze(dim=-1), normal.entropy().unsqueeze(dim=-1)

    def forward(
        self, obs: Dict[str, Tensor], actions: Optional[List[Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Tensor, Tensor, Tensor]:
        feat = self.feature_extractor(obs)
        actor_out: List[Tensor] = self.actor(feat)
        values = self.critic(feat)
        if self.is_continuous:
            if self.distribution == "normal":
                actions, log_prob, entropy = self._normal(actor_out[0], actions)
            elif self.distribution == "tanh_normal":
                actions, log_prob, entropy = self._tanh_normal(actor_out[0], actions)
            return tuple([actions]), log_prob, entropy, values
        else:
            should_append = False
            actions_logprobs: List[Tensor] = []
            actions_entropies: List[Tensor] = []
            actions_dist: List[Distribution] = []
            if actions is None:
                should_append = True
                actions: List[Tensor] = []
            for i, logits in enumerate(actor_out):
                actions_dist.append(OneHotCategorical(logits=logits))
                actions_entropies.append(actions_dist[-1].entropy())
                if should_append:
                    actions.append(actions_dist[-1].sample())
                actions_logprobs.append(actions_dist[-1].log_prob(actions[i]))
            return (
                tuple(actions),
                torch.stack(actions_logprobs, dim=-1).sum(dim=-1, keepdim=True),
                torch.stack(actions_entropies, dim=-1).sum(dim=-1, keepdim=True),
                values,
            )


class PPOPolicy(nn.Module, Policy):
    """The policy of PPO and A2C: `forward` samples the actions from the observations as tensors, `act` plays them in
    the environments (`Policy`), from the observations of the keys of the inputs (`keys`) moved to `device`. The
    actions go to the environments and to the store through `codec`."""

    def __init__(
        self,
        feature_extractor: MultiEncoder,
        actor: PPOActor,
        critic: nn.Module,
        codec: ActionCodec,
        device: str | torch.device = "cpu",
        keys: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.critic = critic
        self.actor = actor
        self.codec = codec
        self.device = torch.device(device)
        # The keys of the inputs of the feature extractor, by default
        self.keys = tuple(keys) or tuple(feature_extractor.keys)

    def _normal(self, actor_out: Tensor) -> Tuple[Tensor, Tensor]:
        mean, log_std = torch.chunk(actor_out, chunks=2, dim=-1)
        std = log_std.exp()
        normal = Independent(Normal(mean, std), 1)
        actions = normal.sample()
        log_prob = normal.log_prob(actions)
        return actions, log_prob.unsqueeze(dim=-1)

    def _tanh_normal(self, actor_out: Tensor) -> Tuple[Tensor, Tensor]:
        mean, log_std = torch.chunk(actor_out, chunks=2, dim=-1)
        std = log_std.exp()
        normal = Independent(Normal(mean, std), 1)
        # Returned before the tanh, which `env_actions` applies: the agent computes the same log-probability from them
        actions = normal.sample().float()
        log_prob = normal.log_prob(actions) - tanh_log_abs_det_jacobian(actions)
        return actions, log_prob.unsqueeze(dim=-1)

    def forward(self, obs: Dict[str, Tensor]) -> Tuple[Sequence[Tensor], Tensor, Tensor]:
        feat = self.feature_extractor(obs, phase="act")
        values = self.critic(feat)
        actor_out: List[Tensor] = self.actor(feat)
        if self.actor.is_continuous:
            if self.actor.distribution == "normal":
                actions, log_prob = self._normal(actor_out[0])
            elif self.actor.distribution == "tanh_normal":
                actions, log_prob = self._tanh_normal(actor_out[0])
            return tuple([actions]), log_prob, values
        else:
            actions_dist: List[Distribution] = []
            actions_logprobs: List[Tensor] = []
            actions: List[Tensor] = []
            for i, logits in enumerate(actor_out):
                actions_dist.append(OneHotCategorical(logits=logits))
                actions.append(actions_dist[-1].sample())
                actions_logprobs.append(actions_dist[-1].log_prob(actions[i]))
            return (
                tuple(actions),
                torch.stack(actions_logprobs, dim=-1).sum(dim=-1, keepdim=True),
                values,
            )

    def get_values(self, obs: Dict[str, Tensor]) -> Tensor:
        """The values of the observations `obs` (the ones the policy plays from, e.g. the last ones of a rollout)."""
        feat = self.feature_extractor(obs, phase="act")
        return self.critic(feat)

    def obs_to_device(self, obs: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        """The observations of the keys of the inputs, as tensors on the device of the policy."""
        return to_device({k: obs[k] for k in self.keys}, self.device)

    def env_actions(self, actions: Sequence[Tensor]) -> np.ndarray:
        """The actions to play in the environments, from the ones returned by `forward`: the indices of the discrete
        actions; the continuous actions, squashed by the tanh with `tanh_normal` (they are stored before it)."""
        if self.actor.is_continuous and self.actor.distribution == "tanh_normal":
            squashed = torch.cat(list(actions), dim=-1)
            return self.codec.to_env([safetanh(squashed, eps=torch.finfo(squashed.dtype).resolution)])
        return self.codec.to_env(actions)

    def act(self, obs: Dict[str, np.ndarray], greedy: bool = False) -> Act:
        """The actions for `obs`, with their columns: the actions as stored (one-hot for discrete actions, while the
        environments take their indices), their log-probabilities and the values of `obs`."""
        obs = self.obs_to_device(obs)
        if greedy:
            # The modes of the distributions, the continuous ones already squashed with `tanh_normal`
            return Act(self.codec.to_env(self.get_actions(obs, greedy=True)))
        actions, logprobs, values = self(obs)
        columns = {
            "actions": self.codec.to_store(actions),
            "logprobs": logprobs.cpu().numpy(),
            "values": values.cpu().numpy(),
        }
        return Act(self.env_actions(actions), columns)

    def get_actions(self, obs: Dict[str, Tensor], greedy: bool = False) -> Sequence[Tensor]:
        feat = self.feature_extractor(obs, phase="act")
        actor_out: List[Tensor] = self.actor(feat)
        if self.actor.is_continuous:
            mean, log_std = torch.chunk(actor_out[0], chunks=2, dim=-1)
            if greedy:
                actions = mean
            else:
                std = log_std.exp()
                normal = Independent(Normal(mean, std), 1)
                actions = normal.sample()
            if self.actor.distribution == "tanh_normal":
                actions = safetanh(actions, eps=torch.finfo(actions.dtype).resolution)
            return tuple([actions])
        else:
            actions: List[Tensor] = []
            actions_dist: List[Distribution] = []
            for logits in actor_out:
                actions_dist.append(OneHotCategorical(logits=logits))
                if greedy:
                    actions.append(actions_dist[-1].mode)
                else:
                    actions.append(actions_dist[-1].sample())
            return tuple(actions)


def build_agent(
    fabric: Fabric,
    codec: ActionCodec,
    cfg: dotdict,
    obs_space: gymnasium.spaces.Dict,
    agent_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[PPOAgent, PPOPolicy]:
    """The agent of PPO and A2C on the inputs of `algo.inputs`, from the spaces of the store (`obs_space`), and its
    policy, which shares its weights."""
    agent = PPOAgent(
        codec=codec,
        feature_extractor=build_encoder(cfg, obs_space),
        actor_cfg=cfg.algo.actor,
        critic_cfg=cfg.algo.critic,
        distribution_cfg=cfg.distribution,
        ortho_init=cfg.algo.encoder.get("ortho_init", False),
    )
    if agent_state:
        agent.load_state_dict(agent_state)

    # Setup policy agent
    policy_fabric = get_single_device_fabric(fabric)
    policy = PPOPolicy(
        copy.deepcopy(agent.feature_extractor),
        copy.deepcopy(agent.actor),
        copy.deepcopy(agent.critic),
        codec,
        device=policy_fabric.device,
        keys=agent.feature_extractor.keys,
    )

    # Setup training agent
    agent.feature_extractor = setup_module(fabric, agent.feature_extractor)
    agent.critic = setup_module(fabric, agent.critic)
    agent.actor = setup_module(fabric, agent.actor)

    # Setup policy agent
    policy.feature_extractor = policy_fabric.setup_module(policy.feature_extractor)
    policy.critic = policy_fabric.setup_module(policy.critic)
    policy.actor = policy_fabric.setup_module(policy.actor)

    # Tie weights between the agent and the policy
    for agent_p, policy_p in zip(agent.feature_extractor.parameters(), policy.feature_extractor.parameters()):
        policy_p.data = agent_p.data
    for agent_p, policy_p in zip(agent.actor.parameters(), policy.actor.parameters()):
        policy_p.data = agent_p.data
    for agent_p, policy_p in zip(agent.critic.parameters(), policy.critic.parameters()):
        policy_p.data = agent_p.data
    # The step of the policy, compiled with `algo.compile` (`compiled_policy`)
    policy.forward = compiled_policy(policy.forward, fabric, cfg)
    return agent, policy
