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
from torch.distributions import Independent, Normal, OneHotCategorical

from sheeprl.core.actions import ActionCodec
from sheeprl.core.collector import Act, Policy
from sheeprl.models.io import Encoder, build_encoder, to_device
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


def normal_head(out: Tensor) -> Independent:
    """The normal distribution of a continuous head from the output of its layer: the means, then the log-stds."""
    mean, log_std = torch.chunk(out, chunks=2, dim=-1)
    std = log_std.exp()
    return Independent(Normal(mean, std), 1)


def head_distribution(codec: ActionCodec, distribution: str) -> str:
    """The distribution of the continuous heads of `codec` (`distribution`, `normal` with `auto`), or `discrete` when
    all the heads are categorical: the ones of PPO, A2C and recurrent PPO."""
    has_continuous = not codec.categorical
    if distribution not in ("auto", "normal", "tanh_normal", "discrete"):
        raise ValueError(
            f"The distribution must be on of: `auto`, `discrete`, `normal` and `tanh_normal`. Found: {distribution}"
        )
    if distribution == "discrete" and has_continuous:
        raise ValueError("The `discrete` distribution plays categorical actions, but some actions are continuous")
    if distribution not in ("discrete", "auto") and not has_continuous:
        raise ValueError(f"The `{distribution}` distribution plays continuous actions, but all actions are discrete")
    if distribution == "auto":
        return "normal" if has_continuous else "discrete"
    return distribution


class PPOAgent(nn.Module):
    """The feature extractor (the inputs of `algo.inputs`), the actor and the critic of PPO and A2C. The actor has one
    head per action of `codec`: a categorical distribution for a discrete one, a normal or a tanh-normal one
    (`distribution`) for a continuous one, so the actions may mix them; the log-probabilities and the entropies are
    summed over the heads. With `ortho_init`, the linear layers of the feature extractor get orthogonal weights (gain
    1) and zero biases."""

    def __init__(
        self,
        codec: ActionCodec,
        feature_extractor: Encoder,
        actor_cfg: dotdict,
        critic_cfg: dotdict,
        distribution_cfg: dotdict,
        ortho_init: bool = False,
    ):
        super().__init__()
        self.codec = codec
        is_continuous = codec.continuous
        actions_dim = codec.dims
        self.is_continuous = is_continuous
        self.distribution_cfg = distribution_cfg
        self.distribution = head_distribution(codec, distribution_cfg.get("type", "auto").lower())
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
        # One layer per action: the means and the log-stds of a continuous one, the logits of a categorical one
        actor_heads = nn.ModuleList(
            [
                nn.Linear(actor_cfg.dense_units, head.size * 2 if head.kind == "continuous" else head.size)
                for head in codec.heads
            ]
        )
        self.actor = PPOActor(actor_backbone, actor_heads, is_continuous, self.distribution)
        # Orthogonal initialization as in the PPO implementation details (https://iclr-blog-track.github.io/2022/03/25/
        # ppo-implementation-details/): hidden layers with gain sqrt(2), the actor's heads with 0.01 (an almost uniform
        # initial policy) and the critic's output with 1
        if actor_cfg.ortho_init:
            ortho_init_linear_layers(actor_backbone, gain=sqrt(2))
            ortho_init_linear_layers(actor_heads, gain=0.01)
        if critic_cfg.ortho_init:
            ortho_init_linear_layers(self.critic, gain=sqrt(2), output_gain=1.0)

    def forward(
        self, obs: Dict[str, Tensor], actions: Optional[Sequence[Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Tensor, Tensor, Tensor]:
        """The actions (`actions`, or sampled without them), their log-probabilities and entropies summed over the
        heads, and the values of `obs`."""
        feat = self.feature_extractor(obs)
        actor_out: List[Tensor] = self.actor(feat)
        values = self.critic(feat)
        played: List[Tensor] = []
        logprobs: List[Tensor] = []
        entropies: List[Tensor] = []
        for i, out in enumerate(actor_out):
            if self.codec.is_continuous(i):
                normal = normal_head(out)
                action = normal.sample() if actions is None else actions[i]
                if self.distribution == "tanh_normal":
                    # The actions played are stored before the tanh (`PPOPolicy.env_actions`): their log-probability
                    # is computed from them, as the policy did, also where the tanh saturates
                    action = action.float()
                    logprobs.append(normal.log_prob(action) - tanh_log_abs_det_jacobian(action))
                else:
                    logprobs.append(normal.log_prob(action))
                entropies.append(normal.entropy())
            else:
                categorical = OneHotCategorical(logits=out)
                entropies.append(categorical.entropy())
                action = categorical.sample() if actions is None else actions[i]
                logprobs.append(categorical.log_prob(action))
            played.append(action)
        return (
            tuple(played),
            torch.stack(logprobs, dim=-1).sum(dim=-1, keepdim=True),
            torch.stack(entropies, dim=-1).sum(dim=-1, keepdim=True),
            values,
        )


class PPOPolicy(nn.Module, Policy):
    """The policy of PPO and A2C: `forward` samples the actions from the observations as tensors, `act` plays them in
    the environments (`Policy`), from the observations of the keys of the inputs (`keys`) moved to `device`. The
    actions go to the environments and to the store through `codec`."""

    def __init__(
        self,
        feature_extractor: Encoder,
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

    def forward(self, obs: Dict[str, Tensor]) -> Tuple[Sequence[Tensor], Tensor, Tensor]:
        """The sampled actions, one per head, their log-probability summed over the heads and the values of `obs`."""
        feat = self.feature_extractor(obs, phase="act")
        values = self.critic(feat)
        actor_out: List[Tensor] = self.actor(feat)
        actions: List[Tensor] = []
        logprobs: List[Tensor] = []
        for i, out in enumerate(actor_out):
            if self.codec.is_continuous(i):
                normal = normal_head(out)
                if self.actor.distribution == "tanh_normal":
                    # Returned before the tanh, which `env_actions` applies: the agent computes the same
                    # log-probability from them
                    action = normal.sample().float()
                    logprobs.append(normal.log_prob(action) - tanh_log_abs_det_jacobian(action))
                else:
                    action = normal.sample()
                    logprobs.append(normal.log_prob(action))
            else:
                categorical = OneHotCategorical(logits=out)
                action = categorical.sample()
                logprobs.append(categorical.log_prob(action))
            actions.append(action)
        return tuple(actions), torch.stack(logprobs, dim=-1).sum(dim=-1, keepdim=True), values

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
        if self.actor.distribution == "tanh_normal":
            actions = [
                safetanh(a, eps=torch.finfo(a.dtype).resolution) if self.codec.is_continuous(i) else a
                for i, a in enumerate(actions)
            ]
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
        actions: List[Tensor] = []
        for i, out in enumerate(actor_out):
            if self.codec.is_continuous(i):
                action = torch.chunk(out, chunks=2, dim=-1)[0] if greedy else normal_head(out).sample()
                if self.actor.distribution == "tanh_normal":
                    action = safetanh(action, eps=torch.finfo(action.dtype).resolution)
            else:
                categorical = OneHotCategorical(logits=out)
                action = categorical.mode if greedy else categorical.sample()
            actions.append(action)
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
