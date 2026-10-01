from __future__ import annotations

from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium
import hydra
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, OneHotCategorical

from sheeprl.models.models import MLP, MultiEncoder, NatureCNN
from sheeprl.utils.utils import safeatanh, safetanh


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features_dim: int,
        screen_size: int,
        keys: Sequence[str],
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = (in_channels, screen_size, screen_size)
        self.output_dim = features_dim
        self.model = NatureCNN(in_channels=in_channels, features_dim=features_dim, screen_size=screen_size)

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-3)
        return self.model(x)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_dim: int | None,
        keys: Sequence[str],
        dense_units: int = 64,
        mlp_layers: int = 2,
        dense_act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = input_dim
        self.output_dim = features_dim if features_dim else dense_units
        if mlp_layers == 0:
            self.model = nn.Identity()
            self.output_dim = input_dim
        else:
            self.model = MLP(
                input_dim,
                features_dim,
                [dense_units] * mlp_layers,
                activation=dense_act,
                norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
                norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
            )

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-1)
        return self.model(x)


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
    def __init__(
        self,
        actions_dim: Sequence[int],
        obs_space: gymnasium.spaces.Dict,
        encoder_cfg: Dict[str, Any],
        actor_cfg: Dict[str, Any],
        critic_cfg: Dict[str, Any],
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        screen_size: int,
        distribution_cfg: Dict[str, Any],
        is_continuous: bool = False,
    ):
        super().__init__()
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
        in_channels = sum([prod(obs_space[k].shape[:-2]) for k in cnn_keys])
        mlp_input_dim = sum([obs_space[k].shape[0] for k in mlp_keys])
        cnn_encoder = (
            CNNEncoder(in_channels, encoder_cfg.cnn_features_dim, screen_size, cnn_keys)
            if cnn_keys is not None and len(cnn_keys) > 0
            else None
        )
        mlp_encoder = (
            MLPEncoder(
                mlp_input_dim,
                encoder_cfg.mlp_features_dim,
                mlp_keys,
                encoder_cfg.dense_units,
                encoder_cfg.mlp_layers,
                hydra.utils.get_class(encoder_cfg.dense_act),
                encoder_cfg.layer_norm,
            )
            if mlp_keys is not None and len(mlp_keys) > 0
            else None
        )
        self.feature_extractor = MultiEncoder(cnn_encoder, mlp_encoder)
        if encoder_cfg.ortho_init:
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
        tanh_actions = actions[0].float()
        actions = safeatanh(tanh_actions, eps=torch.finfo(tanh_actions.dtype).resolution)
        log_prob = normal.log_prob(actions)
        log_prob -= 2.0 * (
            torch.log(torch.tensor([2.0], dtype=actions.dtype, device=actions.device))
            - tanh_actions
            - torch.nn.functional.softplus(-2.0 * tanh_actions)
        ).sum(-1, keepdim=False)
        return tanh_actions, log_prob.unsqueeze(dim=-1), normal.entropy().unsqueeze(dim=-1)

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


class PPOPlayer(nn.Module):
    def __init__(self, feature_extractor: MultiEncoder, actor: PPOActor, critic: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.critic = critic
        self.actor = actor

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
        actions = normal.sample().float()
        tanh_actions = safetanh(actions, eps=torch.finfo(actions.dtype).resolution)
        log_prob = normal.log_prob(actions)
        log_prob -= 2.0 * (
            torch.log(torch.tensor([2.0], dtype=actions.dtype, device=actions.device))
            - tanh_actions
            - torch.nn.functional.softplus(-2.0 * tanh_actions)
        ).sum(-1, keepdim=False)
        return tanh_actions, log_prob.unsqueeze(dim=-1)

    def forward(self, obs: Dict[str, Tensor]) -> Tuple[Sequence[Tensor], Tensor, Tensor]:
        feat = self.feature_extractor(obs)
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
        feat = self.feature_extractor(obs)
        return self.critic(feat)

    def get_actions(self, obs: Dict[str, Tensor], greedy: bool = False) -> Sequence[Tensor]:
        feat = self.feature_extractor(obs)
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
                actions = safeatanh(actions, eps=torch.finfo(actions.dtype).resolution)
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
