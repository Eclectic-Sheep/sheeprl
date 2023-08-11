from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, OneHotCategorical

from sheeprl.models.models import MLP, MultiEncoder, NatureCNN


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
        features_dim: int,
        keys: Sequence[str],
        dense_units: int = 64,
        mlp_layers: int = 2,
        dense_act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = input_dim
        self.output_dim = features_dim
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


class PPOAgent(nn.Module):
    def __init__(
        self,
        actions_dim: List[int],
        obs_space: Dict[str, Any],
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        cnn_features_dim: int = 512,
        mlp_features_dim: int = 64,
        screen_size: int = 64,
        cnn_channels_multiplier: int = 1,
        mlp_layers: int = 2,
        dense_units: int = 64,
        mlp_act: str = "ReLU",
        layer_norm: bool = False,
        is_continuous: bool = False,
    ):
        if cnn_channels_multiplier <= 0:
            raise ValueError(f"cnn_channels_multiplier must be greater than zero, given {cnn_channels_multiplier}")
        if dense_units <= 0:
            raise ValueError(f"dense_units must be greater than zero, given {dense_units}")
        try:
            dense_act = getattr(nn, mlp_act)
        except AttributeError:
            raise ValueError(
                f"Invalid value for mlp_act, given {mlp_act}, must be one of "
                "https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
            )
        super().__init__()
        self.actions_dim = actions_dim
        in_channels = sum([prod(obs_space[k].shape[:-2]) for k in cnn_keys])
        mlp_input_dim = sum([obs_space[k].shape[0] for k in mlp_keys])
        cnn_encoder = (
            CNNEncoder(in_channels, cnn_features_dim, screen_size, cnn_keys)
            if cnn_keys is not None and len(cnn_keys) > 0
            else None
        )
        mlp_encoder = (
            MLPEncoder(mlp_input_dim, mlp_features_dim, mlp_keys, dense_units, mlp_layers, dense_act, layer_norm)
            if mlp_keys is not None and len(mlp_keys) > 0
            else None
        )
        self.feature_extractor = MultiEncoder(cnn_encoder, mlp_encoder)
        self.is_continuous = is_continuous
        features_dim = self.feature_extractor.output_dim
        self.critic = MLP(
            input_dims=features_dim, output_dim=1, hidden_sizes=[dense_units] * mlp_layers, activation=dense_act
        )
        self.actor_backbone = MLP(
            input_dims=features_dim,
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=dense_act,
            flatten_dim=None,
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
        )
        if is_continuous:
            self.actor_heads = nn.ModuleList([nn.Linear(dense_units, sum(actions_dim) * 2)])
        else:
            self.actor_heads = nn.ModuleList([nn.Linear(dense_units, action_dim) for action_dim in actions_dim])

    def forward(
        self, obs: Dict[str, Tensor], actions: Optional[List[Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Tensor, Tensor, Tensor]:
        feat = self.feature_extractor(obs)
        out: Tensor = self.actor_backbone(feat)
        pre_dist: List[Tensor] = [head(out) for head in self.actor_heads]
        values = self.critic(feat)
        if self.is_continuous:
            mean, log_std = torch.chunk(pre_dist[0], chunks=2, dim=-1)
            std = log_std.exp()
            normal = Independent(Normal(mean, std), 1)
            if actions is None:
                actions = normal.sample()
            else:
                # always composed by a tuple of one element containing all the
                # continuous actions
                actions = actions[0]
            log_prob = normal.log_prob(actions)
            return tuple([actions]), log_prob.unsqueeze(dim=-1), normal.entropy().unsqueeze(dim=-1), values
        else:
            should_append = False
            actions_dist: List[Distribution] = []
            actions_entropies: List[Tensor] = []
            actions_logprobs: List[Tensor] = []
            if actions is None:
                should_append = True
                actions: List[Tensor] = []
            for i, logits in enumerate(pre_dist):
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

    def get_value(self, obs: Dict[str, Tensor]) -> Tensor:
        feat = self.feature_extractor(obs)
        return self.critic(feat)

    def get_greedy_actions(self, obs: Dict[str, Tensor]) -> Sequence[Tensor]:
        feat = self.feature_extractor(obs)
        out = self.actor_backbone(feat)
        pre_dist: List[Tensor] = [head(out) for head in self.actor_heads]
        if self.is_continuous:
            return [torch.chunk(pre_dist[0], 2, -1)[0]]
        else:
            return tuple([OneHotCategorical(logits=logits).mode for logits in pre_dist])
