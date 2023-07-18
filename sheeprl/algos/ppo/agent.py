from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, OneHotCategorical

from sheeprl.models.models import MLP, MultiEncoder


class PPOAgent(nn.Module):
    def __init__(
        self,
        actions_dim: List[int],
        obs_space: Dict[str, Any],
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        cnn_channels_multiplier: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        cnn_act: str = "ReLU",
        mlp_act: str = "ReLU",
        device: Union[str, torch.device] = "cpu",
        layer_norm: bool = False,
        is_continuous: bool = False,
    ):
        if cnn_channels_multiplier <= 0:
            raise ValueError(f"cnn_channels_multiplier must be greater than zero, given {cnn_channels_multiplier}")
        if dense_units <= 0:
            raise ValueError(f"dense_units must be greater than zero, given {dense_units}")

        try:
            conv_act = getattr(nn, cnn_act)
        except:
            raise ValueError(
                f"Invalid value for cnn_act, given {cnn_act}, must be one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
            )

        try:
            dense_act = getattr(nn, mlp_act)
        except:
            raise ValueError(
                f"Invalid value for mlp_act, given {mlp_act}, must be one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
            )

        super().__init__()
        self.actions_dim = actions_dim
        self.feature_extractor = MultiEncoder(
            obs_space=obs_space,
            cnn_keys=cnn_keys,
            mlp_keys=mlp_keys,
            cnn_channels_multiplier=cnn_channels_multiplier,
            mlp_layers=mlp_layers,
            dense_units=dense_units,
            cnn_act=conv_act,
            mlp_act=dense_act,
            device=device,
            layer_norm=layer_norm,
        )
        self.is_continuous = is_continuous
        features_dim = self.feature_extractor.cnn_output_dim + self.feature_extractor.mlp_output_dim
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
            self.actor_heads = nn.ModuleList([nn.Linear(dense_units, np.sum(actions_dim) * 2)])
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
            log_prob = normal.log_prob(actions)
            return tuple([actions]), log_prob, normal.entropy(), values
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
