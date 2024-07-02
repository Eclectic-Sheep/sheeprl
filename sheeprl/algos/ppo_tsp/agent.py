from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium
import torch
import torch.nn as nn
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN

from sheeprl.models.models import MLP
from sheeprl.utils.distribution import OneHotCategoricalValidateArgs


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,  # 2
        features_dim: int | None,  # tbd
        keys: Sequence[str],  # ["nodes"]
        dense_units: int = 64,
        mlp_layers: int = 2,  # 1
        dense_act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = input_dim
        self.output_dim = features_dim if features_dim else dense_units
        self.model = MLP(
            input_dim,
            features_dim,
            [dense_units] * mlp_layers,
            activation=dense_act,
            norm_layer=[nn.BatchNorm1d for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
        )

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-1)
        return self.model(x)


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dense_units: int = 256,
    ):
        super().__init__()
        self.linear = Linear(3 * input_dim, input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.value_head = nn.Sequential(Linear(input_dim, dense_units), nn.ReLU(), Linear(dense_units, 1))

    def forward(self, x, first_node, mask) -> Tensor:
        y = x.detach()
        graph_embeddig = torch.mean(y, dim=-2)
        first_node_features = y[torch.arange(y.shape[0]), first_node.long().flatten()]
        context = torch.concat([graph_embeddig, first_node_features, first_node_features], dim=-1)
        context = self.linear(context).unsqueeze(1)
        attn = self.attention(
            context.permute(1, 0, 2),
            x.permute(1, 0, 2),
            x.permute(1, 0, 2),
            need_weights=False,
            key_padding_mask=mask.to(dtype=torch.bool),
        )
        value = self.value_head(attn[0])
        return value.squeeze(0)


class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.linear = Linear(3 * input_dim, input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def forward(self, x, first_node, current_node, mask) -> Tensor:
        y = x.clone().detach()
        graph_embeddig = torch.mean(y, dim=-2)
        first_node_features = y[torch.arange(y.shape[0]), first_node.long().flatten()]
        current_node_features = y[torch.arange(y.shape[0]), current_node.long().flatten()]
        context = torch.concat([graph_embeddig, first_node_features, current_node_features], dim=-1)
        context = self.linear(context).unsqueeze(1)
        _, attn_weights = self.attention(
            context.permute(1, 0, 2),
            x.permute(1, 0, 2),
            x.permute(1, 0, 2),
            need_weights=True,
            key_padding_mask=mask.to(dtype=torch.bool),
        )
        return attn_weights.squeeze(1)


class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_space: gymnasium.spaces.Dict,
        encoder_cfg: Dict[str, Any],
        actor_cfg: Dict[str, Any],
        critic_cfg: Dict[str, Any],
        mlp_keys: Sequence[str],
        distribution_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.distribution_cfg = distribution_cfg
        self.actions_dim = [obs_space[k].shape[0] for k in mlp_keys]
        mlp_input_dim = sum([obs_space[k].shape[-1] for k in mlp_keys])

        self.mlp_encoder = (
            MLPEncoder(
                mlp_input_dim,
                encoder_cfg.mlp_features_dim,
                mlp_keys,
                encoder_cfg.dense_units,
                encoder_cfg.mlp_layers,
                eval(encoder_cfg.dense_act),
                encoder_cfg.layer_norm,
            )
            if mlp_keys is not None and len(mlp_keys) > 0
            else None
        )
        self.feature_extractor = GCN(
            self.mlp_encoder.output_dim,
            encoder_cfg.mlp_features_dim,
            encoder_cfg.num_layers,
        )
        # self.feature_extractor = MultiEncoder(cnn_encoder, mlp_encoder)
        features_dim = encoder_cfg.mlp_features_dim
        self.critic = Critic(features_dim, critic_cfg.num_heads, dense_units=critic_cfg.dense_units)
        self.actor = Actor(features_dim, actor_cfg.num_heads)

    def forward(
        self, obs: Dict[str, Tensor], actions: Optional[List[Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Tensor, Tensor, Tensor]:
        nodes_embeddings = self.mlp_encoder(obs)

        batch_size, num_nodes, _ = nodes_embeddings.shape
        my_loader = DataLoader(
            [
                Data(x=nodes_embeddings[idx], edge_index=obs["edge_links"][idx].transpose(-1, -2))
                for idx in range(nodes_embeddings.shape[0])
            ],
            batch_size=batch_size,
            shuffle=False,
        )
        batch = next(iter(my_loader))

        feat = self.feature_extractor(batch.x, batch.edge_index.to(dtype=torch.long))
        feat = feat.view(batch_size, num_nodes, -1)
        pre_dist: List[Tensor] = [self.actor(feat, obs["first_node"], obs["current_node"], obs["mask"])]
        values = self.critic(feat, obs["first_node"], obs["mask"])

        should_append = False
        actions_logprobs: List[Tensor] = []
        actions_entropies: List[Tensor] = []
        actions_dist: List[Distribution] = []
        if actions is None:
            should_append = True
            actions: List[Tensor] = []
        for i, probs in enumerate(pre_dist):
            actions_dist.append(
                OneHotCategoricalValidateArgs(probs=probs, validate_args=self.distribution_cfg.validate_args)
            )
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
        nodes_embeddings = self.mlp_encoder(obs)

        batch_size, num_nodes, _ = nodes_embeddings.shape
        my_loader = DataLoader(
            [
                Data(x=nodes_embeddings[idx], edge_index=obs["edge_links"][idx].transpose(-1, -2))
                for idx in range(nodes_embeddings.shape[0])
            ],
            batch_size=batch_size,
            shuffle=False,
        )
        batch = next(iter(my_loader))

        feat = self.feature_extractor(batch.x, batch.edge_index.to(dtype=torch.long))
        feat = feat.view(batch_size, num_nodes, -1)
        return self.critic(feat, obs["first_node"], obs["mask"])

    def get_greedy_actions(self, obs: Dict[str, Tensor]) -> Sequence[Tensor]:
        nodes_embeddings = self.mlp_encoder(obs)

        batch_size, num_nodes, _ = nodes_embeddings.shape
        my_loader = DataLoader(
            [
                Data(x=nodes_embeddings[idx], edge_index=obs["edge_links"][idx].transpose(-1, -2))
                for idx in range(nodes_embeddings.shape[0])
            ],
            batch_size=batch_size,
            shuffle=False,
        )
        batch = next(iter(my_loader))

        feat = self.feature_extractor(batch.x, batch.edge_index.to(dtype=torch.long))
        feat = feat.view(batch_size, num_nodes, -1)
        pre_dist: List[Tensor] = [self.actor(feat, obs["first_node"], obs["current_node"], obs["mask"])]
        return tuple(
            [
                OneHotCategoricalValidateArgs(logits=logits, validate_args=self.distribution_cfg.validate_args).mode
                for logits in pre_dist
            ]
        )


def build_agent(
    fabric: Fabric,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    agent_state: Optional[Dict[str, Tensor]] = None,
) -> _FabricModule:
    agent = PPOAgent(
        obs_space=obs_space,
        encoder_cfg=cfg.algo.encoder,
        actor_cfg=cfg.algo.actor,
        critic_cfg=cfg.algo.critic,
        mlp_keys=cfg.algo.mlp_keys.encoder,
        distribution_cfg=cfg.distribution,
    )
    if agent_state:
        agent.load_state_dict(agent_state)
    agent = fabric.setup_module(agent)

    return agent
