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
from torch_geometric.nn import GCNConv

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
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
        )

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-1)
        return self.model(x)


class GCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,  # output_dim of MLPEncoder
        features_dim: int,  # tbd
        num_layers: int = 2,  # tbd
        # keys: Sequence[str], # ["nodes_embeddings", "edge_links"]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = features_dim
        self.convs = [GCNConv(features_dim, features_dim) for _ in range(num_layers)]

    def forward(self, nodes_embeddings, edges):
        for conv in self.convs:
            nodes_embeddings = conv(nodes_embeddings, edges)
        return nodes_embeddings


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.linear = Linear(3 * input_dim, input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.value_head = Linear(input_dim, 1)

    def forward(self, x, visited_nodes) -> Tensor:
        y = x.detach()
        visited_nodes = visited_nodes.detach()
        mask = torch.zeros(y.shape[:-1], dtype=torch.bool)
        if visited_nodes.shape[0] == 0:
            first_node = torch.zeros_like(y[0])
            last_node = torch.zeros_like(y[0])
        else:
            first_node = y[visited_nodes[0].item()]
            last_node = y[visited_nodes[-1].item()]
            mask[torch.tensor(visited_nodes)] = 1

        graph_embeddig = torch.mean(y, dim=0)
        context = torch.concat([graph_embeddig, first_node, last_node], dim=0).unsqueeze(0)
        context = self.linear(context)
        attn = self.attention(context, x, x, need_weights=False, attn_mask=mask.unsqueeze(0))
        value = self.value_head(attn[0])
        return value


class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.linear = Linear(3 * input_dim, input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def forward(self, x, visited_nodes) -> Tensor:
        y = x.clone().detach()
        mask = torch.zeros(y.shape[:-1], dtype=torch.bool)
        if visited_nodes.shape[0] == 0:
            first_node = torch.zeros_like(y[0])
            last_node = torch.zeros_like(y[0])
        else:
            first_node = y[visited_nodes[0].item()]
            last_node = y[visited_nodes[-1].item()]
            mask[torch.tensor(visited_nodes)] = 1
        graph_embeddig = torch.mean(y, dim=0)
        context = torch.concat([graph_embeddig, first_node, last_node], dim=0).unsqueeze(0)
        context = self.linear(context)
        _, attn_weights = self.attention(context, x, x, need_weights=True, attn_mask=mask.unsqueeze(0))
        return attn_weights


class PPOAgent(nn.Module):
    def __init__(
        self,
        # actions_dim: Sequence[int],
        obs_space: gymnasium.spaces.Dict,
        encoder_cfg: Dict[str, Any],
        actor_cfg: Dict[str, Any],
        critic_cfg: Dict[str, Any],
        mlp_keys: Sequence[str],
        # other_keys: Sequence[str],
        distribution_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.distribution_cfg = distribution_cfg
        # self.actions_dim = actions_dim
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
        # TODO add graph encoder? or create a graph ppo agent? CREATE A GRAPH PPO AGENT
        self.feature_extractor = GCNEncoder(
            self.mlp_encoder.output_dim,
            encoder_cfg.mlp_features_dim,
            encoder_cfg.num_layers,
        )
        # self.feature_extractor = MultiEncoder(cnn_encoder, mlp_encoder)
        features_dim = self.feature_extractor.output_dim
        self.critic = Critic(features_dim, critic_cfg.num_heads)
        self.actor = Actor(features_dim, actor_cfg.num_heads)

    def forward(
        self, obs: Dict[str, Tensor], actions: Optional[List[Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Tensor, Tensor, Tensor]:
        nodes_embeddings = self.mlp_encoder(obs)
        feat = self.feature_extractor(nodes_embeddings, obs["edge_links"])
        pre_dist: List[Tensor] = [self.actor(feat, obs["partial_solution"])]
        values = self.critic(feat, obs["partial_solution"])

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
        feat = self.feature_extractor(obs)
        return self.critic(feat)

    def get_greedy_actions(self, obs: Dict[str, Tensor]) -> Sequence[Tensor]:
        feat = self.feature_extractor(obs)
        out = self.actor_backbone(feat)
        pre_dist: List[Tensor] = [head(out) for head in self.actor_heads]
        return tuple(
            [
                OneHotCategoricalValidateArgs(logits=logits, validate_args=self.distribution_cfg.validate_args).mode
                for logits in pre_dist
            ]
        )


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    agent_state: Optional[Dict[str, Tensor]] = None,
) -> _FabricModule:
    agent = PPOAgent(
        actions_dim=actions_dim,
        obs_space=obs_space,
        encoder_cfg=cfg.algo.encoder,
        actor_cfg=cfg.algo.actor,
        critic_cfg=cfg.algo.critic,
        mlp_keys=cfg.algo.mlp_keys.encoder,
        graph_keys=cfg.algo.graph_keys.encoder,
        distribution_cfg=cfg.distribution,
    )
    if agent_state:
        agent.load_state_dict(agent_state)
    agent = fabric.setup_module(agent)

    return agent
