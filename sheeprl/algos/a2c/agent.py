from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, OneHotCategorical

from sheeprl.algos.ppo.agent import PPOActor, PPOPlayer
from sheeprl.models.models import MLP
from sheeprl.utils.fabric import get_single_device_fabric


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


class A2CAgent(nn.Module):
    def __init__(
        self,
        actions_dim: Sequence[int],
        obs_space: gymnasium.spaces.Dict,
        encoder_cfg: Dict[str, Any],
        actor_cfg: Dict[str, Any],
        critic_cfg: Dict[str, Any],
        distribution_cfg: Dict[str, Any],
        mlp_keys: Sequence[str],
        is_continuous: bool = False,
    ):
        super().__init__()
        self.actions_dim = actions_dim
        self.obs_space = obs_space
        self.distribution_cfg = distribution_cfg
        self.mlp_keys = mlp_keys
        self.is_continuous = is_continuous

        # Feature extractor
        mlp_input_dim = sum([obs_space[k].shape[0] for k in mlp_keys])
        self.feature_extractor = (
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
        features_dim = self.feature_extractor.output_dim

        # Critic
        self.critic = MLP(
            input_dims=features_dim,
            output_dim=1,
            hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
            activation=eval(critic_cfg.dense_act),
            norm_layer=[nn.LayerNorm for _ in range(critic_cfg.mlp_layers)] if critic_cfg.layer_norm else None,
            norm_args=(
                [{"normalized_shape": critic_cfg.dense_units} for _ in range(critic_cfg.mlp_layers)]
                if critic_cfg.layer_norm
                else None
            ),
        )

        # Actor
        actor_backbone = MLP(
            input_dims=features_dim,
            output_dim=None,
            hidden_sizes=[actor_cfg.dense_units] * actor_cfg.mlp_layers,
            activation=eval(actor_cfg.dense_act),
            flatten_dim=None,
            norm_layer=[nn.LayerNorm] * actor_cfg.mlp_layers if actor_cfg.layer_norm else None,
            norm_args=(
                [{"normalized_shape": actor_cfg.dense_units} for _ in range(actor_cfg.mlp_layers)]
                if actor_cfg.layer_norm
                else None
            ),
        )
        if is_continuous:
            # Output is a tuple of two elements: mean and log_std, one for every action
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, sum(actions_dim) * 2)])
        else:
            # Output is a tuple of one element: logits, one for every action
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, action_dim) for action_dim in actions_dim])
        self.actor = PPOActor(actor_backbone, actor_heads, is_continuous=is_continuous)

    def forward(
        self, obs: Dict[str, Tensor], actions: Optional[List[Tensor]] = None, greedy: bool = False
    ) -> Tuple[Sequence[Tensor], Tensor, Tensor]:
        feat = self.feature_extractor(obs)
        pre_dist: List[Tensor] = self.actor(feat)
        values = self.critic(feat)
        if self.is_continuous:
            mean, log_std = torch.chunk(pre_dist[0], chunks=2, dim=-1)
            std = log_std.exp()
            normal = Independent(Normal(mean, std), 1)
            if actions is None:
                actions = normal.mode if greedy else normal.sample()
            else:
                # always composed by a tuple of one element containing all the
                # continuous actions
                actions = actions[0]
            log_prob = normal.log_prob(actions)
            return tuple([actions]), log_prob.unsqueeze(dim=-1), values
        else:
            should_append = False
            actions_logprobs: List[Tensor] = []
            actions_dist: List[Distribution] = []
            if actions is None:
                should_append = True
                actions: List[Tensor] = []
            for i, logits in enumerate(pre_dist):
                actions_dist.append(OneHotCategorical(logits=logits))
                if should_append:
                    actions.append(actions_dist[-1].mode if greedy else actions_dist[-1].sample())
                actions_logprobs.append(actions_dist[-1].log_prob(actions[i]))
            return tuple(actions), torch.stack(actions_logprobs, dim=-1).sum(dim=-1, keepdim=True), values

    def get_value(self, obs: Dict[str, Tensor]) -> Tensor:
        feat = self.feature_extractor(obs)
        return self.critic(feat)


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    agent_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[A2CAgent, PPOPlayer]:
    agent = A2CAgent(
        actions_dim=actions_dim,
        obs_space=obs_space,
        encoder_cfg=cfg.algo.encoder,
        actor_cfg=cfg.algo.actor,
        critic_cfg=cfg.algo.critic,
        mlp_keys=cfg.algo.mlp_keys.encoder,
        distribution_cfg=cfg.distribution,
        is_continuous=is_continuous,
    )
    if agent_state:
        agent.load_state_dict(agent_state)

    # Setup player agent
    player = PPOPlayer(copy.deepcopy(agent.feature_extractor), copy.deepcopy(agent.actor), copy.deepcopy(agent.critic))

    # Setup training agent
    agent.feature_extractor = fabric.setup_module(agent.feature_extractor)
    agent.critic = fabric.setup_module(agent.critic)
    agent.actor = fabric.setup_module(agent.actor)

    # Setup player agent
    fabric_player = get_single_device_fabric(fabric)
    player.feature_extractor = fabric_player.setup_module(player.feature_extractor)
    player.critic = fabric_player.setup_module(player.critic)
    player.actor = fabric_player.setup_module(player.actor)

    # Tie weights between the agent and the player
    for agent_p, player_p in zip(agent.feature_extractor.parameters(), player.feature_extractor.parameters()):
        player_p.data = agent_p.data
    for agent_p, player_p in zip(agent.actor.parameters(), player.actor.parameters()):
        player_p.data = agent_p.data
    for agent_p, player_p in zip(agent.critic.parameters(), player.critic.parameters()):
        player_p.data = agent_p.data
    return agent, player
