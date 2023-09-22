from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Independent, Normal, OneHotCategorical

from sheeprl.algos.dreamer_v3.agent import LayerNormGRUCell
from sheeprl.algos.ppo.agent import CNNEncoder, MLPEncoder
from sheeprl.models.models import MLP, MultiEncoder


class RecurrentModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        gru_hidden_size: int,
        mlp_dense_units: int,
        mlp_activation: nn.Module,
        gru_bias: bool,
        mlp_bias: bool,
        gru_layer_norm: bool,
        mlp_layer_norm: bool,
        mlp_pre_rnn: bool = True,
    ) -> None:
        super().__init__()
        self._mlp_pre_rnn = mlp_pre_rnn
        if mlp_pre_rnn:
            self._mlp = MLP(
                input_dims=input_size,
                output_dim=None,
                hidden_sizes=[mlp_dense_units],
                activation=mlp_activation,
                layer_args={"bias": mlp_bias},
                norm_layer=[nn.LayerNorm] if mlp_layer_norm else None,
                norm_args=[{"normalized_shape": mlp_dense_units, "eps": 1e-3}] if mlp_layer_norm else None,
            )
        self._gru = LayerNormGRUCell(
            input_size=mlp_dense_units if mlp_pre_rnn else input_size,
            hidden_size=gru_hidden_size,
            bias=gru_bias,
            batch_first=False,
            layer_norm=gru_layer_norm,
        )

    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        feat = self._mlp(input) if self._mlp_pre_rnn else input
        out = self._gru(feat, hx)
        return out


class RecurrentPPOAgent(nn.Module):
    def __init__(
        self,
        actions_dim: Sequence[int],
        obs_space: Dict[str, Any],
        encoder_cfg: DictConfig,
        rnn_cfg: DictConfig,
        actor_cfg: DictConfig,
        critic_cfg: DictConfig,
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        is_continuous: bool,
        num_envs: int = 1,
        screen_size: int = 64,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.actions_dim = actions_dim
        self.rnn_hidden_size = rnn_cfg.gru.hidden_size
        self.num_envs = num_envs

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
                eval(encoder_cfg.dense_act),
                encoder_cfg.layer_norm,
            )
            if mlp_keys is not None and len(mlp_keys) > 0
            else None
        )
        self.feature_extractor = MultiEncoder(cnn_encoder, mlp_encoder)
        self.is_continuous = is_continuous
        features_dim = self.feature_extractor.output_dim

        self.rnn = RecurrentModel(
            input_size=int(features_dim + sum(actions_dim)),
            gru_hidden_size=rnn_cfg.gru.hidden_size,
            mlp_dense_units=rnn_cfg.mlp.dense_units,
            mlp_activation=eval(rnn_cfg.mlp.activation),
            gru_bias=rnn_cfg.gru.bias,
            mlp_bias=rnn_cfg.mlp.bias,
            gru_layer_norm=rnn_cfg.gru.layer_norm,
            mlp_layer_norm=rnn_cfg.mlp.layer_norm,
            mlp_pre_rnn=rnn_cfg.mlp_pre_rnn,
        )

        self.critic = MLP(
            input_dims=self.rnn_hidden_size,
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
        self.actor_backbone = MLP(
            input_dims=self.rnn_hidden_size,
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
            self.actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, int(sum(actions_dim)) * 2)])
        else:
            self.actor_heads = nn.ModuleList(
                [nn.Linear(actor_cfg.dense_units, action_dim) for action_dim in actions_dim]
            )

        # Initial recurrent states for both the actor and critic rnn
        self._initial_states: Tensor = self.reset_hidden_states()

    @property
    def initial_states(self) -> Tensor:
        return self._initial_states

    @initial_states.setter
    def initial_states(self, value: Tensor) -> None:
        self._initial_states = value

    def reset_hidden_states(self) -> Tensor:
        hx = torch.zeros(1, self.num_envs, self.rnn_hidden_size, device=self.device)
        return hx

    def get_greedy_actions(
        self, obs: Dict[str, Tensor], prev_hx: Tensor, prev_actions: Tensor
    ) -> Tuple[Tuple[Tensor, ...], Tensor]:
        embedded_obs = self.feature_extractor(obs)
        hx = self.rnn(torch.cat((embedded_obs, prev_actions), dim=-1), prev_hx)
        pre_dist = self.get_pre_dist(hx)
        actions = []
        if self.is_continuous:
            dist = Independent(Normal(*pre_dist), 1)
            actions.append(dist.mode)
        else:
            for logits in pre_dist:
                dist = OneHotCategorical(logits=logits)
                actions.append(dist.mode)
        return tuple(actions), hx

    def get_sampled_actions(
        self, pre_dist: Tuple[Tensor, ...], actions: Optional[List[Tensor]] = None
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor]:
        logprobs = []
        sampled_actions = []
        entropies = []
        if self.is_continuous:
            dist = Independent(Normal(*pre_dist), 1)
            if actions is None:
                actions = dist.sample()
            else:
                # always composed by a tuple of one element containing all the
                # continuous actions
                actions = actions[0]
            sampled_actions.append(actions)
            entropies.append(dist.entropy())
            logprobs.append(dist.log_prob(actions))
        else:
            for i, logits in enumerate(pre_dist):
                dist = OneHotCategorical(logits=logits)
                if actions is None:
                    sampled_actions.append(dist.sample())
                else:
                    sampled_actions.append(actions[i])
                entropies.append(dist.entropy())
                logprobs.append(dist.log_prob(sampled_actions[-1]))
        return (
            tuple(sampled_actions),
            torch.stack(logprobs, dim=-1).sum(dim=-1, keepdim=True),
            torch.stack(entropies, dim=-1).sum(dim=-1, keepdim=True),
        )

    def get_pre_dist(self, hx: Tensor) -> Union[Tuple[Tensor, ...], Tuple[Tensor, Tensor]]:
        actor_features = self.actor_backbone(hx)
        pre_dist: List[Tensor] = [head(actor_features) for head in self.actor_heads]
        if self.is_continuous:
            mean, log_std = torch.chunk(pre_dist[0], chunks=2, dim=-1)
            std = log_std.exp()
            return (mean, std)
        else:
            return tuple(pre_dist)

    def get_values(self, hx: Tensor) -> Tensor:
        return self.critic(hx)

    def forward(
        self, obs: Dict[str, Tensor], prev_actions: Tensor, prev_hx: Tensor, actions: Optional[List[Tensor]] = None
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor, Tensor, Tensor]:
        """Compute actor logits and critic values.

        Args:
            obs (Tensor): observations collected (possibly padded with zeros).
            prev_actions (Tensor): the previous actions.
            prev_hx (Tensor): the previous state of the GRU.

        Returns:
            actions (Tuple[Tensor, ...]): the sampled actions
            logprobs (Tensor): the log probabilities of the actions w.r.t. their distributions.
            entropies (Tensor): the entropies of the actions distributions.
            values (Tensor): the state values.
            hx (Tensor): the new recurrent state.
        """
        embedded_obs = self.feature_extractor(obs)
        hx = self.rnn(torch.cat((embedded_obs, prev_actions), dim=-1), prev_hx)
        values = self.get_values(hx)
        pre_dist = self.get_pre_dist(hx)
        actions, logprobs, entropies = self.get_sampled_actions(pre_dist, actions)
        return actions, logprobs, entropies, values, hx
