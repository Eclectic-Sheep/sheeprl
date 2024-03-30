import copy
from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor
from torch.distributions import Independent, Normal, OneHotCategorical

from sheeprl.algos.ppo.agent import CNNEncoder, MLPEncoder, PPOActor
from sheeprl.models.models import MLP, MultiEncoder
from sheeprl.utils.fabric import get_single_device_fabric


class RecurrentModel(nn.Module):
    def __init__(
        self, input_size: int, lstm_hidden_size: int, pre_rnn_mlp_cfg: Dict[str, Any], post_rnn_mlp_cfg: Dict[str, Any]
    ) -> None:
        super().__init__()
        if pre_rnn_mlp_cfg.apply:
            self._pre_mlp = MLP(
                input_dims=input_size,
                output_dim=None,
                hidden_sizes=[pre_rnn_mlp_cfg.dense_units],
                activation=eval(pre_rnn_mlp_cfg.activation),
                layer_args={"bias": pre_rnn_mlp_cfg.bias},
                norm_layer=[nn.LayerNorm] if pre_rnn_mlp_cfg.layer_norm else None,
                norm_args=(
                    [{"normalized_shape": pre_rnn_mlp_cfg.dense_units, "eps": 1e-3}]
                    if pre_rnn_mlp_cfg.layer_norm
                    else None
                ),
            )
        else:
            self._pre_mlp = nn.Identity()
        self._lstm = nn.LSTM(
            input_size=pre_rnn_mlp_cfg.dense_units if pre_rnn_mlp_cfg.apply else input_size,
            hidden_size=lstm_hidden_size,
            batch_first=False,
        )
        if post_rnn_mlp_cfg.apply:
            self._post_mlp = MLP(
                input_dims=lstm_hidden_size,
                output_dim=None,
                hidden_sizes=[post_rnn_mlp_cfg.dense_units],
                activation=eval(post_rnn_mlp_cfg.activation),
                layer_args={"bias": post_rnn_mlp_cfg.bias},
                norm_layer=[nn.LayerNorm] if post_rnn_mlp_cfg.layer_norm else None,
                norm_args=(
                    [{"normalized_shape": post_rnn_mlp_cfg.dense_units, "eps": 1e-3}]
                    if post_rnn_mlp_cfg.layer_norm
                    else None
                ),
            )
            self._output_dim = post_rnn_mlp_cfg.dense_units
        else:
            self._post_mlp = nn.Identity()
            self._output_dim = lstm_hidden_size

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, input: Tensor, states: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x = self._pre_mlp(input)
        self._lstm.flatten_parameters()
        if mask is not None:
            # To avoid: RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
            lengths = mask.sum(dim=0).view(-1).cpu()
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=False, enforce_sorted=False)
        out, states = self._lstm(x, states)
        if mask is not None:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=False, total_length=mask.shape[0])
        shape = out.shape
        return self._post_mlp(out.view(-1, *shape[2:])).view(shape), states


class RecurrentPPOAgent(nn.Module):
    def __init__(
        self,
        actions_dim: Sequence[int],
        obs_space: gymnasium.spaces.Dict,
        encoder_cfg: Dict[str, Any],
        rnn_cfg: Dict[str, Any],
        actor_cfg: Dict[str, Any],
        critic_cfg: Dict[str, Any],
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        is_continuous: bool,
        distribution_cfg: Dict[str, Any],
        num_envs: int = 1,
        screen_size: int = 64,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.num_envs = num_envs
        self.actions_dim = actions_dim
        self.distribution_cfg = distribution_cfg
        self.rnn_hidden_size = rnn_cfg.lstm.hidden_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # Encoder
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

        # Recurrent model
        self.rnn = RecurrentModel(
            input_size=int(features_dim + sum(actions_dim)),
            lstm_hidden_size=rnn_cfg.lstm.hidden_size,
            pre_rnn_mlp_cfg=rnn_cfg.pre_rnn_mlp,
            post_rnn_mlp_cfg=rnn_cfg.post_rnn_mlp,
        )

        # Critic
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

        # Actor
        actor_backbone = MLP(
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
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, int(sum(actions_dim)) * 2)])
        else:
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, action_dim) for action_dim in actions_dim])
        self.actor = PPOActor(actor_backbone, actor_heads, is_continuous)

        # Initial recurrent states for both the actor and critic rnn
        self._initial_states: Tensor = self.reset_hidden_states()

    @property
    def initial_states(self) -> Tuple[Tensor, Tensor]:
        return self._initial_states

    @initial_states.setter
    def initial_states(self, value: Tuple[Tensor, Tensor]) -> None:
        self._initial_states = value

    def reset_hidden_states(self) -> Tuple[Tensor, Tensor]:
        states = (
            torch.zeros(1, self.num_envs, self.rnn_hidden_size, device=self.device),
            torch.zeros(1, self.num_envs, self.rnn_hidden_size, device=self.device),
        )
        return states

    def _get_actions(
        self, pre_dist: Tuple[Tensor, ...], actions: Optional[List[Tensor]] = None
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor]:
        logprobs = []
        entropies = []
        sampled_actions = []
        if self.is_continuous:
            dist = Independent(Normal(*pre_dist), 1)
            if actions is None:
                sampled_actions.append(dist.sample())
            else:
                sampled_actions.append(actions[0])
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

    def _get_pre_dist(self, input: Tensor) -> Union[Tuple[Tensor, ...], Tuple[Tensor, Tensor]]:
        pre_dist: List[Tensor] = self.actor(input)
        if self.is_continuous:
            mean, log_std = torch.chunk(pre_dist[0], chunks=2, dim=-1)
            std = log_std.exp()
            return (mean, std)
        else:
            return tuple(pre_dist)

    def _get_values(self, input: Tensor) -> Tensor:
        return self.critic(input)

    def forward(
        self,
        obs: Dict[str, Tensor],
        prev_actions: Tensor,
        prev_states: Tuple[Tensor, Tensor],
        actions: Optional[List[Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """Compute actor logits and critic values.

        Args:
            obs (Tensor): observations collected (possibly padded with zeros).
            prev_actions (Tensor): the previous actions.
            prev_states (Tuple[Tensor, Tensor]): the previous state of the LSTM.
            actions (List[Tensor], optional): the actions from the replay buffer.
            mask (Tensor, optional): the mask of the padded sequences.

        Returns:
            actions (Tuple[Tensor, ...]): the sampled actions
            logprobs (Tensor): the log probabilities of the actions w.r.t. their distributions.
            entropies (Tensor): the entropies of the actions distributions.
            values (Tensor): the state values.
            states (Tuple[Tensor, Tensor]): the new recurrent states (hx, cx).
        """
        embedded_obs = self.feature_extractor(obs)
        out, states = self.rnn(torch.cat((embedded_obs, prev_actions), dim=-1), prev_states, mask)
        values = self._get_values(out)
        pre_dist = self._get_pre_dist(out)
        actions, logprobs, entropies = self._get_actions(pre_dist, actions)
        return actions, logprobs, entropies, values, states


class RecurrentPPOPlayer(nn.Module):
    def __init__(
        self,
        feature_extractor: MultiEncoder,
        rnn: RecurrentModel,
        actor: PPOActor,
        critic: nn.Module,
        rnn_hidden_size: int,
        actions_dim: Sequence[int],
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.rnn = rnn
        self.critic = critic
        self.actor = actor
        self.rnn_hidden_size = rnn_hidden_size
        self.actions_dim = actions_dim

    @property
    def initial_states(self) -> Tuple[Tensor, Tensor]:
        return self._initial_states

    @initial_states.setter
    def initial_states(self, value: Tuple[Tensor, Tensor]) -> None:
        self._initial_states = value

    def reset_hidden_states(self) -> Tuple[Tensor, Tensor]:
        states = (
            torch.zeros(1, self.num_envs, self.rnn_hidden_size, device=self.device),
            torch.zeros(1, self.num_envs, self.rnn_hidden_size, device=self.device),
        )
        return states

    def _get_actions(
        self, pre_dist: Tuple[Tensor, ...], actions: Optional[List[Tensor]] = None, greedy: bool = False
    ) -> Tuple[Tuple[Tensor, ...], Tensor]:
        logprobs = []
        sampled_actions = []
        if self.actor.is_continuous:
            dist = Independent(Normal(*pre_dist), 1)
            if greedy:
                sampled_actions.append(dist.mode)
            else:
                if actions is None:
                    sampled_actions.append(dist.sample())
                else:
                    sampled_actions.append(actions[0])
            logprobs.append(dist.log_prob(actions))
        else:
            for i, logits in enumerate(pre_dist):
                dist = OneHotCategorical(logits=logits)
                if greedy:
                    sampled_actions.append(dist.mode)
                else:
                    if actions is None:
                        sampled_actions.append(dist.sample())
                    else:
                        sampled_actions.append(actions[i])
                logprobs.append(dist.log_prob(sampled_actions[-1]))
        return (
            tuple(sampled_actions),
            torch.stack(logprobs, dim=-1).sum(dim=-1, keepdim=True),
        )

    def _get_pre_dist(self, input: Tensor) -> Union[Tuple[Tensor, ...], Tuple[Tensor, Tensor]]:
        pre_dist: List[Tensor] = self.actor(input)
        if self.actor.is_continuous:
            mean, log_std = torch.chunk(pre_dist[0], chunks=2, dim=-1)
            std = log_std.exp()
            return (mean, std)
        else:
            return tuple(pre_dist)

    def _get_values(self, input: Tensor) -> Tensor:
        return self.critic(input)

    def forward(
        self,
        obs: Dict[str, Tensor],
        prev_actions: Tensor,
        prev_states: Tuple[Tensor, Tensor],
        actions: Optional[List[Tensor]] = None,
        mask: Optional[Tensor] = None,
        greedy: bool = False,
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """Compute actor logits and critic values.

        Args:
            obs (Tensor): observations collected (possibly padded with zeros).
            prev_actions (Tensor): the previous actions.
            prev_states (Tuple[Tensor, Tensor]): the previous state of the LSTM.
            actions (List[Tensor], optional): the actions from the replay buffer.
            mask (Tensor, optional): the mask of the padded sequences.

        Returns:
            actions (Tuple[Tensor, ...]): the sampled actions
            logprobs (Tensor): the log probabilities of the actions w.r.t. their distributions.
            entropies (Tensor): the entropies of the actions distributions.
            values (Tensor): the state values.
            states (Tuple[Tensor, Tensor]): the new recurrent states (hx, cx).
        """
        embedded_obs = self.feature_extractor(obs)
        out, states = self.rnn(torch.cat((embedded_obs, prev_actions), dim=-1), prev_states, mask)
        values = self._get_values(out)
        pre_dist = self._get_pre_dist(out)
        actions, logprobs = self._get_actions(pre_dist, actions, greedy=greedy)
        return actions, logprobs, values, states

    def get_values(
        self,
        obs: Dict[str, Tensor],
        prev_actions: Tensor,
        prev_states: Tuple[Tensor, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        embedded_obs = self.feature_extractor(obs)
        out, states = self.rnn(torch.cat((embedded_obs, prev_actions), dim=-1), prev_states, mask)
        return self._get_values(out), states

    def get_actions(
        self,
        obs: Dict[str, Tensor],
        prev_actions: Tensor,
        prev_states: Tuple[Tensor, Tensor],
        mask: Optional[Tensor] = None,
        greedy: bool = False,
    ) -> Tuple[Sequence[Tensor], Tuple[Tensor, Tensor]]:
        embedded_obs = self.feature_extractor(obs)
        out, states = self.rnn(torch.cat((embedded_obs, prev_actions), dim=-1), prev_states, mask)
        pre_dist = self._get_pre_dist(out)
        sampled_actions = []
        if self.actor.is_continuous:
            dist = Independent(Normal(*pre_dist), 1)
            if greedy:
                sampled_actions.append(dist.mode)
            else:
                sampled_actions.append(dist.sample())
        else:
            for logits in pre_dist:
                dist = OneHotCategorical(logits=logits)
                if greedy:
                    sampled_actions.append(dist.mode)
                else:
                    sampled_actions.append(dist.sample())
        return tuple(sampled_actions), states


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    agent_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[RecurrentPPOAgent, RecurrentPPOPlayer]:
    agent = RecurrentPPOAgent(
        actions_dim=actions_dim,
        obs_space=obs_space,
        encoder_cfg=cfg.algo.encoder,
        rnn_cfg=cfg.algo.rnn,
        actor_cfg=cfg.algo.actor,
        critic_cfg=cfg.algo.critic,
        cnn_keys=cfg.algo.cnn_keys.encoder,
        mlp_keys=cfg.algo.mlp_keys.encoder,
        is_continuous=is_continuous,
        distribution_cfg=cfg.distribution,
        num_envs=cfg.env.num_envs,
        screen_size=cfg.env.screen_size,
        device=fabric.device,
    )
    if agent_state:
        agent.load_state_dict(agent_state)

    # Setup player agent
    player = RecurrentPPOPlayer(
        copy.deepcopy(agent.feature_extractor),
        copy.deepcopy(agent.rnn),
        copy.deepcopy(agent.actor),
        copy.deepcopy(agent.critic),
        cfg.algo.rnn.lstm.hidden_size,
        actions_dim,
    )

    # Setup training agent
    agent.feature_extractor = fabric.setup_module(agent.feature_extractor)
    agent.rnn = fabric.setup_module(agent.rnn)
    agent.critic = fabric.setup_module(agent.critic)
    agent.actor = fabric.setup_module(agent.actor)

    # Setup player agent
    fabric_player = get_single_device_fabric(fabric)
    player.feature_extractor = fabric_player.setup_module(player.feature_extractor)
    player.rnn = fabric_player.setup_module(player.rnn)
    player.critic = fabric_player.setup_module(player.critic)
    player.actor = fabric_player.setup_module(player.actor)

    # Tie weights between the agent and the player
    for agent_p, player_p in zip(agent.feature_extractor.parameters(), player.feature_extractor.parameters()):
        player_p.data = agent_p.data
    for agent_p, player_p in zip(agent.rnn.parameters(), player.rnn.parameters()):
        player_p.data = agent_p.data
    for agent_p, player_p in zip(agent.actor.parameters(), player.actor.parameters()):
        player_p.data = agent_p.data
    for agent_p, player_p in zip(agent.critic.parameters(), player.critic.parameters()):
        player_p.data = agent_p.data
    return agent, player
