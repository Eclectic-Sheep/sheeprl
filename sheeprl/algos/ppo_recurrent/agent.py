from math import prod, sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor
from torch.distributions import Independent, Normal, OneHotCategorical

from sheeprl.algos.ppo.agent import CNNEncoder, MLPEncoder, PPOActor, ortho_init_linear_layers
from sheeprl.algos.ppo_recurrent.utils import prepare_obs
from sheeprl.core.collector import Act, Policy
from sheeprl.models.models import MLP, MultiEncoder
from sheeprl.utils.compile import compiled_policy
from sheeprl.utils.fabric import get_single_device_fabric, setup_module


def lstm_unroll(
    x: Tensor,
    states: Tuple[Tensor, Tensor],
    weight_ih: Tensor,
    bias_ih: Tensor,
    weight_hh: Tensor,
    bias_hh: Tensor,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """A single-layer `nn.LSTM` on the sequences `x` (time first) from `states`, computed from its weights with the
    equations of PyTorch (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html): the same values up to the
    rounding."""
    h, c = states[0][0], states[1][0]
    # The inputs of all the steps at once
    x = torch.nn.functional.linear(x, weight_ih, bias_ih)
    outputs = []
    for t in range(x.shape[0]):
        i, f, g, o = (x[t] + torch.nn.functional.linear(h, weight_hh, bias_hh)).chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        outputs.append(h)
    return torch.stack(outputs), (h.unsqueeze(0), c.unsqueeze(0))


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
                activation=hydra.utils.get_class(pre_rnn_mlp_cfg.activation),
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
        # The weights of the LSTM, for the compiled loss: `torch.compile` doesn't trace the code that reaches `nn.LSTM`
        # (a tuple: they stay the weights of the LSTM, also in the state dict)
        self._lstm_weights = (
            self._lstm.weight_ih_l0,
            self._lstm.bias_ih_l0,
            self._lstm.weight_hh_l0,
            self._lstm.bias_hh_l0,
        )
        if post_rnn_mlp_cfg.apply:
            self._post_mlp = MLP(
                input_dims=lstm_hidden_size,
                output_dim=None,
                hidden_sizes=[post_rnn_mlp_cfg.dense_units],
                activation=hydra.utils.get_class(post_rnn_mlp_cfg.activation),
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
        """The outputs of the LSTM at every step of the sequences and its last states.

        The padded steps of the sequences (`mask` False) follow their valid ones: the LSTM runs on them too, which
        doesn't change the outputs of the valid steps, without packing the sequences (it read their lengths on the
        host). The returned states are the ones after the last step, padded or not."""
        x = self._pre_mlp(input)
        if torch.compiler.is_compiling():
            # `torch.compile` doesn't trace `nn.LSTM`: the same steps, from its weights
            out, states = lstm_unroll(x, states, *self._lstm_weights)
        else:
            self._lstm.flatten_parameters()
            out, states = self._lstm(x, states)
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
                hydra.utils.get_class(encoder_cfg.dense_act),
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
            activation=hydra.utils.get_class(critic_cfg.dense_act),
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
            activation=hydra.utils.get_class(actor_cfg.dense_act),
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
        # Orthogonal initialization, as PPO's agent: the encoder with gain 1, the hidden layers of actor and critic with
        # gain sqrt(2), the actor's heads with 0.01 and the critic's output with 1 (the LSTM keeps its own)
        if encoder_cfg.ortho_init:
            ortho_init_linear_layers(self.feature_extractor, gain=1.0)
        if actor_cfg.ortho_init:
            ortho_init_linear_layers(actor_backbone, gain=sqrt(2))
            ortho_init_linear_layers(actor_heads, gain=0.01)
        if critic_cfg.ortho_init:
            ortho_init_linear_layers(self.critic, gain=sqrt(2), output_gain=1.0)

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
            logprobs.append(dist.log_prob(sampled_actions[-1]))
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


class RecurrentPPOPolicy(nn.Module, Policy):
    """The policy of recurrent PPO: `forward` samples the actions from the observations, the previous actions and the
    previous recurrent states as tensors; `act` plays them in the environments (`Policy`), from the observations
    `obs_keys` (the images among them are `cnn_keys`) moved to the device of `fabric`, and keeps the recurrent state and
    the previous actions of every environment. When an episode ends, its previous actions are reset, and its recurrent
    state too with `reset_recurrent_state_on_done`."""

    def __init__(
        self,
        feature_extractor: MultiEncoder,
        rnn: RecurrentModel,
        actor: PPOActor,
        critic: nn.Module,
        rnn_hidden_size: int,
        actions_dim: Sequence[int],
        fabric: Optional[Fabric] = None,
        obs_keys: Sequence[str] = (),
        cnn_keys: Sequence[str] = (),
        reset_recurrent_state_on_done: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.rnn = rnn
        self.critic = critic
        self.actor = actor
        self.rnn_hidden_size = rnn_hidden_size
        self.actions_dim = actions_dim
        self.fabric = fabric
        self.obs_keys = obs_keys
        self.cnn_keys = cnn_keys
        self.reset_recurrent_state_on_done = reset_recurrent_state_on_done
        # The recurrent states and the actions preceding the next step of every environment, created at the first step
        self.prev_states: Optional[Tuple[Tensor, Tensor]] = None
        self.prev_actions: Optional[np.ndarray] = None

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
            logprobs.append(dist.log_prob(sampled_actions[-1]))
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

    def act(self, obs: Dict[str, np.ndarray]) -> Act:
        """The actions for `obs`, with their columns: the actions one-hot for discrete actions (the environments take
        their indices), their log-probabilities, the values of `obs`, and the recurrent states (`prev_hx`, `prev_cx`)
        and the actions (`prev_actions`) that preceded them. Its extras are the actions and the recurrent states as
        tensors, which bootstrap the returns."""
        num_envs = len(obs[self.obs_keys[0]])
        device = self.fabric.device
        if self.prev_states is None:
            self.prev_states = (
                torch.zeros(1, num_envs, self.rnn_hidden_size, device=device),
                torch.zeros(1, num_envs, self.rnn_hidden_size, device=device),
            )
            self.prev_actions = np.zeros((1, num_envs, sum(self.actions_dim)))
        torch_obs = prepare_obs(
            self.fabric, {k: obs[k] for k in self.obs_keys}, cnn_keys=self.cnn_keys, num_envs=num_envs
        )
        torch_prev_actions = torch.from_numpy(self.prev_actions).to(device).float()
        actions, logprobs, values, states = self(
            torch_obs, prev_actions=torch_prev_actions, prev_states=self.prev_states
        )
        if self.actor.is_continuous:
            env_actions = torch.stack(actions, -1).cpu().numpy()
        else:
            env_actions = torch.stack([act.argmax(dim=-1) for act in actions], dim=-1).cpu().numpy()
        torch_actions = torch.cat(actions, dim=-1)
        columns = {
            "actions": torch_actions.cpu().numpy(),
            "logprobs": logprobs.cpu().numpy(),
            "values": values.cpu().numpy(),
            "prev_hx": self.prev_states[0].cpu().numpy(),
            "prev_cx": self.prev_states[1].cpu().numpy(),
            "prev_actions": self.prev_actions,
        }
        self.prev_actions, self.prev_states = columns["actions"], states
        return Act(env_actions, columns, {"actions": torch_actions, "states": states})

    def reset_state(self, env_idxes: Optional[Sequence[int]] = None) -> None:
        if env_idxes is None:
            # Created with zeros at the next step
            self.prev_states = self.prev_actions = None
            return
        # Multiplied by the mask of the episodes that go on, as the rollout resets them in the training
        dones = np.zeros((1, self.prev_actions.shape[1], 1), dtype=np.float32)
        dones[:, env_idxes] = 1
        self.prev_actions = (1 - dones) * self.prev_actions
        if self.reset_recurrent_state_on_done:
            self.prev_states = tuple(
                (1 - torch.as_tensor(dones, device=self.fabric.device)) * s for s in self.prev_states
            )

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
) -> Tuple[RecurrentPPOAgent, RecurrentPPOPolicy]:
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

    # Setup training agent
    agent.feature_extractor = setup_module(fabric, agent.feature_extractor)
    agent.rnn = setup_module(fabric, agent.rnn)
    agent.critic = setup_module(fabric, agent.critic)
    agent.actor = setup_module(fabric, agent.actor)

    # Setup policy agent: it plays with the modules of the agent, without the wrappers of the distributed training. A
    # copy with the weights tied lost them on CUDA, where the LSTM moves its weights into a new buffer at every forward
    # (`flatten_parameters`): the policy played with the initial weights for the whole training
    fabric_player = get_single_device_fabric(fabric)
    policy = RecurrentPPOPolicy(
        fabric_player.setup_module(agent.feature_extractor.module),
        fabric_player.setup_module(agent.rnn.module),
        fabric_player.setup_module(agent.actor.module),
        fabric_player.setup_module(agent.critic.module),
        cfg.algo.rnn.lstm.hidden_size,
        actions_dim,
        fabric=fabric_player,
        obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
        cnn_keys=cfg.algo.cnn_keys.encoder,
        reset_recurrent_state_on_done=cfg.algo.reset_recurrent_state_on_done,
    )
    # The step of the policy, compiled with `algo.compile` (`compiled_policy`)
    policy.forward = compiled_policy(policy.forward, fabric, cfg)
    return agent, policy
