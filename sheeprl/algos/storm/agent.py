from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, device
from torch.distributions.utils import probs_to_logits

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state
from sheeprl.algos.dreamer_v3.agent import CNNDecoder, CNNEncoder, MLPDecoder, MLPEncoder
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
from sheeprl.models.models import MLP, MultiDecoder, MultiEncoder


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    batch_size, batch_length = seq.shape[:2]
    subsequent_mask = (
        1 - torch.triu(torch.ones((1, batch_length, batch_length), device=seq.device), diagonal=1)
    ).bool()
    return subsequent_mask


def get_subsequent_mask_with_batch_length(batch_length, device):
    """For masking out the subsequent info."""
    subsequent_mask = (1 - torch.triu(torch.ones((1, batch_length, batch_length), device=device), diagonal=1)).bool()
    return subsequent_mask


def get_vector_mask(batch_length, device):
    mask = torch.ones((1, 1, batch_length), device=device).bool()
    return mask


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_heads: int = 8, d_model: int = 512, attn_dropout_p: float = 0.0, proj_dropout_p: float = 0.1):
        """
        Args:
            n_heads: number of heads
            d_model: dimension of the model
            attn_dropout_p: dropout probability for attention weights
            proj_dropout_p: dropout probability for the output tensor
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"`d_model` ({d_model}) must be divisible by `n_heads` ({n_heads})")

        self.n_heads = n_heads
        self.d_model = d_model
        self.attn_dropout_p = attn_dropout_p
        self.proj_dropout_p = proj_dropout_p
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.proj_dropout = nn.Dropout(proj_dropout_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, C = x.shape

        qkv = self.qkv(x)  # B x T x 3C
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)  # 3 x B x n_heads x T x d_head
        q, k, v = qkv.unbind(0)  # B x n_heads x T x d_head

        x = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=mask is None,
            )  # B x n_heads x T x d_head
            .transpose(1, 2)  # B x T x n_heads x d_head
            .reshape(B, T, C)
        )

        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout_p=0.1, activation: Type[torch.nn.Module] = nn.ReLU):
        """
        Args:
            d_in: the dimension of the input tensor
            d_hid: the dimension of the hidden layer
            dropout_p: dropout probability
            activation: activation function
        """
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = activation()

    def forward(self, x):
        x = self.w_2(self.activation(self.w_1(x)))
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        mlp_factor: int = 2,
        n_heads: int = 8,
        attn_dropout_p: float = 0.0,
        proj_dropout_p=0.1,
        ffn_dropout_p: float = 0.1,
        ffn_activation: Type[torch.nn.Module] = nn.ReLU,
        pre_norm: bool = False,
        norm_layer: Type[torch.nn.Module] = nn.LayerNorm,
        **norm_kwargs,
    ):
        """
        Args:
            d_model: dimension of the model
            mlp_factor: the multiplier of the hidden layer dimension in the pointwise feed-forward network
            n_heads: number of attention heads
            attn_dropout_p: dropout probability for attention weights
            proj_dropout_p: dropout probability for the MHSA output tensor
            ffn_dropout_p: dropout probability for the pointwise feed-forward network
            ffn_activation: activation function for the pointwise feed-forward network
            pre_norm: whether to use pre-norm or post-norm
            norm_layer: normalization layer
            norm_kwargs: keyword arguments for the normalization layer
        """
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(
            n_heads=n_heads,
            d_model=d_model,
            attn_dropout_p=attn_dropout_p,
            proj_dropout_p=proj_dropout_p,
        )
        self.ffn = PositionwiseFeedForward(
            d_in=d_model,
            d_hid=d_model * mlp_factor,
            dropout_p=ffn_dropout_p,
            activation=ffn_activation,
        )
        self.norm1 = norm_layer(d_model, **norm_kwargs)
        self.norm2 = norm_layer(d_model, **norm_kwargs)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.mhsa(self.norm1(x), mask=mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self.mhsa(x, mask=mask))
            x = self.norm2(x + self.ffn(x))
        return x


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_length: int, d_embd: int):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = d_embd
        self.pos_emb = nn.Embedding(self.max_length, d_embd)
        self.range = torch.arange(max_length)

    def forward(self, feat):
        pos_emb = self.pos_emb(self.range.to(feat.device))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])
        feat = feat + pos_emb[:, : feat.shape[1], :]
        return feat


class StochasticTransformer(nn.Module):
    def __init__(
        self,
        d_stoch: int,
        d_action: int,
        d_model: int = 512,
        mlp_factor: int = 2,
        n_layers: int = 2,
        n_heads: int = 8,
        max_length: int = 16,
        attn_dropout_p: float = 0.0,
        proj_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        pre_norm: bool = False,
        block_norm_layer: Type[torch.nn.Module] = nn.LayerNorm,
        **block_norm_norm_kwargs,
    ):
        """
        Args:
            d_stoch: dimension of the stochastic latent variable. Default is 32*32=1024
            d_action: dimension of the discrete action space
            d_model: dimension of the model
            mlp_factor: the multiplier of the hidden layer dimension in the pointwise feed-forward network
            n_layers: number of layers
            n_heads: number of attention heads
            max_length: the maximum length of the sequence
            attn_dropout_p: dropout probability for attention weights
            proj_dropout_p: dropout probability for the output tensor in the MHSA layer
            ffn_dropout_p: dropout probability for the pointwise feed-forward network
            pre_norm: whether to use pre-norm or post-norm
            block_norm_layer: normalization layer for the transformer block
            block_norm_norm_kwargs: keyword arguments for the normalization layer
        """
        super().__init__()
        self.d_stoch = d_stoch
        self.d_action = d_action
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.action_mixer = nn.Sequential(
            nn.Linear(d_stoch + d_action, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, d_embd=d_model)
        self.transformer = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    mlp_factor=mlp_factor,
                    n_heads=n_heads,
                    attn_dropout_p=attn_dropout_p,
                    proj_dropout_p=proj_dropout_p,
                    ffn_dropout_p=ffn_dropout_p,
                    pre_norm=pre_norm,
                    norm_layer=block_norm_layer,
                    **block_norm_norm_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, d_stoch)

    def forward(self, samples: torch.Tensor, action: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.action_mixer(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)

        for block in self.transformer:
            feats = block(feats, mask)

        feats = self.head(feats)
        return feats


class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (StochasticTransformer): the stochastic transformer model.
        representation_model (nn.Module): the representation model composed by a
            multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multi-layer perceptron to predict the stochastic part of the latent state.
        distribution_cfg (Dict[str, Any]): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over states, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`, where `unif = `1 / self.discrete`.
            Defaults to 0.01.
    """

    def __init__(
        self,
        recurrent_model: StochasticTransformer,
        representation_model: nn.Module,
        transition_model: nn.Module,
        distribution_cfg: Dict[str, Any],
        discrete: int = 32,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.discrete = discrete
        self.unimix = unimix
        self.distribution_cfg = distribution_cfg

    def dynamic(self, action: Tensor, posterior: Tensor, mask: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the prior from the recurrent output.
        For more information see [https://openreview.net/forum?id=WxnrX42rnS](https://openreview.net/forum?id=WxnrX42rnS).

        Args:
            posterior (Tensor): the stochastic state computed by the representation model (posterior). It is expected
                to be of dimension `[stoch_size, self.discrete]`, which by default is `[32, 32]`.
            action (Tensor): the action taken by the agent.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model.
            The prior stochastic state (Tensor): computed by the transition model
            The logits of the prior state (Tensor): computed by the transition model from the recurrent state.
        """
        posterior = posterior.view(*posterior.shape[:-2], -1)
        ht = self.recurrent_model(posterior, action, mask)
        prior_logits, prior = self._transition(ht)
        return ht, prior, prior_logits

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def _representation(self, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits: Tensor = self.representation_model(embedded_obs)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(
            logits, discrete=self.discrete, validate_args=self.distribution_cfg.validate_args
        )

    def _transition(self, recurrent_out: Tensor, sample_state=True) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.
            sampler_state (bool): whether or not to sample the stochastic state.
                Default to True

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits: Tensor = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(
            logits, discrete=self.discrete, sample=sample_state, validate_args=self.distribution_cfg.validate_args
        )

    def imagination(self, prior: Tensor, recurrent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One-step imagination of the next latent state.
        It can be used several times to imagine trajectories in the latent space (Transition Model).

        Args:
            prior (Tensor): the prior state.
            recurrent_state (Tensor): the recurrent state of the recurrent model.
            actions (Tensor): the actions taken by the agent.

        Returns:
            The imagined prior state (Tuple[Tensor, Tensor]): the imagined prior state.
            The recurrent state (Tensor).
        """
        raise NotImplementedError("The imagination method is not implemented yet.")


class Player(nn.Module):
    """
    The model of the Dreamer_v3 player.

    Args:
        encoder (_FabricModule): the encoder.
        recurrent_model (_FabricModule): the recurrent model.
        representation_model (_FabricModule): the representation model.
        actor (_FabricModule): the actor.
        actions_dim (Sequence[int]): the dimension of the actions.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        device (torch.device): the device to work on.
        transition_model (_FabricModule): the transition model.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
            Defaults to 32.
        actor_type (str, optional): which actor the player is using ('task' or 'exploration').
            Default to None.
    """

    def __init__(
        self,
        encoder: _FabricModule,
        rssm: RSSM,
        actor: _FabricModule,
        actions_dim: Sequence[int],
        num_envs: int,
        stochastic_size: int,
        recurrent_state_size: int,
        device: device = "cpu",
        discrete_size: int = 32,
        actor_type: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = RSSM(
            recurrent_model=rssm.recurrent_model.module,
            representation_model=rssm.representation_model.module,
            transition_model=rssm.transition_model.module,
            distribution_cfg=actor.distribution_cfg,
            discrete=rssm.discrete,
            unimix=rssm.unimix,
        )
        self.actor = actor
        self.device = device
        self.actions_dim = actions_dim
        self.stochastic_size = stochastic_size
        self.discrete_size = discrete_size
        self.recurrent_state_size = recurrent_state_size
        self.num_envs = num_envs
        self.validate_args = self.actor.distribution_cfg.validate_args
        self.actor_type = actor_type

    def get_exploration_action(
        self,
        posteriors: Tensor,
        actions: Tensor,
        action_mask: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Return the actions with a certain amount of noise for exploration.

        Args:
            obs (Dict[str, Tensor]): the current observations.
            action_mask (Dict[str, Tensor], optional): the mask of the actions.
                Default to None.

        Returns:
            The actions the agent has to perform.
        """
        actions = self.get_greedy_action(posteriors=posteriors, actions=actions, action_mask=action_mask)
        expl_actions = None
        if self.actor.expl_amount > 0:
            expl_actions = self.actor.add_exploration_noise(actions, mask=action_mask)
            self.actions = torch.cat(expl_actions, dim=-1)
        return expl_actions or actions

    def get_greedy_action(
        self,
        posteriors: Tensor,
        actions: Tensor,
        is_training: bool = True,
        action_mask: Tensor | None = None,
    ) -> Sequence[Tensor]:
        """
        Return the greedy actions.

        Args:
            obs (Dict[str, Tensor]): the current observations.
            is_training (bool): whether it is training.
                Default to True.

        Returns:
            The actions the agent has to perform.
        """
        posteriors = posteriors.view(*posteriors.shape[:-2], self.stochastic_size * self.discrete_size)
        temporal_mask = get_subsequent_mask(posteriors)
        ht = self.rssm.recurrent_model(posteriors, actions, temporal_mask)
        last_ht = ht[:, -1:, ...]
        _, prior = self.rssm._transition(last_ht)
        prior = prior.view(*prior.shape[:-2], self.stochastic_size * self.discrete_size)
        actions, _ = self.actor(torch.cat((prior, last_ht), -1), is_training, action_mask)
        return actions


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_state: Optional[Dict[str, Tensor]] = None,
    critic_state: Optional[Dict[str, Tensor]] = None,
    target_critic_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, torch.nn.Module]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        actions_dim (Sequence[int]): the dimension of the actions.
        is_continuous (bool): whether or not the actions are continuous.
        cfg (DictConfig): the configs of DreamerV3.
        obs_space (Dict[str, Any]): the observation space.
        world_model_state (Dict[str, Tensor], optional): the state of the world model.
            Default to None.
        actor_state: (Dict[str, Tensor], optional): the state of the actor.
            Default to None.
        critic_state: (Dict[str, Tensor], optional): the state of the critic.
            Default to None.
        target_critic_state: (Dict[str, Tensor], optional): the state of the critic.
            Default to None.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and
        reward models and the continue model.
        The actor (_FabricModule).
        The critic (_FabricModule).
        The target critic (nn.Module).
    """
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic

    # Sizes
    recurrent_state_size = world_model_cfg.recurrent_model.d_model
    stochastic_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    stochastic_size + recurrent_state_size

    # Define models
    cnn_stages = int(np.log2(cfg.env.screen_size) - np.log2(4))
    cnn_encoder = (
        CNNEncoder(
            keys=cfg.algo.cnn_keys.encoder,
            input_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cfg.algo.cnn_keys.encoder],
            image_size=obs_space[cfg.algo.cnn_keys.encoder[0]].shape[-2:],
            channels_multiplier=world_model_cfg.encoder.cnn_channels_multiplier,
            layer_norm=world_model_cfg.encoder.layer_norm,
            activation=eval(world_model_cfg.encoder.cnn_act),
            stages=cnn_stages,
        )
        if cfg.algo.cnn_keys.encoder is not None and len(cfg.algo.cnn_keys.encoder) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(
            keys=cfg.algo.mlp_keys.encoder,
            input_dims=[obs_space[k].shape[0] for k in cfg.algo.mlp_keys.encoder],
            mlp_layers=world_model_cfg.encoder.mlp_layers,
            dense_units=world_model_cfg.encoder.dense_units,
            activation=eval(world_model_cfg.encoder.dense_act),
            layer_norm=world_model_cfg.encoder.layer_norm,
        )
        if cfg.algo.mlp_keys.encoder is not None and len(cfg.algo.mlp_keys.encoder) > 0
        else None
    )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder)
    recurrent_model = StochasticTransformer(
        d_stoch=stochastic_size, d_action=int(sum(actions_dim)), **world_model_cfg.recurrent_model
    )
    representation_model = MLP(
        input_dims=encoder.output_dim,
        output_dim=stochastic_size,
        hidden_sizes=(),
        activation=None,
        layer_args={"bias": not world_model_cfg.representation_model.layer_norm},
        flatten_dim=None,
        norm_layer=[nn.LayerNorm] if world_model_cfg.representation_model.layer_norm else None,
        norm_args=(
            [{"normalized_shape": world_model_cfg.representation_model.hidden_size}]
            if world_model_cfg.representation_model.layer_norm
            else None
        ),
    )
    transition_model = MLP(
        input_dims=stochastic_size,
        output_dim=stochastic_size,
        hidden_sizes=(),
        activation=None,
        layer_args={"bias": not world_model_cfg.transition_model.layer_norm},
        flatten_dim=None,
        norm_layer=[nn.LayerNorm] if world_model_cfg.transition_model.layer_norm else None,
        norm_args=(
            [{"normalized_shape": world_model_cfg.transition_model.hidden_size}]
            if world_model_cfg.transition_model.layer_norm
            else None
        ),
    )
    rssm = RSSM(
        recurrent_model=recurrent_model.apply(init_weights),
        representation_model=representation_model.apply(init_weights),
        transition_model=transition_model.apply(init_weights),
        distribution_cfg=cfg.distribution,
        discrete=world_model_cfg.discrete_size,
        unimix=cfg.algo.unimix,
    )
    cnn_decoder = (
        CNNDecoder(
            keys=cfg.algo.cnn_keys.decoder,
            output_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cfg.algo.cnn_keys.decoder],
            channels_multiplier=world_model_cfg.observation_model.cnn_channels_multiplier,
            latent_state_size=stochastic_size,
            cnn_encoder_output_dim=cnn_encoder.output_dim,
            image_size=obs_space[cfg.algo.cnn_keys.decoder[0]].shape[-2:],
            activation=eval(world_model_cfg.observation_model.cnn_act),
            layer_norm=world_model_cfg.observation_model.layer_norm,
            stages=cnn_stages,
        )
        if cfg.algo.cnn_keys.decoder is not None and len(cfg.algo.cnn_keys.decoder) > 0
        else None
    )
    mlp_decoder = (
        MLPDecoder(
            keys=cfg.algo.mlp_keys.decoder,
            output_dims=[obs_space[k].shape[0] for k in cfg.algo.mlp_keys.decoder],
            latent_state_size=stochastic_size,
            mlp_layers=world_model_cfg.observation_model.mlp_layers,
            dense_units=world_model_cfg.observation_model.dense_units,
            activation=eval(world_model_cfg.observation_model.dense_act),
            layer_norm=world_model_cfg.observation_model.layer_norm,
        )
        if cfg.algo.mlp_keys.decoder is not None and len(cfg.algo.mlp_keys.decoder) > 0
        else None
    )
    observation_model = MultiDecoder(cnn_decoder, mlp_decoder)
    reward_model = MLP(
        input_dims=stochastic_size,
        output_dim=world_model_cfg.reward_model.bins,
        hidden_sizes=[world_model_cfg.reward_model.dense_units] * world_model_cfg.reward_model.mlp_layers,
        activation=eval(world_model_cfg.reward_model.dense_act),
        layer_args={"bias": not world_model_cfg.reward_model.layer_norm},
        flatten_dim=None,
        norm_layer=(
            [nn.LayerNorm for _ in range(world_model_cfg.reward_model.mlp_layers)]
            if world_model_cfg.reward_model.layer_norm
            else None
        ),
        norm_args=(
            [
                {"normalized_shape": world_model_cfg.reward_model.dense_units}
                for _ in range(world_model_cfg.reward_model.mlp_layers)
            ]
            if world_model_cfg.reward_model.layer_norm
            else None
        ),
    )
    continue_model = MLP(
        input_dims=stochastic_size,
        output_dim=1,
        hidden_sizes=[world_model_cfg.discount_model.dense_units] * world_model_cfg.discount_model.mlp_layers,
        activation=eval(world_model_cfg.discount_model.dense_act),
        layer_args={"bias": not world_model_cfg.discount_model.layer_norm},
        flatten_dim=None,
        norm_layer=(
            [nn.LayerNorm for _ in range(world_model_cfg.discount_model.mlp_layers)]
            if world_model_cfg.discount_model.layer_norm
            else None
        ),
        norm_args=(
            [
                {"normalized_shape": world_model_cfg.discount_model.dense_units}
                for _ in range(world_model_cfg.discount_model.mlp_layers)
            ]
            if world_model_cfg.discount_model.layer_norm
            else None
        ),
    )
    world_model = WorldModel(
        encoder.apply(init_weights),
        rssm,
        observation_model.apply(init_weights),
        reward_model.apply(init_weights),
        continue_model.apply(init_weights),
    )
    actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)
    actor: nn.Module = actor_cls(
        latent_state_size=stochastic_size * 2,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        dense_units=actor_cfg.dense_units,
        activation=eval(actor_cfg.dense_act),
        mlp_layers=actor_cfg.mlp_layers,
        distribution_cfg=cfg.distribution,
        layer_norm=actor_cfg.layer_norm,
        unimix=cfg.algo.unimix,
    )
    critic = MLP(
        input_dims=stochastic_size,
        output_dim=critic_cfg.bins,
        hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
        activation=eval(critic_cfg.dense_act),
        layer_args={"bias": not critic_cfg.layer_norm},
        flatten_dim=None,
        norm_layer=[nn.LayerNorm for _ in range(critic_cfg.mlp_layers)] if critic_cfg.layer_norm else None,
        norm_args=(
            [{"normalized_shape": critic_cfg.dense_units} for _ in range(critic_cfg.mlp_layers)]
            if critic_cfg.layer_norm
            else None
        ),
    )
    actor.apply(init_weights)
    critic.apply(init_weights)

    if cfg.algo.hafner_initialization:
        actor.mlp_heads.apply(uniform_init_weights(1.0))
        critic.model[-1].apply(uniform_init_weights(0.0))
        rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
        rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
        world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
        world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
        if mlp_decoder is not None:
            mlp_decoder.heads.apply(uniform_init_weights(1.0))
        if cnn_decoder is not None:
            cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

    # Load models from checkpoint
    if world_model_state:
        world_model.load_state_dict(world_model_state)
    if actor_state:
        actor.load_state_dict(actor_state)
    if critic_state:
        critic.load_state_dict(critic_state)

    # Setup models with Fabric
    world_model.encoder = fabric.setup_module(world_model.encoder)
    world_model.observation_model = fabric.setup_module(world_model.observation_model)
    world_model.reward_model = fabric.setup_module(world_model.reward_model)
    world_model.rssm.recurrent_model = fabric.setup_module(world_model.rssm.recurrent_model)
    world_model.rssm.representation_model = fabric.setup_module(world_model.rssm.representation_model)
    world_model.rssm.transition_model = fabric.setup_module(world_model.rssm.transition_model)
    if world_model.continue_model:
        world_model.continue_model = fabric.setup_module(world_model.continue_model)
    actor = fabric.setup_module(actor)
    critic = fabric.setup_module(critic)
    target_critic = copy.deepcopy(critic.module)
    if target_critic_state:
        target_critic.load_state_dict(target_critic_state)

    return world_model, actor, critic, target_critic
