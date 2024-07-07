from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from math import prod
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn

from sheeprl.algos.mopo.loss import world_model_loss
from sheeprl.algos.sac.agent import SACAgent, SACPlayer
from sheeprl.algos.sac.agent import build_agent as build_agent_sac
from sheeprl.models.models import MLP
from sheeprl.utils.model import ModuleType
from sheeprl.utils.scaler import TensorStandardScaler
from sheeprl.utils.utils import dotdict


class Ensembles(nn.Module):
    """Ensemble model."""

    def __init__(
        self,
        input_dims: int,
        output_dim: int,
        hidden_sizes: Sequence[str],
        penalty_coef: float,
        activation: ModuleType = nn.SiLU,
        num_ensembles: int = 7,
        num_elites: int = 5,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self._models = nn.ModuleList(
            [MLP(input_dims, output_dim, hidden_sizes, activation=activation) for _ in range(num_ensembles)]
        )
        self.min_logvar = nn.Parameter(-10 * torch.ones((1, output_dim // 2)))
        self.max_logvar = nn.Parameter(0.5 * torch.ones((1, output_dim // 2)))
        self.scaler = TensorStandardScaler(input_dims, device=device)
        self.penalty_coef = penalty_coef
        self.num_elites = num_elites
        self.elites_idxes: Tensor | None = None

    @cached_property
    def num_ensembles(self) -> int:
        """Get the number of models in the ensemble."""
        return len(self._models)

    def forward(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Perform forward method of nn.Module.

        Args:
            obs: The observations of shape (num_ensembles, batch_size, obs_dim)
                or (batch_size, obs_dim). In the latter case, the inputs are repeated
                for each model of the ensembles.
            actions: The actions of shape (num_ensembles, batch_size, actions_dim).

        Returns the mean and logvar predicted by the model.
            The shape is (num_ensembles, batch_size, 2 * (obs_dim + rewards_dim))
        """
        x = self.scaler.transform(torch.cat((obs, actions), dim=-1))
        if len(x.shape) == 2:
            x = x.repeat(self.num_ensembles, 1, 1)
        outputs = []
        for inp, model in zip(x, self._models, strict=True):
            outputs.append(model(inp))

        mean, logvar = torch.stack(outputs, dim=0).chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    @torch.inference_mode()
    def validate(self, mean: Tensor, logvar: Tensor, targets: Tensor, update_elites: bool = False) -> Tensor:
        """Run validation loss for selecting the best `num_elites` models.

        Args:
            mean: The mean computed by the ensembles on the validation set (validation_set_dim, obs_shape).
            logvar: The logvar computed by the ensembles on the validation set (validation_set_dim, actions_shape).
            targets: The targets of the validation set (validation_set_dim, target_shape).
            update_elites: Whether or not to select the best `num_elites` models.

        Returns the validation loss (one for each model in the ensembles).
        """
        losses = world_model_loss(mean, logvar, targets, use_logvar=False)
        if update_elites:
            sorted_idxes = torch.argsort(losses, dim=0)
            self.elites_idxes = sorted_idxes[: self.num_elites]
        return losses

    @torch.inference_mode()
    def predict(self, mean: Tensor, logvar: Tensor, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Predict the next observations and rewards.

        Args:
            mean: The mean computed by the ensembles on the validation set (batch_shape, obs_shape).
            logvar: The logvar computed by the ensembles on the validation set (batch_shape, actions_shape).
            obs: The observations of shape (batch_shape, obs_shape).

        Returns three tensors:
            next_observations: The next observations.
            rewards: The penalized rewards.
        """
        logvar = torch.exp(logvar)

        # Add obs to the mean (the ensembles learn the difference between the next_obs and the obs)
        mean[:, :, 1:] = mean[:, :, 1:] + obs
        std = torch.sqrt(logvar)

        # Sample from predicted distributions and select one ensemble foreach element in the batch
        ensembles_samples = mean + torch.normal(torch.zeros_like(mean), torch.ones_like(mean)) * std
        if self.elites_idxes is not None:
            ensembles_idxes = self.elites_idxes[torch.randint(0, self.num_elites, size=(mean.shape[1],))]
        else:
            ensembles_idxes = torch.randint(0, self.num_ensembles, size=(mean.shape[1],))
        batch_idxes = torch.arange(mean.shape[1])
        samples = ensembles_samples[ensembles_idxes, batch_idxes]

        # Compute penalized rewards and split from next_observations
        rewards, next_observations = samples[:, :1], samples[:, 1:]
        penalty = torch.amax(torch.linalg.norm(std, dim=2, keepdim=True), dim=0)
        rewards = rewards - self.penalty_coef * penalty
        return next_observations, rewards

    def get_linear_weights(self):
        """Return the weights of the linear layers, divided for ensemble model."""
        return [
            [model.weight for model in ensemble_model.model if isinstance(model, nn.Linear)]
            for ensemble_model in self._models
        ]


def build_agent(
    fabric: Fabric,
    cfg: dotdict[str, Any],
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    state: dict[str, Any] | None = None,
) -> tuple[_FabricModule, SACAgent, SACPlayer]:
    """Instantiate the ensemble's models and the SAC agent used for training and evaluation."""
    if not isinstance(obs_space, gym.spaces.Dict):
        raise RuntimeError(f"The observation space must be a Dict, got {type(obs_space)}")
    if not isinstance(action_space, gym.spaces.Box | gym.spaces.Discrete | gym.spaces.MultiDiscrete):
        raise RuntimeError(f"The observation space must be a Box or Discrete or MultiDiscrete, got {type(obs_space)}")

    action_shape = tuple(
        action_space.shape
        if isinstance(action_space, gym.spaces.Box)
        else (action_space.nvec.tolist() if isinstance(action_space, gym.spaces.MultiDiscrete) else [action_space.n])
    )
    act_dim = prod(action_shape)
    obs_dim = np.sum([np.prod(obs_space[k].shape or ()) for k in cfg.algo.mlp_keys.encoder], dtype=np.int32).item()
    # Assume the reward has only one dimension
    ensembles = Ensembles(
        input_dims=act_dim + obs_dim,
        output_dim=(obs_dim + 1) * 2,  # mean and std
        hidden_sizes=[cfg.algo.ensembles.dense_units] * cfg.algo.ensembles.mlp_layers,
        penalty_coef=cfg.algo.ensembles.penalty_coef,
        activation=hydra.utils.get_class(cfg.algo.ensembles.dense_act),
        num_ensembles=cfg.algo.ensembles.num_ensembles,
        num_elites=cfg.algo.ensembles.num_elites,
    ).apply(init_weights)

    if state is not None:
        ensembles.load_state_dict(state["ensembles"])

    ensembles = fabric.setup_module(ensembles)

    sac_agent, sac_player = build_agent_sac(fabric, cfg, obs_space, action_space, (state or {}).get("agent", None))
    return ensembles, sac_agent, sac_player


def init_weights(m):
    """Truncated Normal initialization."""
    if isinstance(m, nn.Linear):
        std = 1 / (2 * np.sqrt(m.in_features))
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
