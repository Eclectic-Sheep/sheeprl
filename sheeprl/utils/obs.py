"""The observations of the environments as the agents read them: float tensors on the device of the agent, with the
stacked frames of the images as their channels."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import Tensor


def images_as_channels(obs: Dict[str, np.ndarray], cnn_keys: Sequence[str], num_envs: int) -> Dict[str, np.ndarray]:
    """`obs` with the stacked frames of the images `cnn_keys` as their channels."""
    return {k: v.reshape(num_envs, -1, *v.shape[-2:]) if k in cnn_keys else v for k, v in obs.items()}


def prepare_obs(
    device: str | torch.device,
    obs: Dict[str, np.ndarray],
    *,
    cnn_keys: Sequence[str] = (),
    num_envs: int = 1,
    time_dim: bool = False,
    image_shift: float = 0.5,
) -> Dict[str, Tensor]:
    """The observations `obs` of `num_envs` environments as float32 tensors on `device`, one row per environment
    (`[num_envs, ...]`, or `[1, num_envs, ...]` with `time_dim`, for the recurrent models): the images `cnn_keys` with
    their stacked frames as channels (`[..., C, H, W]`) scaled to `[-image_shift, 1 - image_shift]`, the other
    observations flattened."""
    batch_shape = (1, num_envs) if time_dim else (num_envs,)
    torch_obs = {}
    for k, v in obs.items():
        tensor = torch.from_numpy(v.copy()).to(device).float()
        if k in cnn_keys:
            tensor = tensor.reshape(*batch_shape, -1, *v.shape[-2:]) / 255
            if image_shift:
                tensor = tensor - image_shift
        else:
            tensor = tensor.reshape(*batch_shape, -1)
        torch_obs[k] = tensor
    return torch_obs


def normalize_obs(
    obs: Dict[str, np.ndarray | Tensor], cnn_keys: Sequence[str], obs_keys: Sequence[str]
) -> Dict[str, np.ndarray | Tensor]:
    """The observations `obs_keys` of `obs` with the images `cnn_keys` scaled to `[-0.5, 0.5]` (e.g. the ones of a
    batch of the training)."""
    return {k: obs[k] / 255 - 0.5 if k in cnn_keys else obs[k] for k in obs_keys}
