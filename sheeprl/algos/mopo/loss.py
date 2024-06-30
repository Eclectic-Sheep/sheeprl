from __future__ import annotations

from collections.abc import Sequence

import torch
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor


def world_model_loss(
    mean: Tensor,
    logvar: Tensor,
    targets: Tensor,
    weight_decays: Sequence[float] | None = None,
    ensembles: _FabricModule | None = None,
    min_logvar: Tensor | None = None,
    max_logvar: Tensor | None = None,
    use_logvar: bool = True,
) -> Tensor:
    """World Model Loss.

    Args:
        mean: The predicted mean.
        logvar: The predicted logvar.
        targets: The target tensor (concatenation of reward and the difference between the next obs and the obs).
        weight_decays: The sequence of weight decays to apply to the ensembles weights.
        ensembles: The esembles to which the decays apply.
        min_logvar: The min logvar from the ensembles (only during training).
        max_logvar: The max logvar from the ensembles (only during training).
        use_logvar: Whether or not to add the mean of logvars to the loss.
            Default to True.

    Returns the world model loss (num_ensembles,).
    """
    inv_var = torch.exp(-logvar) if use_logvar else torch.ones_like(logvar)
    losses = torch.mean(torch.mean(torch.square(mean - targets) * inv_var, dim=-1), dim=-1)
    if use_logvar:
        losses += torch.mean(torch.mean(logvar, dim=-1), dim=-1)
    if weight_decays is not None and ensembles is not None:
        decay_loss = torch.stack(
            [
                torch.sum(
                    torch.stack(
                        [
                            decay * torch.sum(torch.square(weight)) / 2
                            for decay, weight in zip(weight_decays, model_weights, strict=True)
                        ],
                        dim=0,
                    )
                )
                for model_weights in ensembles.get_linear_weights()
            ],
            dim=0,
        )
        losses += decay_loss
    if min_logvar is not None and max_logvar is not None:
        return torch.sum(losses) + 0.01 * torch.sum(max_logvar) - 0.01 * torch.sum(min_logvar)
    return losses
