from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

import gymnasium as gym
import hydra
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor
from torch.distributions import Independent, OneHotCategoricalStraightThrough

from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/post_entropy",
    "State/prior_entropy",
    "State/kl",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic"}


def compute_stochastic_state(logits: Tensor, discrete: int = 32, sample=True) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        sample (bool): whether or not to sample the stochastic state.
            Default to True.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
    stochastic_state = dist.rsample() if sample else dist.mode
    return stochastic_state


def init_weights(m: nn.Module, mode: str = "uniform"):
    """
    Initialize the parameters of the m module acording to the Xavier method, by default the uniform one (the default
    initializer of the layers of Keras, which the official implementation uses), with zero biases.

    Args:
        m (nn.Module): the module to be initialized.
        mode (str): `uniform`, `normal` or `zero`. Default: `uniform`.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if mode == "normal":
            nn.init.xavier_normal_(m.weight.data)
        elif mode == "uniform":
            nn.init.xavier_uniform_(m.weight.data)
        elif mode == "zero":
            nn.init.constant_(m.weight.data, 0)
        else:
            raise RuntimeError(f"Unrecognized initialization: {mode}. Choose between: `normal`, `uniform` and `zero`")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def build_optimizer(optimizer_cfg: Dict[str, Any], params: Any) -> torch.optim.Optimizer:
    """The optimizer of `optimizer_cfg` for `params`, with the weight decay of DreamerV2: before every step the weights
    are multiplied by `1 - weight_decay`, whatever the learning rate (`Optimizer._apply_weight_decay` of
    https://github.com/danijar/dreamerv2). Adam is built as AdamW with the weight decay divided by the learning rate,
    which decays the weights that way: the weight decay of Adam is added to the gradients, where its normalization makes
    it negligible."""
    optimizer_cfg = dict(optimizer_cfg)
    weight_decay = optimizer_cfg.get("weight_decay") or 0
    if weight_decay > 0 and optimizer_cfg["_target_"] == "torch.optim.Adam":
        optimizer_cfg["_target_"] = "torch.optim.AdamW"
        optimizer_cfg["weight_decay"] = weight_decay / optimizer_cfg["lr"]
    return hydra.utils.instantiate(optimizer_cfg, params=params, _convert_="all")


# The most batches sampled (and moved to the device) at once: the first training can do many gradient steps
# (`algo.per_rank_pretrain_steps`)
MAX_SAMPLED_BATCHES = 16


def reinforce_weight(objective_mix: Optional[float], is_continuous: bool) -> float:
    """The weight of REINFORCE in the objective of the actor (`algo.actor.objective_mix`), the rest being the dynamics
    backpropagation. `None`: 0 for continuous actions and 1 for discrete ones, as DreamerV2 (`actor_grad: auto`) and
    DreamerV3 (`actor_grad_cont: backprop`, `actor_grad_disc: reinforce`) do."""
    if objective_mix is None:
        return 0.0 if is_continuous else 1.0
    return objective_mix


def actor_objective(
    objective_mix: Optional[float], is_continuous: bool, dynamics: Tensor, reinforce: Callable[[], Tensor]
) -> Tensor:
    """The objective of the DreamerV2 and DreamerV3 actors: `objective_mix` times the REINFORCE objective
    (`reinforce()`) plus `1 - objective_mix` times the dynamics backpropagation (`dynamics`, the lambda-values, or their
    advantages). `None` (the default of `algo.actor.objective_mix`): see `reinforce_weight`."""
    objective_mix = reinforce_weight(objective_mix, is_continuous)
    if objective_mix == 0:
        return dynamics
    if objective_mix == 1:
        return reinforce()
    return objective_mix * reinforce() + (1 - objective_mix) * dynamics


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    bootstrap: Optional[Tensor] = None,
    horizon: int = 15,
    lmbda: float = 0.95,
) -> Tensor:
    if bootstrap is None:
        bootstrap = torch.zeros_like(values[-1:])
    agg = bootstrap
    next_val = torch.cat((values[1:], bootstrap), dim=0)
    inputs = rewards + continues * next_val * (1 - lmbda)
    lv = []
    for i in reversed(range(horizon)):
        agg = inputs[i] + continues[i] * lmbda * agg
        lv.append(agg)
    return torch.cat(list(reversed(lv)), dim=0)


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    from sheeprl.algos.dreamer_v2.dreamer_v2 import DreamerV2
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    return log_trained_models(
        fabric,
        env,
        cfg,
        state,
        DreamerV2(fabric, cfg.to_log),
        lambda trained: {
            "world_model": trained.world_model,
            "actor": trained.actor,
            "critic": trained.critic,
            "target_critic": trained.target_critic,
        },
    )
