from __future__ import annotations

import copy
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import gymnasium as gym
import mlflow
import rich.syntax
import rich.tree
import torch
import torch.nn as nn
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from mlflow.models.model import ModelInfo
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

from sheeprl.utils.env import make_env
from sheeprl.utils.model_manager import MlflowModelManager


class dotdict(dict):
    """
    A dictionary supporting dot notation.
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = dotdict(v)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)


@torch.no_grad()
def gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    num_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """Compute returns and advantages following https://arxiv.org/abs/1506.02438

    Args:
        rewards (Tensor): all rewards collected from the last rollout
        values (Tensor): all values collected from the last rollout
        dones (Tensor): all dones collected from the last rollout
        next_values (Tensor): estimated values for the next observations
        num_steps (int): the number of steps played
        gamma (float): discout factor
        gae_lambda (float): lambda for GAE estimation

    Returns:
        estimated returns
        estimated advantages
    """
    lastgaelam = 0
    nextvalues = next_value
    not_dones = torch.logical_not(dones)
    nextnonterminal = not_dones[-1]
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(num_steps)):
        if t < num_steps - 1:
            nextnonterminal = not_dones[t]
            nextvalues = values[t + 1]
        delta = rewards[t] + nextvalues * nextnonterminal * gamma - values[t]
        advantages[t] = lastgaelam = delta + nextnonterminal * lastgaelam * gamma * gae_lambda
    returns = advantages + values
    return returns, advantages


def init_weights(m: nn.Module):
    """
    Initialize the parameters of the m module acording to the method described in
    [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852) using a uniform distribution.

    Args:
        m (nn.Module): the module to be initialized.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def normalize_tensor(tensor: Tensor, eps: float = 1e-8, mask: Optional[Tensor] = None) -> Tensor:
    unmasked = mask is None
    if unmasked:
        mask = torch.ones_like(tensor, dtype=torch.bool)
    masked_tensor = tensor[mask]
    normalized = (masked_tensor - masked_tensor.mean()) / (masked_tensor.std() + eps)
    if unmasked:
        return normalized.reshape_as(mask)
    else:
        return normalized


def polynomial_decay(
    current_step: int,
    *,
    initial: float = 1.0,
    final: float = 0.0,
    max_decay_steps: int = 100,
    power: float = 1.0,
) -> float:
    if current_step > max_decay_steps or initial == final:
        return final
    else:
        return (initial - final) * ((1 - current_step / max_decay_steps) ** power) + final


# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = ("algo", "buffer", "checkpoint", "env", "fabric", "metric"),
    resolve: bool = True,
    cfg_save_path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config: Configuration composed by Hydra.
        fields: Determines which main fields from config will
            be printed and in what order.
        resolve: Whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    if cfg_save_path is not None:
        with open(os.path.join(os.getcwd(), "config_tree.txt"), "w") as fp:
            rich.print(tree, file=fp)


def unwrap_fabric(model: _FabricModule | nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    if isinstance(model, _FabricModule):
        model = model.module
    for name, child in model.named_children():
        setattr(model, name, unwrap_fabric(child))
    return model


def register_model(fabric: Fabric, log_models: Callable[[str], Dict[str, ModelInfo]], cfg: Dict[str, Any]):
    tracking_uri = getattr(fabric.logger, "_tracking_uri", None) or os.getenv("MLFLOW_TRACKING_URI", None)
    if tracking_uri is None:
        raise ValueError(
            "The tracking uri is not defined, use an mlflow logger with a tracking uri or define the "
            "MLFLOW_TRACKING_URI environment variable."
        )
    cfg_model_manager = cfg.model_manager
    # Retrieve run_id, if None, create a new run
    run_id = None
    if len(fabric.loggers) > 0:
        run_id = getattr(fabric.logger, "run_id", None)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = None
    run_name = None
    if run_id is None:
        experiment = mlflow.get_experiment_by_name(cfg.exp_name)
        experiment_id = mlflow.create_experiment(cfg.exp_name) if experiment is None else experiment.experiment_id
        run_name = f"{cfg.algo.name}_{cfg.env.id}_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
    models_info = log_models(run_id, experiment_id, run_name)
    model_manager = MlflowModelManager(fabric, tracking_uri)
    if len(models_info) != len(cfg_model_manager.models):
        raise RuntimeError(
            f"The number of models of the {cfg.algo.name} agent must be equal to the number "
            f"of models you want to register. {len(cfg_model_manager.models)} model registration "
            f"configs are given, but the agent has {len(models_info)} models."
        )
    for k, cfg_model in cfg_model_manager.models.items():
        model_manager.register_model(
            models_info[k]._model_uri, cfg_model["model_name"], cfg_model["description"], cfg_model["tags"]
        )


def register_model_from_checkpoint(
    fabric: Fabric,
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    log_models_from_checkpoint: Callable[
        [Fabric, gym.Env | gym.Wrapper, Dict[str, Any], Dict[str, Any]], Dict[str, ModelInfo]
    ],
):
    tracking_uri = getattr(cfg, "tracking_uri", None) or os.getenv("MLFLOW_TRACKING_URI", None)
    if tracking_uri is None:
        raise ValueError(
            "The tracking uri is not defined, use an mlflow logger with a tracking uri or define the "
            "MLFLOW_TRACKING_URI environment variable."
        )
    # Creating the environment for agent instantiation
    env = make_env(
        cfg,
        cfg.seed,
        0,
        None,
        "test",
        vector_env_idx=0,
    )()
    observation_space = env.observation_space
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: "
            "`cnn_keys.encoder=[rgb]` or `mlp_keys.encoder=[state]`"
        )

    mlflow.set_tracking_uri(tracking_uri)
    # If the user does not specify the experiment, than, create a new experiment
    if cfg.run.id is None and cfg.experiment.id is None:
        cfg.experiment.id = mlflow.create_experiment(cfg.experiment.name)
    # Log the models
    models_info = log_models_from_checkpoint(fabric, env, cfg, state)
    model_manager = MlflowModelManager(fabric, tracking_uri)
    if not set(cfg.model_manager.models.keys()).issubset(models_info.keys()):
        raise RuntimeError(
            f"The models you want to register must be a subset of the models of the {cfg.algo.name} agent. "
            f"{len(cfg.model_manager.models)} model registration "
            f"configs are given, but the agent has {len(models_info)} models. "
            f"\nModels specified in the configs: {cfg.model_manager.models.keys()}."
            f"\nModels of the {cfg.algo.name} agent: {cfg.model_manager.models.keys()}."
        )
    # Register the models specified in the configs
    for k, cfg_model in cfg.model_manager.models.items():
        model_manager.register_model(
            models_info[k]._model_uri, cfg_model["model_name"], cfg_model["description"], cfg_model["tags"]
        )
