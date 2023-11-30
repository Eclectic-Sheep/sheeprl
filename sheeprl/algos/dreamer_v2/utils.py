from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn
from lightning import Fabric
from mlflow.models.model import ModelInfo
from torch import Tensor
from torch.distributions import Independent

from sheeprl.utils.distribution import OneHotCategoricalStraightThroughValidateArgs
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v1.agent import PlayerDV1
    from sheeprl.algos.dreamer_v2.agent import PlayerDV2


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
    "Params/exploration_amount",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic"}


def compute_stochastic_state(logits: Tensor, discrete: int = 32, sample=True, validate_args=False) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        sample (bool): whether or not to sample the stochastic state.
            Default to True.
        validate_args: whether or not to validate distribution arguments.
            Default to False.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThroughValidateArgs(logits=logits, validate_args=validate_args), 1)
    stochastic_state = dist.rsample() if sample else dist.mode
    return stochastic_state


def init_weights(m: nn.Module, mode: str = "normal"):
    """
    Initialize the parameters of the m module acording to the Xavier
    normal method.

    Args:
        m (nn.Module): the module to be initialized.
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


@torch.no_grad()
def test(
    player: Union["PlayerDV2", "PlayerDV1"],
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    test_name: str = "",
    sample_actions: bool = False,
):
    """Test the model on the environment with the frozen model.

    Args:
        player (PlayerDV2 | PlayerDV1): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        cfg (Dict[str, Any]): the hyper-parameters.
        log_dir (str): the logging directory.
        test_name (str): the name of the test.
            Default to "".
        sample_actoins (bool): whether or not to sample actions.
            Default to False.
    """
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()
    done = False
    cumulative_rew = 0
    device = fabric.device
    next_obs = env.reset(seed=cfg.seed)[0]
    for k in next_obs.keys():
        next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
    player.num_envs = 1
    player.init_states()
    while not done:
        # Act greedly through the environment
        preprocessed_obs = {}
        for k, v in next_obs.items():
            if k in cfg.algo.cnn_keys.encoder:
                preprocessed_obs[k] = v[None, ...].to(device) / 255 - 0.5
            elif k in cfg.algo.mlp_keys.encoder:
                preprocessed_obs[k] = v[None, ...].to(device)
        real_actions = player.get_greedy_action(
            preprocessed_obs, sample_actions, {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.cat(real_actions, -1).cpu().numpy()
        else:
            real_actions = np.array([real_act.cpu().argmax(dim=-1).numpy() for real_act in real_actions])

        # Single environment step
        next_obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        for k in next_obs.keys():
            next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
        done = done or truncated or cfg.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0 and len(fabric.loggers) > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence[ModelInfo]:
    from sheeprl.algos.dreamer_v2.agent import build_agent

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    world_model, actor, critic, target_critic = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["actor"],
        state["critic"],
        state["target_critic"],
    )

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor"] = mlflow.pytorch.log_model(unwrap_fabric(actor), artifact_path="actor")
        model_info["critic"] = mlflow.pytorch.log_model(unwrap_fabric(critic), artifact_path="critic")
        model_info["target_critic"] = mlflow.pytorch.log_model(target_critic, artifact_path="target_critic")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
