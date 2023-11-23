from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import mlflow
import torch
import torch.nn as nn
from lightning import Fabric
from mlflow.models.model import ModelInfo
from torch import Tensor

from sheeprl.algos.sac.utils import AGGREGATOR_KEYS
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from sheeprl.algos.sac_ae.agent import SACAEContinuousActor

AGGREGATOR_KEYS = AGGREGATOR_KEYS.union({"Loss/reconstruction_loss"})
MODELS_TO_REGISTER = {"agent", "encoder", "decoder"}


@torch.no_grad()
def test_sac_ae(actor: "SACAEContinuousActor", fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, cfg.seed, 0, log_dir, "test", vector_env_idx=0)()
    cnn_keys = actor.encoder.cnn_keys
    mlp_keys = actor.encoder.mlp_keys
    actor.eval()
    done = False
    cumulative_rew = 0
    next_obs = {}
    o = env.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    for k in o.keys():
        if k in mlp_keys + cnn_keys:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
            if k in cnn_keys:
                torch_obs = torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255
            if k in mlp_keys:
                torch_obs = torch_obs.float()
            next_obs[k] = torch_obs

    while not done:
        # Act greedly through the environment
        action = actor.get_greedy_actions(next_obs)

        # Single environment step
        o, reward, done, truncated, _ = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward

        next_obs = {}
        for k in o.keys():
            if k in mlp_keys + cnn_keys:
                torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
                if k in cnn_keys:
                    torch_obs = torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255
                if k in mlp_keys:
                    torch_obs = torch_obs.float()
                next_obs[k] = torch_obs

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


def preprocess_obs(obs: Tensor, bits: int = 8):
    """Preprocessing the observations, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def weight_init(m: nn.Module):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence[ModelInfo]:
    from sheeprl.algos.sac_ae.agent import build_agent

    # Create the models
    agent, encoder, decoder = build_agent(
        fabric, cfg, env.observation_space, env.action_space, state["agent"], state["encoder"], state["decoder"]
    )

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["agent"] = mlflow.pytorch.log_model(unwrap_fabric(agent), artifact_path="agent")
        model_info["encoder"] = mlflow.pytorch.log_model(unwrap_fabric(encoder), artifact_path="encoder")
        model_info["decoder"] = mlflow.pytorch.log_model(unwrap_fabric(decoder), artifact_path="decoder")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
