from typing import TYPE_CHECKING

import gymnasium as gym
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor

from sheeprl.algos.sac_ae.args import SACAEArgs

if TYPE_CHECKING:
    from sheeprl.algos.sac_ae.agent import SACPixelContinuousActor


@torch.no_grad()
def test_sac_pixel(
    actor: "SACPixelContinuousActor",
    env: gym.Env,
    fabric: Fabric,
    args: SACAEArgs,
):
    cnn_keys = actor.encoder.cnn_keys
    mlp_keys = actor.encoder.mlp_keys
    actor.eval()
    done = False
    cumulative_rew = 0
    next_obs = {}
    o = env.reset(seed=args.seed)[0]  # [N_envs, N_obs]
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

        if args.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
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
