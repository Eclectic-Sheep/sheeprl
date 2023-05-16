import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from lightning import Fabric

from sheeprl.algos.ppo.args import PPOArgs


@torch.no_grad()
def test(actor: nn.Module, envs: SyncVectorEnv, fabric: Fabric, args: PPOArgs):
    actor.eval()
    done = False
    cumulative_rew = 0
    env = envs.envs[0]
    next_obs = torch.tensor(np.array(env.reset(seed=args.seed)[0]), device=fabric.device).unsqueeze(0)
    while not done:
        # Act greedly through the environment
        logits = actor(next_obs)
        action = F.softmax(logits, dim=-1).argmax(dim=-1)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(np.array(next_obs), device=fabric.device).unsqueeze(0)

        if args.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
