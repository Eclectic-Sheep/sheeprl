import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric

from sheeprl.algos.sac.agent import SACActor
from sheeprl.algos.sac.args import SACArgs


@torch.no_grad()
def test(actor: SACActor, env: gym.Env, fabric: Fabric, args: SACArgs):
    actor.eval()
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(
        np.array(env.reset(seed=args.seed)[0]), device=fabric.device, dtype=torch.float32
    ).unsqueeze(0)
    while not done:
        # Act greedly through the environment
        action = actor.get_greedy_actions(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=fabric.device, dtype=torch.float32).unsqueeze(0)

        if args.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
