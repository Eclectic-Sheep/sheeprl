import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric

from sheeprl.algos.sac_pixel.agent import SACPixelContinuousActor
from sheeprl.algos.sac_pixel.args import SACPixelContinuousArgs


@torch.no_grad()
def test_sac_pixel(
    actor: SACPixelContinuousActor, env: gym.Env, fabric: Fabric, args: SACPixelContinuousArgs, normalize: bool = False
):
    actor.eval()
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(np.array(env.reset(seed=args.seed)[0]), device=fabric.device).unsqueeze(0)
    while not done:
        # Act greedly through the environment
        next_obs = next_obs.flatten(start_dim=1, end_dim=-3)
        if normalize:
            next_obs = next_obs / 255.0 - 0.5
        action = actor.get_greedy_actions(next_obs)

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
