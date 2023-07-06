import gymnasium as gym
import torch
from lightning import Fabric

from sheeprl.algos.ppo.args import PPOArgs
from sheeprl.algos.ppo_continuous.agent import PPOContinuousActor


@torch.no_grad()
def test(actor: PPOContinuousActor, env: gym.Env, fabric: Fabric, args: PPOArgs):
    actor.eval()
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=fabric.device).unsqueeze(0)
    while not done:
        # Act greedly through the environment
        action = actor.get_greedy_actions(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=fabric.device).unsqueeze(0)

        if args.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
