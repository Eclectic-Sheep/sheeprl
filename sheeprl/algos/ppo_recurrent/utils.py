import gymnasium as gym
import torch
from lightning import Fabric
from omegaconf import DictConfig

from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent


@torch.no_grad()
def test(agent: RecurrentPPOAgent, env: gym.Env, fabric: Fabric, cfg: DictConfig):
    agent.eval()
    done = False
    cumulative_rew = 0
    with fabric.device:
        o = env.reset(seed=cfg.seed)[0]
        next_obs = torch.cat([torch.tensor(o[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1).view(
            1, 1, -1
        )
        state = (
            torch.zeros(1, 1, agent.lstm_hidden_size),
            torch.zeros(1, 1, agent.lstm_hidden_size),
        )
    while not done:
        # Act greedly through the environment
        action, state = agent.get_greedy_action(next_obs, state)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        with fabric.device:
            next_obs = torch.cat(
                [torch.tensor(next_obs[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1
            ).view(1, 1, -1)

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
