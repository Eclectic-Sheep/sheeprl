import torch
from gymnasium.vector import SyncVectorEnv
from lightning import Fabric

from sheeprl.algos.ppo.args import PPOArgs
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent


@torch.no_grad()
def test(agent: RecurrentPPOAgent, envs: SyncVectorEnv, fabric: Fabric, args: PPOArgs):
    agent.eval()
    done = False
    cumulative_rew = 0
    env = envs.envs[0]
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=fabric.device).view(1, 1, -1)
    state = (
        torch.zeros(1, 1, agent.lstm_hidden_size, device=fabric.device),
        torch.zeros(1, 1, agent.lstm_hidden_size, device=fabric.device),
    )
    while not done:
        # Act greedly through the environment
        action, state = agent.get_greedy_action(next_obs, state)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=fabric.device).view(1, 1, -1)
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
