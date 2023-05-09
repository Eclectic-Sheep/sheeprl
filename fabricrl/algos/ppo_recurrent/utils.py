import torch
from gymnasium.vector import SyncVectorEnv
from lightning import Fabric

from fabricrl.algos.ppo.args import PPOArgs
from fabricrl.algos.ppo.utils import make_env
from fabricrl.algos.ppo_recurrent.agent import RecurrentPPOAgent


@torch.inference_mode()
def test(agent: RecurrentPPOAgent, fabric: Fabric, args: PPOArgs):
    env = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                0,
                args.capture_video,
                fabric.logger.log_dir,
                "test",
                mask_velocities=args.mask_vel,
            )
        ]
    )
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=fabric.device).unsqueeze(0)
    state = (
        torch.zeros(1, 1, agent._actor_fc.output_dim, device=fabric.device),
        torch.zeros(1, 1, agent._actor_fc.output_dim, device=fabric.device),
    )
    while not done:
        # Act greedly through the environment
        action, state = agent.get_greedy_action(next_obs, state)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward.item()
        next_obs = torch.tensor(next_obs, device=fabric.device).unsqueeze(0)
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
