import torch
from gymnasium.vector import SyncVectorEnv
from lightning import Fabric

from fabricrl.algos.ppo.args import PPOArgs
from fabricrl.algos.ppo.utils import make_env
from fabricrl.algos.ppo_continuous.agent import PPOContinuousActor


@torch.inference_mode()
def test(actor: PPOContinuousActor, fabric: Fabric, args: PPOArgs):
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
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=fabric.device)
    while not done:
        # Act greedly through the environment
        action = actor.get_greedy_actions(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
        done = done or truncated
        cumulative_rew += reward.item()
        next_obs = torch.tensor(next_obs, device=fabric.device)
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
