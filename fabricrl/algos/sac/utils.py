import torch
from torch.utils.tensorboard import SummaryWriter

from fabricrl.algos.ppo.utils import make_env
from fabricrl.algos.sac.agent import Actor
from fabricrl.algos.sac.args import SACArgs


@torch.no_grad()
def test(actor: Actor, device: torch.device, logger: SummaryWriter, args: SACArgs):
    env = make_env(args.env_id, args.seed, 0, args.capture_video, logger.log_dir, "test", mask_velocities=False)()
    step = 0
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=device)
    while not done:
        # Act greedly through the environment
        action = actor.get_greedy_action(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=device)
        step += 1
    logger.add_scalar("Test/cumulative_reward", cumulative_rew, 0)
    env.close()
