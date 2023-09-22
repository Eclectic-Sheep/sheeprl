import torch
from lightning import Fabric
from omegaconf import DictConfig

from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent
from sheeprl.utils.env import make_env


@torch.no_grad()
def test(agent: RecurrentPPOAgent, fabric: Fabric, cfg: DictConfig):
    env = make_env(cfg, None, 0, fabric.logger.log_dir, "test", vector_env_idx=0)()

    agent.eval()
    done = False
    cumulative_rew = 0
    with fabric.device:
        o = env.reset(seed=cfg.seed)[0]
        next_obs = {
            k: torch.tensor(o[k], dtype=torch.float32).view(1, 1, -1, *o[k].shape[-2:]) for k in cfg.cnn_keys.encoder
        }
        next_obs.update({k: torch.tensor(o[k], dtype=torch.float32).view(1, 1, -1) for k in cfg.mlp_keys.encoder})
        state = torch.zeros(1, 1, agent.rnn_hidden_size, device=fabric.device)
        actions = torch.zeros(1, 1, sum(agent.actions_dim), device=fabric.device)
    while not done:
        # Act greedly through the environment
        actions, state = agent.get_greedy_actions(next_obs, state, actions)
        if agent.is_continuous:
            actions = torch.cat(actions, dim=-1)
        else:
            actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)

        # Single environment step
        o, reward, done, truncated, info = env.step(actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        with fabric.device:
            next_obs = {
                k: torch.tensor(o[k], dtype=torch.float32).view(1, 1, -1, *o[k].shape[-2:])
                for k in cfg.cnn_keys.encoder
            }
            next_obs.update({k: torch.tensor(o[k], dtype=torch.float32).view(1, 1, -1) for k in cfg.mlp_keys.encoder})

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
