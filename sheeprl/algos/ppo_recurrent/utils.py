from typing import TYPE_CHECKING

import torch
from lightning import Fabric
from omegaconf import DictConfig

from sheeprl.utils.env import make_env

if TYPE_CHECKING:
    from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent


@torch.no_grad()
def test(agent: "RecurrentPPOAgent", fabric: Fabric, cfg: DictConfig):
    env = make_env(cfg, None, 0, fabric.logger.log_dir, "test", vector_env_idx=0)()

    agent.eval()
    done = False
    cumulative_rew = 0
    with fabric.device:
        o = env.reset(seed=cfg.seed)[0]
        next_obs = {
            k: torch.tensor(o[k], dtype=torch.float32, device=fabric.device).view(1, 1, -1, *o[k].shape[-2:])
            for k in cfg.cnn_keys.encoder
        }
        next_obs.update(
            {
                k: torch.tensor(o[k], dtype=torch.float32, device=fabric.device).view(1, 1, -1)
                for k in cfg.mlp_keys.encoder
            }
        )
        state = (torch.zeros(1, 1, agent.rnn_hidden_size, device=fabric.device),) * 2
        actions = torch.zeros(1, 1, sum(agent.actions_dim), device=fabric.device)
    while not done:
        # Act greedly through the environment
        actions, state = agent.get_greedy_actions(next_obs, state, actions)
        if agent.is_continuous:
            real_actions = torch.cat(actions, -1)
            actions = torch.cat(actions, dim=-1).view(1, 1, -1)
        else:
            real_actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)
            actions = torch.cat([act for act in actions], dim=-1).view(1, 1, -1)

        # Single environment step
        o, reward, done, truncated, info = env.step(real_actions.cpu().numpy().reshape(env.action_space.shape))
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


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(n):
    def f(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight.data, torch.tensor(n).sqrt())
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.orthogonal_(m.weight.data, torch.tensor(n).sqrt())
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, torch.nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f
