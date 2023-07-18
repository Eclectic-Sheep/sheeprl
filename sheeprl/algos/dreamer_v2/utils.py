from typing import TYPE_CHECKING, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor, nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough

from sheeprl.utils.utils import make_dict_env

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v2.agent import Player

from sheeprl.algos.dreamer_v2.args import DreamerV2Args


def compute_stochastic_state(
    logits: Tensor,
    discrete: int = 32,
) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
    return dist.rsample()


def init_weights(m: nn.Module):
    """
    Initialize the parameters of the m module acording to the Xavier
    normal method.

    Args:
        m (nn.Module): the module to be initialized.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def test(
    player: "Player", fabric: Fabric, args: DreamerV2Args, cnn_keys: List[str], mlp_keys: List[str], test_name: str = ""
):
    """Test the model on the environment with the frozen model.

    Args:
        player (Player): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
    """
    env: gym.Env = make_dict_env(
        args.env_id, args.seed, 0, args, fabric.logger.log_dir, "test" + (f"_{test_name}" if test_name != "" else "")
    )()
    done = False
    cumulative_rew = 0
    device = fabric.device
    next_obs = env.reset(seed=args.seed)[0]
    for k in next_obs.keys():
        next_obs[k] = torch.from_numpy(next_obs[k]).view(args.num_envs, *next_obs[k].shape).float()
    player.init_states()
    while not done:
        # Act greedly through the environment
        preprocessed_obs = {}
        for k, v in next_obs.items():
            if k in cnn_keys:
                preprocessed_obs[k] = v[None, ...].to(device) / 255 - 0.5
            else:
                preprocessed_obs[k] = v[None, ...].to(device)
        real_actions = player.get_greedy_action(
            preprocessed_obs, False, {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.cat(real_actions, -1).cpu().numpy()
        else:
            real_actions = np.array([real_act.cpu().argmax() for real_act in real_actions])

        # Single environment step
        next_obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        for k in next_obs.keys():
            next_obs[k] = torch.from_numpy(next_obs[k]).view(args.num_envs, *next_obs[k].shape).float()
        done = done or truncated or args.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
