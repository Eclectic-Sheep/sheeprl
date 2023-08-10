import os
from typing import TYPE_CHECKING, Any, List

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor

from sheeprl.utils.env import make_dict_env

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v3.agent import PlayerDV3
    from sheeprl.algos.dreamer_v3.args import DreamerV3Args


class Moments(torch.nn.Module):
    def __init__(
        self,
        fabric: Fabric,
        decay: float = 0.99,
        max_: float = 1e8,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._fabric = fabric
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor) -> Any:
        gathered_x = self._fabric.all_gather(x).detach()
        low = torch.quantile(gathered_x, self._percentile_low)
        high = torch.quantile(gathered_x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
):
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret


@torch.no_grad()
def test(
    player: "PlayerDV3",
    fabric: Fabric,
    args: "DreamerV3Args",
    cnn_keys: List[str],
    mlp_keys: List[str],
    test_name: str = "",
    sample_actions: bool = False,
):
    """Test the model on the environment with the frozen model.

    Args:
        player (PlayerDV2): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        args (Union[DreamerV3Args, DreamerV2Args, DreamerV1Args]): the hyper-parameters.
        cnn_keys (Sequence[str]): the keys encoded by the cnn encoder.
        mlp_keys (Sequence[str]): the keys encoded by the mlp encoder.
        test_name (str): the name of the test.
            Default to "".
    """
    log_dir = fabric.logger.log_dir if len(fabric.loggers) > 0 else os.getcwd()
    env: gym.Env = make_dict_env(
        args.env_id, args.seed, 0, args, log_dir, "test" + (f"_{test_name}" if test_name != "" else "")
    )()
    done = False
    cumulative_rew = 0
    device = fabric.device
    next_obs = env.reset(seed=args.seed)[0]
    for k in next_obs.keys():
        next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
    player.num_envs = 1
    player.init_states()
    while not done:
        # Act greedly through the environment
        preprocessed_obs = {}
        for k, v in next_obs.items():
            if k in cnn_keys:
                preprocessed_obs[k] = v[None, ...].to(device) / 255
            elif k in mlp_keys:
                preprocessed_obs[k] = v[None, ...].to(device)
        real_actions = player.get_greedy_action(
            preprocessed_obs, sample_actions, {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.cat(real_actions, -1).cpu().numpy()
        else:
            real_actions = np.array([real_act.cpu().argmax(dim=-1).numpy() for real_act in real_actions])

        # Single environment step
        next_obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        for k in next_obs.keys():
            next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
        done = done or truncated or args.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    if len(fabric.loggers) > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
