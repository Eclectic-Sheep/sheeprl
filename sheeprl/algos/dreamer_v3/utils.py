from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor, nn

from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

    from sheeprl.algos.dreamer_v3.agent import DreamerV3Policy

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic", "moments"}


class Moments(nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1e8,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor, fabric: Fabric) -> Any:
        gathered_x = fabric.all_gather(x).float().detach()
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


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    for k, v in obs.items():
        torch_obs[k] = torch.from_numpy(v.copy()).to(fabric.device).float()
        if k in cnn_keys:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1, *v.shape[-2:]) / 255 - 0.5
        else:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1)

    return torch_obs


@torch.no_grad()
def test(
    policy: "DreamerV3Policy",
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    test_name: str = "",
    greedy: bool = True,
    policy_step: int = 0,
):
    """Test the model on the environment with the frozen model.

    Args:
        policy (DreamerV3Policy): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        cfg (DictConfig): the hyper-parameters.
        log_dir (str): the logging directory.
        test_name (str): the name of the test.
            Default to "".
        greedy (bool): whether or not to sample the actions.
            Default to True.
    """
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()
    done = False
    cumulative_rew = 0
    obs = env.reset(seed=cfg.seed)[0]
    policy.num_envs = 1
    policy.init_states()
    while not done:
        # Act greedly through the environment
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        real_actions = policy.get_actions(
            torch_obs, greedy, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        )
        if policy.actor.is_continuous:
            real_actions = torch.stack(real_actions, -1).cpu().numpy()
        else:
            real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

        # Single environment step
        obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        done = done or truncated or cfg.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0 and len(fabric.loggers) > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, policy_step)
    env.close()


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    from sheeprl.algos.dreamer_v3.dreamer_v3 import DreamerV3
    from sheeprl.core import log_models_from_checkpoint as log_trained_models

    return log_trained_models(
        fabric,
        env,
        cfg,
        state,
        DreamerV3(fabric, cfg.to_log),
        lambda trained: {
            "world_model": trained.world_model,
            "actor": trained.actor,
            "critic": trained.critic,
            "target_critic": trained.target_critic,
            "moments": trained.moments,
        },
    )
