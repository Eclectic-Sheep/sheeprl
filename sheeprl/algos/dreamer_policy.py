"""The `Policy` shared by the policies of the Dreamers (V1, V2, V3, V3.5) and of Plan2Explore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sheeprl.core.collector import Act, Policy
from sheeprl.utils.obs import prepare_obs

if TYPE_CHECKING:
    from sheeprl.core.environment import Environment


class DreamerPolicy(Policy):
    """The `Policy` of the Dreamers' policies, on their `get_actions`, which updates the latent states of the
    environments (created by `init_states`, reset by `reset_state`): the observations are moved to `device` (the images
    `cnn_keys` scaled to [-0.5, 0.5]), the actions are written one-hot for discrete actions (the environments take their
    indices), and the action masks of the observations (`mask*`, e.g. MineDojo) mask the actions.

    A subclass is an `nn.Module` with the attributes `device`, `cnn_keys`, `actions_dim` and `actor`.
    """

    def act(self, obs: Dict[str, np.ndarray], greedy: bool = False) -> Act:
        num_envs = len(next(iter(obs.values())))
        torch_obs = prepare_obs(self.device, obs, cnn_keys=self.cnn_keys, num_envs=num_envs, time_dim=True)
        mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        actions = self.sample_actions(torch_obs, mask if len(mask) > 0 else None, greedy)
        if self.actor.is_continuous:
            env_actions = torch.stack(actions, dim=-1)
        else:
            env_actions = torch.stack([act.argmax(dim=-1) for act in actions], dim=-1)
        return Act(env_actions.cpu().numpy(), {"actions": torch.cat(actions, -1).cpu().numpy()})

    def sample_actions(
        self, obs: Dict[str, Tensor], mask: Optional[Dict[str, Tensor]], greedy: bool = False
    ) -> Sequence[Tensor]:
        """The actions of `act`, one tensor per action."""
        return self.get_actions(obs, greedy=greedy, mask=mask)

    def random(self, env: Environment) -> Act:
        """Uniformly random actions; with action masks in the observations (MineDojo), the actions of the policy, since
        random actions would ignore them."""
        if any(k.startswith("mask") for k in env.obs):
            return self.act(env.obs)
        env_actions = actions = np.array(env.random_actions())
        if not self.actor.is_continuous:
            # One row per environment, one column per discrete action: one-hot each column
            per_action = actions.reshape(env.num_envs, len(self.actions_dim)).T
            actions = np.concatenate(
                [
                    F.one_hot(torch.as_tensor(act), act_dim).numpy()
                    for act, act_dim in zip(per_action, self.actions_dim)
                ],
                axis=-1,
            )
        return Act(env_actions, {"actions": actions})
