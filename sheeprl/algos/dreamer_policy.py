"""The `Policy` shared by the policies of the Dreamers (V1, V2, V3, V3.5) and of Plan2Explore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from sheeprl.core.collector import Act, Policy
from sheeprl.models.io import to_device

if TYPE_CHECKING:
    from sheeprl.core.actions import ActionCodec
    from sheeprl.core.environment import Environment


class DreamerPolicy(Policy):
    """The `Policy` of the Dreamers' policies, on their `get_actions`, which updates the latent states of the
    environments (created by `init_states`, reset by `reset_state`): the observations of the keys of the inputs and of
    the action masks are moved to `device` with a time dimension of one step, the action masks (`mask_keys`, the extra
    keys the actor reads, e.g. the ones of MineDojo) mask the actions, and the actions go to the environments and to the
    store through `codec` (one-hot for discrete actions, while the environments take their indices).

    A subclass is an `nn.Module` with the attributes `device`, `keys` (the keys of the inputs of its encoder),
    `mask_keys`, `codec` and `actor`, and the method `get_actions`.
    """

    # Set by the subclasses
    device: Union[str, torch.device]
    keys: Tuple[str, ...]
    mask_keys: Tuple[str, ...]
    codec: ActionCodec

    def get_actions(
        self, obs: Dict[str, Tensor], greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None
    ) -> Sequence[Tensor]:
        """The actions for `obs`, one tensor per action, updating the latent states of the environments."""
        raise NotImplementedError

    def obs_to_device(self, obs: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        """The observations of the keys of the inputs and of the action masks, on the device of the policy, as
        `[1, Num_Envs, ...]`."""
        keys = tuple(self.keys) + tuple(k for k in self.mask_keys if k not in self.keys)
        return {k: v[None] for k, v in to_device({k: obs[k] for k in keys}, self.device).items()}

    def act(self, obs: Dict[str, np.ndarray], greedy: bool = False) -> Act:
        torch_obs = self.obs_to_device(obs)
        mask = {k: torch_obs[k] for k in self.mask_keys}
        actions = self.sample_actions(torch_obs, mask if len(mask) > 0 else None, greedy)
        return Act(self.codec.to_env(actions), {"actions": self.codec.to_store(actions)})

    def sample_actions(
        self, obs: Dict[str, Tensor], mask: Optional[Dict[str, Tensor]], greedy: bool = False
    ) -> Sequence[Tensor]:
        """The actions of `act`, one tensor per action."""
        return self.get_actions(obs, greedy=greedy, mask=mask)

    def random(self, env: Environment) -> Act:
        """Uniformly random actions; with action masks (MineDojo), the actions of the policy, since random actions would
        ignore them."""
        if len(self.mask_keys) > 0:
            return self.act(env.obs)
        env_actions = np.array(env.random_actions())
        return Act(env_actions, {"actions": self.codec.from_env(env_actions)})
