from __future__ import annotations

import torch
from torch import Tensor


def ant_termination_fn(obs: Tensor, actions: Tensor, next_obs: Tensor):
    """Termination function of the ant environment."""
    assert len(obs.shape) == len(next_obs.shape) == len(actions.shape) == 2

    x = next_obs[:, 0]
    not_done = torch.isfinite(next_obs).all(dim=-1) * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    done = done[:, None]
    return done


def walker_termination_fn(obs: Tensor, actions: Tensor, next_obs: Tensor):
    """Termination function of the walker environment."""
    assert len(obs.shape) == len(next_obs.shape) == len(actions.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


# def humanoid_termination_fn(obs: Tensor, actions: Tensor, next_obs: Tensor):
# """Termination function of the humanoid environment."""
#     assert len(obs.shape) == len(next_obs.shape) == len(actions.shape) == 2

#     z = next_obs[:, 0]
#     done = (z < 1.0) + (z > 2.0)

#     done = done[:, None]
#     return done


def halfcheetah_termination_fn(obs: Tensor, actions: Tensor, next_obs: Tensor):
    """Termination function of the halfcheetah environment."""
    assert len(obs.shape) == len(next_obs.shape) == len(actions.shape) == 2

    done = torch.zeros(obs.shape[0], 1, dtype=torch.bool)
    return done


def hopper_termination_fn(obs: Tensor, actions: Tensor, next_obs: Tensor):
    """Termination function of the hopper environment."""
    assert len(obs.shape) == len(next_obs.shape) == len(actions.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        torch.isfinite(next_obs).all(dim=-1)
        * torch.abs(next_obs[:, 1:] < 100).all(dim=-1)
        * (height > 0.7)
        * (torch.abs(angle) < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done


TERMINATION_FUNCTIONS = {
    "halfcheetah-random-v0": halfcheetah_termination_fn,
    "halfcheetah-random-v2": halfcheetah_termination_fn,
    "halfcheetah-medium-v0": halfcheetah_termination_fn,
    "halfcheetah-medium-v2": halfcheetah_termination_fn,
    "halfcheetah-expert-v0": halfcheetah_termination_fn,
    "halfcheetah-expert-v2": halfcheetah_termination_fn,
    "halfcheetah-medium-replay-v0": halfcheetah_termination_fn,
    "halfcheetah-medium-replay-v2": halfcheetah_termination_fn,
    "halfcheetah-medium-expert-v0": halfcheetah_termination_fn,
    "halfcheetah-medium-expert-v2": halfcheetah_termination_fn,
    "walker2d-random-v0": walker_termination_fn,
    "walker2d-random-v2": walker_termination_fn,
    "walker2d-medium-v0": walker_termination_fn,
    "walker2d-medium-v2": walker_termination_fn,
    "walker2d-expert-v0": walker_termination_fn,
    "walker2d-expert-v2": walker_termination_fn,
    "walker2d-medium-replay-v0": walker_termination_fn,
    "walker2d-medium-replay-v2": walker_termination_fn,
    "walker2d-medium-expert-v0": walker_termination_fn,
    "walker2d-medium-expert-v2": walker_termination_fn,
    "hopper-random-v0": hopper_termination_fn,
    "hopper-random-v2": hopper_termination_fn,
    "hopper-medium-v0": hopper_termination_fn,
    "hopper-medium-v2": hopper_termination_fn,
    "hopper-expert-v0": hopper_termination_fn,
    "hopper-expert-v2": hopper_termination_fn,
    "hopper-medium-replay-v0": hopper_termination_fn,
    "hopper-medium-replay-v2": hopper_termination_fn,
    "hopper-medium-expert-v0": hopper_termination_fn,
    "hopper-medium-expert-v2": hopper_termination_fn,
    "ant-random-v0": ant_termination_fn,
    "ant-random-v2": ant_termination_fn,
    "ant-medium-v0": ant_termination_fn,
    "ant-medium-v2": ant_termination_fn,
    "ant-expert-v0": ant_termination_fn,
    "ant-expert-v2": ant_termination_fn,
    "ant-medium-replay-v0": ant_termination_fn,
    "ant-medium-replay-v2": ant_termination_fn,
    "ant-medium-expert-v0": ant_termination_fn,
    "ant-medium-expert-v2": ant_termination_fn,
}
