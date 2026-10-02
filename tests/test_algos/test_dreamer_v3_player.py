"""The rows the DreamerV3 player writes in the replay buffer."""

from types import SimpleNamespace

import numpy as np
import pytest
from lightning import Fabric

from sheeprl.algos.dreamer_v3.dreamer_v3 import SequencePlayer
from sheeprl.core import EnvStep
from sheeprl.utils.utils import dotdict


@pytest.mark.parametrize("actions_dim", [[3], [3, 5]])
@pytest.mark.parametrize("num_envs", [1, 4])
def test_random_actions_are_stored_as_played(actions_dim, num_envs):
    # The random actions are written one-hot: with multi-discrete actions and several envs they were mixed between the
    # envs (#10), so the buffer didn't hold the actions the envs played
    rng = np.random.default_rng(0)
    played, rows = [], []
    obs = {"state": np.zeros((num_envs, 2), dtype=np.float32)}
    no_end = np.zeros(num_envs, dtype=bool)

    def random_actions():
        actions = np.stack([rng.integers(0, n, num_envs) for n in actions_dim], -1)
        return actions[:, 0] if len(actions_dim) == 1 else actions

    def step(actions):
        played.append(np.array(actions).reshape(num_envs, -1))
        return EnvStep(obs, obs, np.zeros(num_envs), no_end, no_end, {})

    env = SimpleNamespace(num_envs=num_envs, obs=obs, policy_step=0, random_actions=random_actions, step=step)
    buffer = SimpleNamespace(add=lambda data, indices=None, validate_args=False: rows.append(data["actions"].copy()))
    cfg = dotdict(
        {
            "algo": {"cnn_keys": {"encoder": []}, "mlp_keys": {"encoder": ["state"]}},
            "buffer": {"validate_args": False},
            "env": {"clip_rewards": False},
        }
    )
    policy = SimpleNamespace(init_states=lambda reset_envs=None: None)
    schedule = SimpleNamespace(warmup=lambda policy_step: True)
    player = SequencePlayer(Fabric(accelerator="cpu", devices=1), cfg, policy, schedule, actions_dim, False, True)
    for _ in range(20):
        player.step(env, buffer)

    split = np.cumsum(actions_dim)[:-1]
    stored = [np.stack([a.argmax(-1) for a in np.split(row[0], split, axis=-1)], -1) for row in rows]
    np.testing.assert_array_equal(stored, played)
