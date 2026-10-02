"""The rows the Dreamer players write in the replay buffer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning import Fabric

from sheeprl.algos.dreamer_v1.dreamer_v1 import SequencePlayer as DV1SequencePlayer
from sheeprl.algos.dreamer_v2.dreamer_v2 import SequencePlayer as DV2SequencePlayer
from sheeprl.algos.dreamer_v3.dreamer_v3 import SequencePlayer as DV3SequencePlayer
from sheeprl.core import EnvStep
from sheeprl.utils.utils import dotdict

CFG = dotdict(
    {
        "algo": {"cnn_keys": {"encoder": []}, "mlp_keys": {"encoder": ["state"]}},
        "buffer": {"validate_args": False, "type": "sequential"},
        "env": {"clip_rewards": False},
        "dry_run": False,
    }
)


def play(player_cls, actions_dim, num_envs, random_warmup, policy, random_actions=None, ends=None, n_steps=20):
    """Play `n_steps` steps with a player of `player_cls` in a stand-in for the environments, which records the actions
    they receive; `ends(i)` are the episodes ended by the `i`-th step. Returns the played actions and the rows written.
    """
    played, rows = [], []
    obs = {"state": np.zeros((num_envs, 2), dtype=np.float32)}

    def step(actions):
        played.append(np.array(actions).reshape(num_envs, -1))
        done = np.zeros(num_envs, dtype=bool) if ends is None else ends(len(played))
        info = {"final_obs": [{"state": np.ones(2, dtype=np.float32)} if d else None for d in done]}
        return EnvStep(obs, obs, np.zeros(num_envs), done, np.zeros(num_envs, dtype=bool), info)

    env = SimpleNamespace(num_envs=num_envs, obs=obs, policy_step=0, random_actions=random_actions, step=step)
    # The players can write the next rows in the same arrays: keep a copy
    buffer = SimpleNamespace(
        add=lambda data, indices=None, validate_args=False: rows.append(
            ({k: np.array(v, copy=True) for k, v in data.items()}, indices)
        )
    )
    schedule = SimpleNamespace(warmup=lambda policy_step: random_warmup)
    player = player_cls(Fabric(accelerator="cpu", devices=1), CFG, policy, schedule, actions_dim, False, random_warmup)
    for _ in range(n_steps):
        player.step(env, buffer)
    return played, rows


def stored_actions(player_cls, rows, actions_dim):
    """The indices of the actions stored one-hot in `rows`, in the order they were played."""
    # DreamerV1 and V2 write the first observations before the first step, with a zero action, and every action in
    # the row of the observation it led to; DreamerV3 writes every action in the row of the observation it was
    # played from
    if player_cls in (DV1SequencePlayer, DV2SequencePlayer):
        rows = rows[1:]
    split = np.cumsum(actions_dim)[:-1]
    return [np.stack([a.argmax(-1) for a in np.split(data["actions"][0], split, axis=-1)], -1) for data, _ in rows]


@pytest.mark.parametrize("player_cls", [DV1SequencePlayer, DV2SequencePlayer, DV3SequencePlayer])
@pytest.mark.parametrize("actions_dim", [[3], [3, 5]])
@pytest.mark.parametrize("num_envs", [1, 4])
def test_random_actions_are_stored_as_played(actions_dim, num_envs, player_cls):
    # The random actions are written one-hot: with multi-discrete actions and several envs they were mixed between the
    # envs (#10), so the buffer didn't hold the actions the envs played
    rng = np.random.default_rng(0)

    def random_actions():
        actions = np.stack([rng.integers(0, n, num_envs) for n in actions_dim], -1)
        return actions[:, 0] if len(actions_dim) == 1 else actions

    policy = SimpleNamespace(init_states=lambda reset_envs=None: None)
    played, rows = play(player_cls, actions_dim, num_envs, True, policy, random_actions)
    np.testing.assert_array_equal(stored_actions(player_cls, rows, actions_dim), played)


@pytest.mark.parametrize("player_cls", [DV1SequencePlayer, DV2SequencePlayer, DV3SequencePlayer])
@pytest.mark.parametrize("actions_dim", [[3], [3, 5]])
@pytest.mark.parametrize("num_envs", [1, 4])
def test_policy_actions_are_played_as_stored(actions_dim, num_envs, player_cls):
    # The policy gives one-hot actions, one tensor per discrete action: DreamerV1 and the P2E-DV1 exploration joined
    # their indices along the envs (#10), so with multi-discrete actions and several envs the envs played a mix of the
    # actions of the others while the buffer held the policy's
    generator = torch.Generator().manual_seed(0)

    def get_actions(obs, *args, **kwargs):
        return [
            torch.nn.functional.one_hot(torch.randint(0, n, (1, num_envs), generator=generator), n).float()
            for n in actions_dim
        ]

    policy = SimpleNamespace(
        init_states=lambda reset_envs=None: None, get_actions=get_actions, get_exploration_actions=get_actions
    )
    played, rows = play(player_cls, actions_dim, num_envs, False, policy)
    np.testing.assert_array_equal(stored_actions(player_cls, rows, actions_dim), played)


def test_dreamer_v1_marks_the_first_steps_of_the_episodes():
    # DreamerV1 didn't store `is_first` (#24): the world model went on through the ends of the episodes in its sequences
    num_envs = 3
    resets = []
    policy = SimpleNamespace(
        init_states=lambda reset_envs=None: resets.append(reset_envs),
        get_exploration_actions=lambda obs, *args, **kwargs: [torch.zeros(1, num_envs, 2)],
    )
    # The 2nd step ends the episode of env 1, the 3rd the ones of envs 0 and 2
    ends = {2: [False, True, False], 3: [True, False, True]}
    _, rows = play(
        DV1SequencePlayer,
        [2],
        num_envs,
        False,
        policy,
        ends=lambda i: np.array(ends.get(i, [False] * num_envs)),
        n_steps=4,
    )
    first = [(data["is_first"].reshape(-1).tolist(), indices) for data, indices in rows]
    assert first == [
        ([1, 1, 1], None),  # The first observations
        ([0, 0, 0], None),
        ([0, 0, 0], None),  # Ends env 1 with its final observation...
        ([1], [1]),  # ... and starts its next episode
        ([0, 0, 0], None),
        ([1, 1], [0, 2]),
        ([0, 0, 0], None),
    ]
    assert resets == [None, [1], [0, 2]]
