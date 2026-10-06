"""The actions the Dreamer policies play and the rows their writers write in the replay buffer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning import Fabric

from sheeprl.algos.dreamer_v1.dreamer_v1 import SequenceWriter as DV1SequenceWriter
from sheeprl.algos.dreamer_v2.agent import DreamerPolicy
from sheeprl.algos.dreamer_v2.dreamer_v2 import SequenceWriter as DV2SequenceWriter
from sheeprl.algos.dreamer_v3.dreamer_v3 import SequenceWriter as DV3SequenceWriter
from sheeprl.core import Collector, EnvStep
from sheeprl.utils.utils import dotdict

CFG = dotdict(
    {
        "algo": {"cnn_keys": {"encoder": []}, "mlp_keys": {"encoder": ["state"]}},
        "buffer": {"validate_args": False, "type": "sequential"},
        "env": {"clip_rewards": False},
        "dry_run": False,
    }
)


class StandInPolicy(DreamerPolicy):
    """A Dreamer policy whose actions come from `get_actions`, which records its resets."""

    def __init__(self, actions_dim, get_actions=None):
        self.fabric = Fabric(accelerator="cpu", devices=1)
        self.cnn_keys = []
        self.actions_dim = actions_dim
        self.actor = SimpleNamespace(is_continuous=False)
        self.get_actions = get_actions
        self.resets = []

    def init_states(self, reset_envs=None):
        self.resets.append(reset_envs)


def play(writer_cls, num_envs, random_warmup, policy, random_actions=None, ends=None, n_steps=20):
    """Play `n_steps` steps of `policy` with a writer of `writer_cls` in a stand-in for the environments, which records
    the actions they receive; `ends(i)` are the episodes ended by the `i`-th step. Returns the played actions and the
    rows written."""
    played, rows = [], []
    obs = {"state": np.zeros((num_envs, 2), dtype=np.float32)}

    def step(actions):
        played.append(np.array(actions).reshape(num_envs, -1))
        done = np.zeros(num_envs, dtype=bool) if ends is None else ends(len(played))
        final_obs = [{"state": np.ones(2, dtype=np.float32)} if d else None for d in done]
        no_envs = np.zeros(num_envs, dtype=bool)
        return EnvStep(obs, obs, np.zeros(num_envs), done, no_envs, final_obs=final_obs, restarted=no_envs)

    env = SimpleNamespace(num_envs=num_envs, obs=obs, random_actions=random_actions, step=step, reset=lambda: obs)
    # The writers can write the next rows in the same arrays: keep a copy
    buffer = SimpleNamespace(
        add=lambda data, indices=None, validate_args=False: rows.append(
            ({k: np.array(v, copy=True) for k, v in data.items()}, indices)
        )
    )
    schedule = SimpleNamespace(policy_step=0, policy_steps_per_step=num_envs, warmup=lambda policy_step: random_warmup)
    cadence = SimpleNamespace(accumulate_episodes=lambda episodes, policy_step: None)
    collector = Collector(env, policy, writer_cls(CFG, policy.actions_dim), buffer, schedule, cadence)
    collector.reset()
    collector.collect(n_steps)
    return played, rows


def stored_actions(writer_cls, rows, actions_dim):
    """The indices of the actions stored one-hot in `rows`, in the order they were played."""
    # DreamerV1 and V2 write the first observations before the first step, with a zero action, and every action in
    # the row of the observation it led to; DreamerV3 writes every action in the row of the observation it was
    # played from
    if writer_cls in (DV1SequenceWriter, DV2SequenceWriter):
        rows = rows[1:]
    split = np.cumsum(actions_dim)[:-1]
    return [np.stack([a.argmax(-1) for a in np.split(data["actions"][0], split, axis=-1)], -1) for data, _ in rows]


@pytest.mark.parametrize("writer_cls", [DV1SequenceWriter, DV2SequenceWriter, DV3SequenceWriter])
@pytest.mark.parametrize("actions_dim", [[3], [3, 5]])
@pytest.mark.parametrize("num_envs", [1, 4])
def test_random_actions_are_stored_as_played(actions_dim, num_envs, writer_cls):
    # The random actions are written one-hot: with multi-discrete actions and several envs they were mixed between the
    # envs, so the buffer didn't hold the actions the envs played
    rng = np.random.default_rng(0)

    def random_actions():
        actions = np.stack([rng.integers(0, n, num_envs) for n in actions_dim], -1)
        return actions[:, 0] if len(actions_dim) == 1 else actions

    played, rows = play(writer_cls, num_envs, True, StandInPolicy(actions_dim), random_actions)
    np.testing.assert_array_equal(stored_actions(writer_cls, rows, actions_dim), played)


@pytest.mark.parametrize("writer_cls", [DV1SequenceWriter, DV2SequenceWriter, DV3SequenceWriter])
@pytest.mark.parametrize("actions_dim", [[3], [3, 5]])
@pytest.mark.parametrize("num_envs", [1, 4])
def test_policy_actions_are_played_as_stored(actions_dim, num_envs, writer_cls):
    # The policy gives one-hot actions, one tensor per discrete action: DreamerV1 and the P2E-DV1 exploration joined
    # their indices along the envs, so with multi-discrete actions and several envs the envs played a mix of the
    # actions of the others while the buffer held the policy's
    generator = torch.Generator().manual_seed(0)

    def get_actions(obs, *args, **kwargs):
        return [
            torch.nn.functional.one_hot(torch.randint(0, n, (1, num_envs), generator=generator), n).float()
            for n in actions_dim
        ]

    played, rows = play(writer_cls, num_envs, False, StandInPolicy(actions_dim, get_actions))
    np.testing.assert_array_equal(stored_actions(writer_cls, rows, actions_dim), played)


def test_dreamer_v1_marks_the_first_steps_of_the_episodes():
    # DreamerV1 didn't store `is_first`: the world model went on through the ends of the episodes in its sequences
    num_envs = 3
    policy = StandInPolicy([2], lambda obs, *args, **kwargs: [torch.zeros(1, num_envs, 2)])
    # The 2nd step ends the episode of env 1, the 3rd the ones of envs 0 and 2
    ends = {2: [False, True, False], 3: [True, False, True]}
    _, rows = play(
        DV1SequenceWriter,
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
    assert policy.resets == [None, [1], [0, 2]]


def test_dreamer_v3_resets_the_policy_of_the_ended_episodes():
    # The writer of DreamerV3 zeroes the episode flags of the ended episodes in place, in the arrays of the step: the
    # collector found no ended episodes after it, and the policy went on with the latent states of the ended ones
    num_envs = 3
    policy = StandInPolicy([2], lambda obs, *args, **kwargs: [torch.zeros(1, num_envs, 2)])
    ends = {2: [False, True, False], 3: [True, False, True]}
    play(
        DV3SequenceWriter, num_envs, False, policy, ends=lambda i: np.array(ends.get(i, [False] * num_envs)), n_steps=4
    )
    assert policy.resets == [None, [1], [0, 2]]


def test_masked_actions_are_never_random():
    # With action masks in the observations (MineDojo) the policy plays also before the training starts: random actions
    # would ignore the masks
    num_envs = 2
    calls = []

    def get_actions(obs, mask=None, **kwargs):
        calls.append(sorted(mask))
        return [torch.nn.functional.one_hot(torch.zeros(1, num_envs, dtype=torch.long), 3).float()]

    def random_actions():
        raise AssertionError("random actions with action masks")

    policy = StandInPolicy([3], get_actions)
    env = SimpleNamespace(
        num_envs=num_envs,
        obs={"state": np.zeros((num_envs, 2)), "mask_action_type": np.ones((num_envs, 3))},
        random_actions=random_actions,
    )
    act = policy.random(env)
    assert calls == [["mask_action_type"]]
    np.testing.assert_array_equal(act.env_actions.reshape(num_envs, -1), np.zeros((num_envs, 1)))
