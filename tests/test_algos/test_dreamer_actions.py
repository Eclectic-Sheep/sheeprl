"""The Dreamers and the explorations of Plan2Explore play and store the multi-discrete actions of every environment."""

from __future__ import annotations

import copy
import os
import shutil
import sys
from typing import Dict, List, Tuple
from unittest import mock

import numpy as np
import pytest

from sheeprl import ROOT_DIR
from sheeprl.data.buffers import EnvIndependentReplayBuffer
from sheeprl.envs.dummy import MultiDiscreteDummyEnv

# Two discrete actions of different sizes: one-hot encoded one after the other in the stored actions
ACTION_DIMS = [3, 5]
NUM_ENVS = 4

ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "env=dummy",
    "env.id=multidiscrete_dummy",
    f"env.num_envs={NUM_ENVS}",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=0",
    "checkpoint.save_last=False",
    "algo.run_test=False",
    "algo.cnn_keys.encoder=[]",
    "algo.cnn_keys.decoder=[]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
    "algo.dense_units=8",
    "algo.world_model.recurrent_model.recurrent_state_size=8",
    "algo.world_model.representation_model.hidden_size=8",
    "algo.world_model.transition_model.hidden_size=8",
    "algo.per_rank_batch_size=1",
    "algo.per_rank_sequence_length=1",
    # Played, not trained: 6 iterations of the 4 environments
    "algo.replay_ratio=0",
    "algo.total_steps=24",
    "buffer.size=100",
]

EXPERIMENTS = [
    "dreamer_v1",
    "dreamer_v2",
    "dreamer_v3",
    "p2e_dv1_exploration",
    "p2e_dv2_exploration",
    "p2e_dv3_exploration",
]


def play(exp: str, args: List[str]) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
    """Run `exp` and return the actions played by the environments at every step (one row per environment) and the
    rows written in the replay buffer."""
    from sheeprl.cli import run

    played: List[np.ndarray] = []
    rows: List[Dict[str, np.ndarray]] = []
    env_step = MultiDiscreteDummyEnv.step
    buffer_add = EnvIndependentReplayBuffer.add

    def recording_step(self, action):
        # The vectorized environment steps its environments in order: a new step every `NUM_ENVS` calls
        if len(played) == 0 or len(played[-1]) == NUM_ENVS:
            played.append([])
        played[-1].append(np.array(action))
        return env_step(self, action)

    def recording_add(self, data, *args, **kwargs):
        rows.append(copy.deepcopy({k: np.asarray(v) for k, v in data.items()}))
        return buffer_add(self, data, *args, **kwargs)

    root_dir = f"pytest_actions_{exp}"
    argv = [os.path.join(ROOT_DIR, "__main__.py"), *ARGS, f"exp={exp}", *args, f"root_dir={root_dir}"]
    try:
        with (
            mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
            mock.patch.object(sys, "argv", argv),
            mock.patch("sheeprl.utils.env.get_dummy_env", lambda id: MultiDiscreteDummyEnv(action_dims=ACTION_DIMS)),
            mock.patch.object(MultiDiscreteDummyEnv, "step", recording_step),
            mock.patch.object(EnvIndependentReplayBuffer, "add", recording_add),
        ):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    return [np.stack(step) for step in played], rows


def stored_actions(rows: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
    """The actions of the rows of every environment, decoded from their one-hots (the rows of the first observations,
    without an action, skipped)."""
    decoded = []
    for row in rows:
        actions = row["actions"].reshape(NUM_ENVS, sum(ACTION_DIMS))
        if not actions.any():
            continue
        assert actions.sum(-1).tolist() == [len(ACTION_DIMS)] * NUM_ENVS
        one_hots = np.split(actions, np.cumsum(ACTION_DIMS)[:-1], axis=-1)
        decoded.append(np.stack([one_hot.argmax(-1) for one_hot in one_hots], -1))
    return decoded


@pytest.mark.parametrize("exp", EXPERIMENTS)
@pytest.mark.parametrize("policy", [False, True])
def test_the_stored_actions_are_the_played_ones(exp, policy):
    # The random actions of the first steps were one-hot encoded after `reshape(n_actions, -1)`, which mixes the
    # environments; DreamerV1 and its exploration joined the policy's discrete actions with `cat`, mixing the
    # environments they were played in
    played, rows = play(exp, ["algo.learning_starts=0"] if policy else ["algo.learning_starts=24"])
    assert len(played) == 6
    stored = stored_actions(rows)
    # DreamerV1 and V2 store every action in the row of the observation it led to: the last one is not stored yet
    assert len(stored) >= len(played) - 1
    for step, (played_actions, stored_actions_of_step) in enumerate(zip(played, stored)):
        np.testing.assert_array_equal(stored_actions_of_step, played_actions, err_msg=f"step {step}")
