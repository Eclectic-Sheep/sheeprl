"""The SAC collectors preserve timeout observations when next states are sampled from adjacent replay rows."""

import glob
import os
import sys
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from sheeprl import ROOT_DIR
from sheeprl.cli import run
from sheeprl.data.buffers import ReplayBuffer


class TimeoutEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(0, 2, (1,), np.float32), "rgb": gym.spaces.Box(0, 255, (3, 64, 64), np.uint8)}
        )
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def observation(self):
        return {
            "state": np.array([self.step_count], np.float32),
            "rgb": np.full((3, 64, 64), 50 * self.step_count, np.uint8),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self.observation(), {}

    def step(self, action):
        self.step_count += 1
        return self.observation(), 1.0, False, False, {}


@pytest.mark.parametrize("algo,frames", [("sac", 1), ("droq", 1), ("sac_ae", 1), ("sac_ae", 3)])
@pytest.mark.parametrize("sample_next_obs,resume", [(False, False), (True, False), (True, True)])
def test_training_bootstraps_from_final_observations(monkeypatch, algo, frames, sample_next_obs, resume):
    env_id = "SheepRLTimeoutTest-v0"
    # Register only for this test, without leaking a new environment into the following tests.
    monkeypatch.setitem(
        gym.envs.registry,
        env_id,
        gym.envs.registration.EnvSpec(id=env_id, entry_point=TimeoutEnv, max_episode_steps=2),
    )
    original_sample = ReplayBuffer.sample_tensors
    sampled_timeouts = []

    def sample(buffer, *args, **kwargs):
        data = original_sample(buffer, *args, **kwargs)
        next_key = "next_state" if algo == "sac_ae" else "next_observations"
        # DroQ's actor-only batch needs no next observations.
        if next_key in data:
            timeout = data["truncated"].bool().squeeze(-1)
            if timeout.any():
                sampled_timeouts.append(int(timeout.sum()))
                obs_key = "state" if algo == "sac_ae" else "observations"
                # TimeLimit ends at 2; a checkpoint saved mid-episode ends at 1. Neither may bootstrap from reset 0.
                assert (data[next_key][timeout] == data[obs_key][timeout] + 1).all()
                assert not data["terminated"][timeout].any()
                if algo == "sac_ae":
                    expected_pixel = 50 * data[next_key][timeout].reshape(-1, 1, 1)
                    assert (data["next_rgb"][timeout][:, -1] == expected_pixel).all()
            if sample_next_obs:
                assert next_key not in buffer.buffer
        return data

    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        f"exp={algo}",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
        f"env.id={env_id}",
        "env.num_envs=2",
        "env.sync_env=True",
        "env.capture_video=False",
        "env.screen_size=64",
        f"env.frame_stack={frames}",
        "fabric.devices=1",
        "fabric.accelerator=cpu",
        "metric.log_level=0",
        "model_manager.disabled=True",
        "algo.run_test=False",
        "algo.learning_starts=4",
        f"algo.total_steps={10 if resume else 12}",
        "algo.replay_ratio=1",
        "algo.per_rank_batch_size=2",
        "algo.per_rank_pretrain_steps=0",
        "algo.hidden_size=8",
        "buffer.size=16",
        "buffer.memmap=True",
        f"buffer.sample_next_obs={sample_next_obs}",
        f"checkpoint.save_last={resume}",
        "root_dir=pytest_timeout",
        "run_name=first",
    ]
    if algo == "sac_ae":
        argv.extend(["algo.cnn_keys.encoder=[rgb]", "algo.dense_units=8", "algo.encoder.features_dim=8"])
    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(ReplayBuffer, "sample_tensors", sample),
    ):
        run()
        if resume:
            (checkpoint,) = glob.glob("logs/runs/pytest_timeout/first/version_*/checkpoint/*.ckpt")
            with mock.patch.object(
                sys, "argv", [*argv, "algo.total_steps=18", "run_name=resumed", f"checkpoint.resume_from={checkpoint}"]
            ):
                run()
    assert sum(sampled_timeouts) > 0
