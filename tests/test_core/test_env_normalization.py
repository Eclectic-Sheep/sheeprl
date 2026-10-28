"""The normalization of the observations (the `Standardize` transform) and of the rewards (the `ScaleByReturnStd`
transform): the environments normalize their steps, the checkpoints save the statistics, a resumed run goes on from
them and the evaluation plays with them."""

from __future__ import annotations

import glob
import os
import shutil
import sys
from contextlib import contextmanager
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from sheeprl import ROOT_DIR
from sheeprl.core import Episode, GymEnvironment, loop
from sheeprl.transforms import Pipeline, ScaleByReturnStd, Standardize, Tanh
from sheeprl.transforms.observations import EnvsTransforms
from sheeprl.transforms.rewards import REWARD_SPACE
from sheeprl.utils.env import get_vector_env_cls
from tests.test_transforms import oracles

COMMON_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=0",
    "metric.disable_timer=True",
    "checkpoint.save_last=True",
]

# A small PPO on the vector observations of the dummy environment: 4 steps of 2 environments per iteration
PPO_ARGS = [
    "exp=ppo",
    "env=dummy",
    "env.id=continuous_dummy",
    "algo.inputs.image.keys=[]",
    "algo.inputs.vector.keys=[state]",
    "algo.rollout_steps=4",
    "algo.per_rank_batch_size=4",
    "algo.inputs.vector.transforms=[{_target_:sheeprl.transforms.Standardize}]",
    "algo.rewards.transforms=[{_target_:sheeprl.transforms.ScaleByReturnStd,gamma:${algo.gamma}}]",
]


class TwoStepEnv(gym.Env):
    """Episodes of two steps, each with reward 1; the observation is the step of the episode."""

    observation_space = gym.spaces.Dict({"state": gym.spaces.Box(0, 2, (1,), np.float32)})
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return {"state": np.zeros(1, np.float32)}, {}

    def step(self, action):
        self.t += 1
        return {"state": np.full(1, self.t, np.float32)}, 1.0, self.t == 2, False, {}


def two_step_environment() -> GymEnvironment:
    envs = get_vector_env_cls(True)([lambda: gym.wrappers.RecordEpisodeStatistics(TwoStepEnv())] * 2)
    pipeline = Pipeline([Standardize()], envs.single_observation_space["state"], "envs", "envs")
    rewards = Pipeline([ScaleByReturnStd(gamma=0.5)], REWARD_SPACE, "envs", "envs")
    return GymEnvironment(envs, seed=0, transforms=EnvsTransforms({"state": pipeline}), rewards=rewards)


def standardize_of(env: GymEnvironment) -> Standardize:
    return env.transforms.pipelines[0].stages["envs"][0]


def count_of(env_state, key: str = "state") -> float:
    """The count of the statistics of `Standardize` of `key` in the state of the environments of a checkpoint."""
    return float(env_state["transforms"][key]["stages.envs.0.stats.count"])


def rewards_count_of(env_state) -> float:
    """The count of the statistics of `ScaleByReturnStd` in the state of the environments of a checkpoint."""
    return float(env_state["rewards"]["stages.envs.0.stats.count"])


def test_gym_environment_normalizes_the_observations_and_the_rewards_of_its_steps():
    env = two_step_environment()
    first_obs = env.reset()
    first = env.step(np.zeros(2, dtype=np.int64))
    second = env.step(np.zeros(2, dtype=np.int64))
    # The statistics of the observations played: 0 (reset), 1, 0 (the next episodes), not the final 2
    stats = standardize_of(env).stats
    assert float(stats.count) == pytest.approx(6 + 1e-4)
    assert float(stats.mean) == pytest.approx(1 / 3, rel=1e-4) and float(stats.var) == pytest.approx(2 / 9, rel=1e-4)
    # The steps hold the observations normalized with the statistics of their time
    assert first.obs is first_obs and second.obs is first.next_obs
    frozen = lambda x: env.transforms.envs({"state": x}, update=False)["state"]  # noqa: E731
    assert np.array_equal(second.next_obs["state"], frozen(np.zeros((2, 1))))
    assert np.array_equal(second.stack_final_obs([0, 1], ["state"])["state"], frozen(np.full((2, 1), 2.0)))
    # The discounted returns are 1 and 1.5: the reward of the second step is 1 / std([1, 1, 1.5, 1.5]) (up to the
    # initial statistics)
    assert np.allclose(second.rewards, 1 / 0.25, rtol=1e-3)
    # The ended episodes keep the rewards of the environment, and the returns start again
    assert second.episodes == (Episode(0, 2.0, 2), Episode(1, 2.0, 2))
    assert np.array_equal(env.rewards.stages["envs"][0].returns, np.zeros(2))
    state = env.state_dict()
    env.close()

    restored = two_step_environment()
    restored.load_state_dict(state)
    assert float(standardize_of(restored).stats.count) == float(stats.count)
    assert float(restored.rewards.stages["envs"][0].stats.count) == pytest.approx(4 + 1e-4)
    restored.close()


def run_cli(argv, devices: int = 1) -> None:
    from sheeprl.cli import run

    with mock.patch.dict(os.environ, {"LT_DEVICES": str(devices)}), mock.patch.object(sys, "argv", argv):
        run()


def last_checkpoint(root_dir: str, run_name: str) -> str:
    (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, run_name, "version_*", "checkpoint", "*.ckpt"))
    return ckpt_path


def test_the_statistics_of_the_normalization_are_resumed_and_evaluated():
    from sheeprl.cli import evaluation

    root_dir = "pytest_env_normalization"
    argv = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *PPO_ARGS, f"root_dir={root_dir}"]
    try:
        # 2 iterations, then 2 more: every iteration adds 4 observations and rewards of the 2 environments, and the
        # first observations of the resets of the 2 runs are added too
        with mock.patch.object(Standardize, "envs", autospec=True, side_effect=Standardize.envs) as norm:
            run_cli([*argv, "algo.total_steps=16", "run_name=first", "algo.run_test=True"])
        first = torch.load(last_checkpoint(root_dir, "first"), weights_only=False)["env"]
        assert count_of(first) == pytest.approx(2 + 16 + 1e-4)
        assert rewards_count_of(first) == pytest.approx(16 + 1e-4)
        # The test episode plays with the statistics of the training
        assert float(norm.call_args.args[0].stats.count) == pytest.approx(2 + 16 + 1e-4)
        assert norm.call_args.kwargs["update"] is False

        run_cli(
            [
                *argv,
                "algo.total_steps=32",
                "run_name=resumed",
                "algo.run_test=False",
                f"checkpoint.resume_from={last_checkpoint(root_dir, 'first')}",
            ]
        )
        ckpt_path = last_checkpoint(root_dir, "resumed")
        resumed = torch.load(ckpt_path, weights_only=False)["env"]
        assert count_of(resumed) == pytest.approx(2 * (2 + 16) + 1e-4)
        assert rewards_count_of(resumed) == pytest.approx(2 * 16 + 1e-4)

        # The evaluation normalizes with the statistics of the checkpoint, without changing them
        with (
            mock.patch.object(Standardize, "envs", autospec=True, side_effect=Standardize.envs) as norm,
            mock.patch.object(
                sys, "argv", ["sheeprl_eval.py", f"checkpoint_path={ckpt_path}", "env.capture_video=False"]
            ),
        ):
            evaluation()
        assert norm.called and all(call.kwargs["update"] is False for call in norm.call_args_list)
        evaluated = norm.call_args.args[0].stats
        assert float(evaluated.count) == count_of(resumed)
        assert torch.equal(evaluated.mean, resumed["transforms"]["state"]["stages.envs.0.stats.mean"])
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)


@pytest.mark.skipif(sys.platform == "win32", reason="2 processes")
def test_the_checkpoint_has_the_statistics_of_all_the_processes():
    # 2 processes, 2 iterations: every process adds the first observations of its 2 environments and 16 more, and 16
    # rewards. The checkpoint, saved by the first process, has the statistics of both
    root_dir = "pytest_env_normalization_processes"
    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *[arg for arg in COMMON_ARGS if not arg.startswith("fabric.devices")],
        "fabric.devices=2",
        *PPO_ARGS,
        "algo.total_steps=32",
        "algo.run_test=False",
        f"root_dir={root_dir}",
        "run_name=run",
    ]
    try:
        run_cli(argv, devices=2)
        state = torch.load(last_checkpoint(root_dir, "run"), weights_only=False)["env"]
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert count_of(state) == pytest.approx(2 * (2 + 16) + 1e-4)
    assert rewards_count_of(state) == pytest.approx(2 * 16 + 1e-4)


def test_the_sync_of_the_environments_is_not_timed_with_the_interaction(monkeypatch):
    # A process waits in the sync for the rollouts of the others: timed with the interaction, the wait lowered the
    # logged speed of the interaction (`Time/sps_env_interaction`, from the timers of the first process)
    open_phases, phases_at_sync = [], []
    phase_timer = loop.phase_timer

    @contextmanager
    def tracked_phase_timer(name):
        open_phases.append(name)
        try:
            with phase_timer(name):
                yield
        finally:
            open_phases.remove(name)

    monkeypatch.setattr(loop, "phase_timer", tracked_phase_timer)
    monkeypatch.setattr(GymEnvironment, "sync", lambda self, fabric: phases_at_sync.append(list(open_phases)))
    root_dir = "pytest_env_normalization_timer"
    argv = [os.path.join(ROOT_DIR, "__main__.py"), *COMMON_ARGS, *PPO_ARGS, "dry_run=True", f"root_dir={root_dir}"]
    try:
        run_cli(argv)
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    # One iteration (dry run), synced outside every phase
    assert phases_at_sync == [[]]


@pytest.mark.parametrize(
    "transforms",
    [
        "algo.inputs.vector.transforms=[{_target_:sheeprl.transforms.Standardize}]",
        "algo.rewards.transforms=[{_target_:sheeprl.transforms.ScaleByReturnStd,gamma:0.9}]",
    ],
)
def test_off_policy_algorithms_reject_the_normalization(transforms):
    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *COMMON_ARGS,
        "exp=sac",
        "env.id=Pendulum-v1",
        "dry_run=True",
        transforms,
        "root_dir=pytest_env_normalization_sac",
    ]
    try:
        with pytest.raises(ValueError, match="only by on-policy algorithms"):
            run_cli(argv)
    finally:
        shutil.rmtree(os.path.join("logs", "runs", "pytest_env_normalization_sac"), ignore_errors=True)


def test_the_rewards_are_normalized_then_squashed_in_the_environments():
    # The former `env.normalization.rewards` and `env.clip_rewards`: the normalizer ran in the environments, the tanh
    # in the writers, after it
    env = two_step_environment()
    env.rewards = Pipeline([ScaleByReturnStd(gamma=0.5), Tanh()], REWARD_SPACE, "envs", "envs")
    oracle = oracles.RewardNormalizer(2, gamma=0.5)
    env.reset()
    for _ in range(4):
        step = env.step(np.zeros(2, dtype=np.int64))
        dones = step.terminated | step.truncated
        assert np.array_equal(step.rewards, np.tanh(oracle(np.ones(2), dones)))
    env.close()


def test_ppo_bootstraps_the_transformed_rewards():
    # The writer of PPO bootstraps the episodes truncated by the time limit with the rewards of the steps, so after
    # their transforms: normalized, squashed, then bootstrapped
    from sheeprl.algos.ppo import ppo
    from sheeprl.core import environment

    stepped, bootstrapped = [], []
    gym_step, bootstrap = environment.GymEnvironment.step, ppo.bootstrap_truncated

    def recording_step(self, actions):
        step = gym_step(self, actions)
        stepped.append((step.rewards.copy(), step.truncated.copy()))
        return step

    def recording_bootstrap(rewards, *args, **kwargs):
        bootstrapped.append(np.array(rewards, copy=True))
        return bootstrap(rewards, *args, **kwargs)

    root_dir = "pytest_env_normalization_bootstrap"
    argv = [
        os.path.join(ROOT_DIR, "__main__.py"),
        *COMMON_ARGS,
        "exp=ppo",
        "env.id=CartPole-v1",
        "env.max_episode_steps=3",
        "algo.rollout_steps=8",
        "algo.per_rank_batch_size=8",
        "algo.total_steps=16",
        "algo.run_test=False",
        "algo.rewards.transforms=[{_target_:sheeprl.transforms.ScaleByReturnStd,gamma:0.9},{_target_:sheeprl.transforms.Tanh}]",
        f"root_dir={root_dir}",
        "run_name=run",
    ]
    try:
        with (
            mock.patch.object(environment.GymEnvironment, "step", recording_step),
            mock.patch.object(ppo, "bootstrap_truncated", recording_bootstrap),
        ):
            run_cli(argv)
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert len(stepped) == len(bootstrapped) == 8 and any(truncated.any() for _, truncated in stepped)
    # The rewards of CartPole are 1: squashed, they are below
    assert all(np.abs(rewards).max() < 1 for rewards, _ in stepped)
    assert all(np.array_equal(rewards, given) for (rewards, _), given in zip(stepped, bootstrapped))
