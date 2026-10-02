"""PPO and A2C: the rewards of the rollout, the losses, the advantages."""

import glob
import os
import shutil
import sys
from unittest import mock

import numpy as np
import pytest
import torch

from sheeprl import ROOT_DIR
from sheeprl.algos.ppo.loss import value_loss
from sheeprl.algos.ppo.utils import bootstrap_truncated
from sheeprl.utils.utils import normalize_tensor


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_value_loss_has_the_same_scale_with_and_without_clipping(reduction):
    # When the new values stay within `clip_coef` of the old ones, clipping changes nothing: the two losses are equal
    # (the clipped one was halved and always averaged)
    torch.manual_seed(0)
    old_values = torch.randn(16, 1)
    new_values = old_values + 0.1 * torch.rand(16, 1)
    returns = torch.randn(16, 1)
    unclipped = value_loss(new_values, old_values, returns, 0.2, clip_vloss=False, reduction=reduction)
    clipped = value_loss(new_values, old_values, returns, 0.2, clip_vloss=True, reduction=reduction)
    torch.testing.assert_close(clipped, unclipped)


def test_normalize_tensor_of_one_element():
    # The last minibatch of a rollout can hold a single advantage: it has no standard deviation (it was NaN)
    torch.testing.assert_close(normalize_tensor(torch.tensor([[2.5]])), torch.tensor([[2.5]]))
    mask = torch.tensor([[True], [False]])
    torch.testing.assert_close(normalize_tensor(torch.tensor([[2.5], [1.0]]), mask=mask), torch.tensor([2.5]))
    normalized = normalize_tensor(torch.tensor([[1.0], [3.0]]))
    torch.testing.assert_close(normalized.mean(), torch.tensor(0.0))


@pytest.mark.parametrize("clip_rewards", [False, True])
def test_only_the_truncated_episodes_are_bootstrapped(clip_rewards):
    # Three environments: truncated, truncated and terminated in the same step (gymnasium's `TimeLimit` truncates also
    # when the termination falls on the last allowed step), terminated. Only the first one is bootstrapped, with the
    # value of its final observation added after the reward is clipped (it was clipped with it)
    rewards = np.full(3, 3.0)
    if clip_rewards:
        rewards = np.tanh(rewards)
    asked = []

    def final_values(env_idxes):
        asked.append(env_idxes.tolist())
        return np.full((len(env_idxes), 1), 2.0)

    terminated = np.array([False, True, True])
    truncated = np.array([True, True, False])
    bootstrapped = bootstrap_truncated(rewards, terminated, truncated, final_values, gamma=0.9)
    reward = np.tanh(3.0) if clip_rewards else 3.0
    np.testing.assert_allclose(bootstrapped, [reward + 0.9 * 2.0, reward, reward])
    assert asked == [[0]]
    # The rewards given are left as they are
    np.testing.assert_allclose(rewards, np.full(3, reward))


@pytest.mark.parametrize("exp", ["ppo", "a2c"])
def test_the_buffer_holds_one_rollout(exp):
    # A larger buffer was accepted: the training read its rows never written, and from the second rollout computed the
    # returns over rows of different rollouts
    from sheeprl.cli import run

    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        f"exp={exp}",
        "dry_run=True",
        "algo.rollout_steps=4",
        "buffer.size=8",
        "env.num_envs=1",
        "env.sync_env=True",
        "env.capture_video=False",
        "fabric.devices=1",
        "fabric.accelerator=cpu",
        "metric.log_level=0",
        "root_dir=pytest_ppo_buffer_size",
    ]
    try:
        with (
            mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
            mock.patch.object(sys, "argv", args),
            pytest.raises(ValueError, match=r"The size of the buffer \(8\) must be equal to the rollout steps \(4\)"),
        ):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", "pytest_ppo_buffer_size"), ignore_errors=True)


def test_a_resumed_run_anneals_with_its_own_total_steps():
    # A run of 2 iterations resumed for 4: the learning rate and the clip coefficient of iterations 3 and 4 are the ones
    # of a run of 4 iterations. The scheduler of the checkpoint kept the horizon of 2 (learning rate 0 after it), and
    # the coefficients restarted from the configured ones
    from sheeprl.algos.ppo import ppo
    from sheeprl.cli import run

    root_dir = "pytest_ppo_anneal"
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        "exp=ppo",
        "env.num_envs=2",
        "algo.rollout_steps=4",
        "algo.per_rank_batch_size=4",
        "algo.anneal_lr=True",
        "algo.anneal_clip_coef=True",
        "algo.optimizer.lr=1e-3",
        "algo.clip_coef=0.2",
        "algo.run_test=False",
        "env.sync_env=True",
        "env.capture_video=False",
        "fabric.devices=1",
        "fabric.accelerator=cpu",
        "metric.log_level=0",
        "checkpoint.save_last=True",
        f"root_dir={root_dir}",
    ]
    schedule = []

    def recording_train(fabric, agent, optimizer, data, aggregator, cfg):
        schedule.append((optimizer.param_groups[0]["lr"], cfg.algo.clip_coef))
        return ppo_train(fabric, agent, optimizer, data, aggregator, cfg)

    ppo_train = ppo.train
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(ppo, "train", recording_train):
            with mock.patch.object(sys, "argv", [*args, "algo.total_steps=16", "run_name=first"]):
                run()
            (ckpt_path,) = glob.glob(os.path.join("logs", "runs", root_dir, "first", "version_*", "checkpoint", "*"))
            with mock.patch.object(
                sys, "argv", [*args, "algo.total_steps=32", "run_name=resumed", f"checkpoint.resume_from={ckpt_path}"]
            ):
                run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    np.testing.assert_allclose(schedule, [(1e-3, 0.2), (5e-4, 0.1), (5e-4, 0.1), (2.5e-4, 0.05)])
