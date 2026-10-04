"""PPO and A2C: the rewards of the rollout, the losses, the advantages."""

import glob
import importlib
import os
import shutil
import sys
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl import ROOT_DIR
from sheeprl.algos.a2c import a2c
from sheeprl.algos.ppo import ppo
from sheeprl.algos.ppo.loss import value_loss
from sheeprl.algos.ppo.utils import bootstrap_truncated
from sheeprl.utils import compile as compile_utils
from sheeprl.utils.utils import dotdict, normalize_tensor
from tests.test_algos.compiled import assert_same_step, no_host_reads, same_random_numbers


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


@pytest.mark.parametrize("exp", ["ppo", "ppo_recurrent"])
def test_a_resumed_run_anneals_with_its_own_total_steps(exp):
    # A run of 2 iterations resumed for 4: the learning rate and the clip coefficient of iterations 3 and 4 are the ones
    # of a run of 4 iterations. The scheduler of the checkpoint kept the horizon of 2 (learning rate 0 after it), and
    # the coefficients restarted from the configured ones
    from sheeprl.cli import run

    module = importlib.import_module(f"sheeprl.algos.{exp}.{exp}")
    root_dir = f"pytest_{exp}_anneal"
    args = [
        os.path.join(ROOT_DIR, "__main__.py"),
        f"exp={exp}",
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
    if exp == "ppo_recurrent":
        args.append("algo.per_rank_sequence_length=2")
    schedule = []

    # The values used by the training of every iteration
    algorithm = module.PPO if exp == "ppo" else module.PPORecurrent

    def recording_end_iteration(self, state, iteration):
        schedule.append((state.optimizer.param_groups[0]["lr"], self.cfg.algo.clip_coef))
        return end_iteration(self, state, iteration)

    end_iteration = algorithm.end_iteration
    try:
        with (
            mock.patch.dict(os.environ, {"LT_DEVICES": "1"}),
            mock.patch.object(algorithm, "end_iteration", recording_end_iteration),
        ):
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


def small_on_policy(algo, actions, overrides=(), accelerator="cpu", tmp_path="."):
    """PPO or A2C with small models, built as by the training, and a minibatch of 8 steps of vector observations."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"exp={algo}",
                "algo.mlp_keys.encoder=[state]",
                "algo.cnn_keys.encoder=[]",
                "algo.dense_units=8",
                *(["algo.normalize_advantages=True"] if algo == "ppo" else []),
                *overrides,
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    fabric = Fabric(accelerator=accelerator, devices=1)
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)})
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32) if actions == "continuous" else gym.spaces.Discrete(3)
    torch.manual_seed(0)
    algorithm = (ppo.PPO if algo == "ppo" else a2c.A2C)(fabric, cfg)
    state, _ = algorithm.build(obs_space, action_space, SimpleNamespace(total_iters=4), str(tmp_path))
    generator = torch.Generator().manual_seed(1)
    if actions == "continuous":
        taken = torch.randn(8, 2, generator=generator)
    else:
        taken = torch.nn.functional.one_hot(torch.randint(0, 3, (8,), generator=generator), 3).float()
    batch = {"state": torch.randn(8, 5, generator=generator), "actions": taken}
    for k in ("logprobs", "values", "returns", "advantages"):
        batch[k] = torch.randn(8, 1, generator=generator)
    return cfg, algorithm, state, {k: v.to(fabric.device) for k, v in batch.items()}


@pytest.mark.parametrize("actions", ["continuous", "discrete"])
@pytest.mark.parametrize("algo", ["ppo", "a2c"])
def test_the_on_policy_losses_compile_into_single_graphs_without_synchronizations(monkeypatch, tmp_path, algo, actions):
    # The normalization of the advantages selected them with a boolean mask: a shape that depends on the data, which
    # breaks the graph and synchronizes with the device
    cfg, algorithm, state, batch = small_on_policy(algo, actions, tmp_path=tmp_path)
    no_host_reads(monkeypatch)
    graph = lambda fn: torch.compile(fn, backend="eager", fullgraph=True)  # noqa: E731
    obs = {"state": batch["state"]}
    actions_dim = tuple(int(dim) for dim in state.agent.actions_dim)
    if algo == "ppo":
        graph(ppo.ppo_loss)(
            state.agent,
            obs,
            batch["actions"],
            batch["logprobs"],
            batch["values"],
            batch["returns"],
            batch["advantages"],
            algorithm.clip_coef,
            algorithm.ent_coef,
            actions_dim=actions_dim,
            normalize_advantages=True,
            vf_coef=cfg.algo.vf_coef,
            clip_vloss=True,
            reduction=cfg.algo.loss_reduction,
        )
    else:
        graph(a2c.a2c_loss)(
            state.agent,
            obs,
            batch["actions"],
            batch["values"],
            batch["returns"],
            batch["advantages"],
            actions_dim=actions_dim,
            normalize_advantages=True,
            vf_coef=cfg.algo.vf_coef,
            ent_coef=cfg.algo.ent_coef,
            reduction=cfg.algo.loss_reduction,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled losses are compared on the GPU")
@pytest.mark.parametrize("algo", ["ppo", "a2c"])
def test_the_compiled_on_policy_losses_are_the_ones_of_the_eager_losses(monkeypatch, tmp_path, algo):
    # The same weights, the same minibatch and the same random numbers: the same losses and gradients of a gradient
    # step, with and without `torch.compile` (and its CUDA graphs)
    same_random_numbers(monkeypatch)
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    results = []
    for enabled in (False, True):
        _, algorithm, state, batch = small_on_policy(
            algo, "continuous", [f"algo.compile.enabled={enabled}"], accelerator="cuda", tmp_path=tmp_path
        )
        torch.manual_seed(1)
        metrics = algorithm.train_step(state, batch if algo == "ppo" else [batch, batch], 0)
        losses = [(name, value.detach().clone().reshape(-1)) for name, value in metrics.items()]
        grads = [[None if p.grad is None else p.grad.detach().clone() for p in state.agent.parameters()]]
        results.append((losses, grads))
    assert_same_step(*results)
