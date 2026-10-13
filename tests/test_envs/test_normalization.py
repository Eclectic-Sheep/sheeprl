import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from lightning import Fabric

from sheeprl.envs.normalization import ObsNormalizer, RewardNormalizer, RunningMeanStd
from sheeprl.utils.utils import dotdict

OBS_SPACE = gym.spaces.Dict(
    {
        "state": gym.spaces.Box(-np.inf, np.inf, (3,), np.float64),
        "rgb": gym.spaces.Box(0, 255, (3, 4, 4), np.uint8),
    }
)


def normalization_cfg(**normalization):
    return dotdict(
        {
            "env": {"normalization": {"clip_obs": 10.0, "clip_rewards": 10.0, "epsilon": 1e-8, **normalization}},
            "algo": {"gamma": 0.9, "mlp_keys": {"encoder": ["state", "missing"]}},
        }
    )


def test_running_mean_std_is_the_mean_and_variance_of_all_the_batches():
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, size=(n, 2)) for n in (1, 5, 17)]
    stats = RunningMeanStd(shape=(2,))
    for batch in batches:
        stats.update(batch)
    samples = np.concatenate(batches)
    # The initial statistics weigh as `epsilon` samples
    assert np.allclose(stats.mean, samples.mean(axis=0), rtol=1e-4)
    assert np.allclose(stats.var, samples.var(axis=0), rtol=1e-4)
    assert stats.count == pytest.approx(len(samples) + 1e-4)


def process_batches(rank):
    """The batches of the process `rank` in the two rollouts, of different sizes in every process."""
    rng = np.random.default_rng(rank)
    return [[rng.normal(rank, 1 + rank, size=(n, 2)) for n in sizes] for sizes in ([1, 3], [2 + rank])]


def sync_on_two_processes(fabric):
    stats = RunningMeanStd(shape=(2,))
    first, second = process_batches(fabric.global_rank)
    for batch in first:
        stats.update(batch)
    stats.sync(fabric)
    synced_count = stats.count
    # Between the syncs, every process adds its own batches to the statistics of all of them
    for batch in second:
        stats.update(batch)
    own = stats.count == synced_count + sum(len(batch) for batch in second)
    stats.sync(fabric)
    moments = torch.as_tensor(np.concatenate([stats.mean, stats.var, [stats.count]]))
    gathered = fabric.all_gather(moments)
    samples = np.concatenate([batch for rank in range(2) for rollout in process_batches(rank) for batch in rollout])
    return (
        own
        and torch.equal(gathered[0], gathered[1])
        and np.allclose(stats.mean, samples.mean(axis=0), rtol=1e-4)
        and np.allclose(stats.var, samples.var(axis=0), rtol=1e-4)
        and stats.count == pytest.approx(len(samples) + 1e-4)
    )


@pytest.mark.skipif(sys.platform == "win32", reason="2 processes")
def test_running_mean_std_syncs_the_batches_of_all_the_processes():
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    assert fabric.launch(sync_on_two_processes)


def test_running_mean_std_of_one_process_does_not_change_when_synced():
    stats = RunningMeanStd(shape=(2,))
    stats.update(np.arange(6.0).reshape(3, 2))
    before = stats.state_dict()
    stats.sync(SimpleNamespace(world_size=1))
    assert np.array_equal(stats.mean, before["mean"]) and np.array_equal(stats.var, before["var"])
    assert stats.count == before["count"]


def test_obs_normalizer_standardizes_and_clips_only_its_keys():
    rng = np.random.default_rng(0)
    obs = {"state": rng.normal(5.0, 3.0, size=(1000, 3)), "rgb": rng.integers(0, 255, (1000, 3, 4, 4), np.uint8)}
    normalizer = ObsNormalizer(OBS_SPACE, ["state"])
    normalizer.update(obs)
    normalized = normalizer.normalize(obs)
    assert normalized["state"].dtype == np.float32
    assert np.allclose(normalized["state"].mean(axis=0), 0, atol=1e-3)
    assert np.allclose(normalized["state"].std(axis=0), 1, atol=1e-3)
    assert normalized["rgb"] is obs["rgb"]
    # The observations of one environment, without the rows, are normalized as the rows are
    assert np.array_equal(normalizer.normalize({"state": obs["state"][7]})["state"], normalized["state"][7])
    normalizer.clip = 1.0
    assert np.abs(normalizer.normalize(obs)["state"]).max() == 1.0


def test_obs_normalizer_restores_its_statistics():
    normalizer = ObsNormalizer(OBS_SPACE, ["state"])
    normalizer.update({"state": np.arange(12.0).reshape(4, 3)})
    restored = ObsNormalizer(OBS_SPACE, ["state"])
    restored.load_state_dict(normalizer.state_dict())
    obs = {"state": np.ones((2, 3))}
    assert np.array_equal(restored.normalize(obs)["state"], normalizer.normalize(obs)["state"])


def test_reward_normalizer_divides_the_rewards_by_the_std_of_the_discounted_returns():
    rng = np.random.default_rng(0)
    gamma, num_envs = 0.9, 3
    normalizer = RewardNormalizer(num_envs, gamma)
    returns, all_returns = np.zeros(num_envs), []
    for t in range(20):
        rewards = rng.normal(1.0, 5.0, size=num_envs)
        # The episode of the second environment ends every 4 steps
        dones = np.array([False, t % 4 == 3, False])
        normalized = normalizer(rewards, dones)
        returns = returns * gamma + rewards
        all_returns.append(returns.copy())
        expected = np.clip(rewards / np.sqrt(np.var(np.concatenate(all_returns)) + 1e-8), -10, 10)
        assert np.allclose(normalized, expected, rtol=1e-4)
        returns[dones] = 0
        assert np.array_equal(normalizer.returns, returns)
    assert normalized.dtype == np.float64
    assert normalizer(np.ones(num_envs, dtype=np.float32), np.zeros(num_envs, bool)).dtype == np.float32


def test_normalizers_are_created_as_configured():
    assert ObsNormalizer.from_config(normalization_cfg(obs=False), OBS_SPACE) is None
    assert RewardNormalizer.from_config(normalization_cfg(rewards=False), 2) is None
    # A configuration saved before the option doesn't normalize
    old_cfg = normalization_cfg()
    old_cfg.env.pop("normalization")
    assert ObsNormalizer.from_config(old_cfg, OBS_SPACE) is None
    assert RewardNormalizer.from_config(old_cfg, 2) is None
    # Only the vector observations the agent encodes, among the ones of the environment
    assert list(ObsNormalizer.from_config(normalization_cfg(obs=True), OBS_SPACE).stats) == ["state"]
    reward_normalizer = RewardNormalizer.from_config(normalization_cfg(rewards=True), 2)
    assert reward_normalizer.gamma == 0.9 and len(reward_normalizer.returns) == 2
