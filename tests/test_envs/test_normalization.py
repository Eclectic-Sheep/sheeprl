import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from lightning import Fabric

from sheeprl.transforms import ScaleByReturnStd, Tanh
from sheeprl.transforms.observations import RowsTransforms
from sheeprl.transforms.rewards import REWARD_SPACE, reward_pipeline
from sheeprl.transforms.stats import RunningMeanStd
from sheeprl.utils.utils import dotdict
from tests.test_transforms import oracles

OBS_SPACE = gym.spaces.Dict(
    {
        "state": gym.spaces.Box(-np.inf, np.inf, (3,), np.float64),
        "rgb": gym.spaces.Box(0, 255, (3, 4, 4), np.uint8),
    }
)


def normalization_cfg(obs=False, clip=10.0, rewards=False):
    transforms = [{"_target_": "sheeprl.transforms.Standardize", "clip": clip, "epsilon": 1e-8}] if obs else []
    reward_transforms = [{"_target_": "sheeprl.transforms.ScaleByReturnStd", "gamma": 0.9}] if rewards else []
    return dotdict(
        {
            "algo": {
                "rewards": {"transforms": reward_transforms},
                "gamma": 0.9,
                "inputs": {"vector": {"keys": ["state"], "transforms": transforms}, "image": {"keys": ["rgb"]}},
            },
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
    assert np.allclose(stats.mean.numpy(), samples.mean(axis=0), rtol=1e-4)
    assert np.allclose(stats.var.numpy(), samples.var(axis=0), rtol=1e-4)
    assert float(stats.count) == pytest.approx(len(samples) + 1e-4)


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
    synced_count = float(stats.count)
    # Between the syncs, every process adds its own batches to the statistics of all of them
    for batch in second:
        stats.update(batch)
    own = float(stats.count) == synced_count + sum(len(batch) for batch in second)
    stats.sync(fabric)
    gathered = fabric.all_gather(torch.cat([stats.mean, stats.var, stats.count[None]]))
    samples = np.concatenate([batch for rank in range(2) for rollout in process_batches(rank) for batch in rollout])
    return (
        own
        and torch.equal(gathered[0], gathered[1])
        and np.allclose(stats.mean.numpy(), samples.mean(axis=0), rtol=1e-4)
        and np.allclose(stats.var.numpy(), samples.var(axis=0), rtol=1e-4)
        and float(stats.count) == pytest.approx(len(samples) + 1e-4)
    )


@pytest.mark.skipif(sys.platform == "win32", reason="2 processes")
def test_running_mean_std_syncs_the_batches_of_all_the_processes():
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    assert fabric.launch(sync_on_two_processes)


def test_running_mean_std_of_one_process_does_not_change_when_synced():
    stats = RunningMeanStd(shape=(2,))
    stats.update(np.arange(6.0).reshape(3, 2))
    before = {k: v.clone() for k, v in stats.state_dict().items()}
    stats.sync(SimpleNamespace(world_size=1))
    assert all(torch.equal(v, before[k]) for k, v in stats.state_dict().items())


def test_the_rows_stage_standardizes_and_clips_only_the_keys_of_its_input():
    rng = np.random.default_rng(0)
    obs = {"state": rng.normal(5.0, 3.0, size=(1000, 3)), "rgb": rng.integers(0, 255, (1000, 3, 4, 4), np.uint8)}
    transforms = RowsTransforms.from_config(normalization_cfg(obs=True), OBS_SPACE)
    normalized = transforms.rows(obs)
    assert normalized["state"].dtype == np.float32
    assert np.allclose(normalized["state"].mean(axis=0), 0, atol=1e-3)
    assert np.allclose(normalized["state"].std(axis=0), 1, atol=1e-3)
    assert normalized["rgb"] is obs["rgb"]
    # The observation of one environment, without the rows, is transformed as the rows are
    assert np.array_equal(transforms.one({"state": obs["state"][7]})["state"], normalized["state"][7])
    clipped = RowsTransforms.from_config(normalization_cfg(obs=True, clip=1.0), OBS_SPACE)
    assert np.abs(clipped.rows(obs)["state"]).max() == 1.0


def test_the_rows_stage_restores_its_statistics():
    transforms = RowsTransforms.from_config(normalization_cfg(obs=True), OBS_SPACE)
    transforms.rows({"state": np.arange(12.0).reshape(4, 3)})
    restored = RowsTransforms.from_config(normalization_cfg(obs=True), OBS_SPACE)
    restored.load_key_state_dict(transforms.key_state_dict())
    obs = {"state": np.ones((2, 3))}
    assert np.array_equal(restored.rows(obs, update=False)["state"], transforms.rows(obs, update=False)["state"])
    # Every key with statistics must find its own
    with pytest.raises(KeyError, match="state"):
        restored.load_key_state_dict({})


def test_scale_by_return_std_divides_the_rewards_by_the_std_of_the_discounted_returns():
    rng = np.random.default_rng(0)
    gamma, num_envs = 0.9, 3
    oracle = oracles.RewardNormalizer(num_envs, gamma)
    transform = ScaleByReturnStd(gamma)
    transform.space(REWARD_SPACE)
    returns, all_returns = np.zeros(num_envs), []
    for t in range(20):
        rewards = rng.normal(1.0, 5.0, size=num_envs)
        # The episode of the second environment ends every 4 steps
        dones = np.array([False, t % 4 == 3, False])
        normalized = transform.rows(rewards, dones=dones)
        # Bit for bit the former `RewardNormalizer`
        assert np.array_equal(normalized, oracle(rewards, dones))
        returns = returns * gamma + rewards
        all_returns.append(returns.copy())
        expected = np.clip(rewards / np.sqrt(np.var(np.concatenate(all_returns)) + 1e-8), -10, 10)
        assert np.allclose(normalized, expected, rtol=1e-4)
        returns[dones] = 0
        assert np.array_equal(transform.returns, returns)
    assert normalized.dtype == np.float64
    assert transform.rows(np.ones(num_envs, dtype=np.float32), dones=np.zeros(num_envs, bool)).dtype == np.float32
    # The returns are episode state: only the statistics are saved
    assert sorted(transform.state_dict()) == ["stats.count", "stats.mean", "stats.var"]


def test_the_transforms_are_created_as_configured():
    # Without a `Standardize` or a `ScaleByReturnStd`, nothing has statistics
    assert RowsTransforms.from_config(normalization_cfg(obs=False), OBS_SPACE).key_state_dict() == {}
    assert len(reward_pipeline(normalization_cfg(rewards=False)).transforms("rows")) == 0
    # Only the vector observations the agent encodes, among the ones of the environment
    assert list(RowsTransforms.from_config(normalization_cfg(obs=True), OBS_SPACE).key_state_dict()) == ["state"]
    (scale,) = reward_pipeline(normalization_cfg(rewards=True)).transforms("rows")
    assert isinstance(scale, ScaleByReturnStd) and scale.gamma == 0.9
    # The statistics of the data in a replay buffer: an error for the off-policy algorithms
    with pytest.raises(ValueError, match="only by on-policy algorithms"):
        RowsTransforms.from_config(normalization_cfg(obs=True), OBS_SPACE, off_policy=True)
    with pytest.raises(ValueError, match="only by on-policy algorithms"):
        reward_pipeline(normalization_cfg(rewards=True), off_policy=True)
    # The stateless ones are for every algorithm, at `rows` only
    cfg = normalization_cfg()
    cfg.algo.rewards.transforms = [{"_target_": "sheeprl.transforms.Tanh"}]
    assert isinstance(reward_pipeline(cfg, off_policy=True).transforms("rows")[0], Tanh)
    cfg.algo.rewards.transforms = [{"_target_": "sheeprl.transforms.Tanh", "where": "batch"}]
    with pytest.raises(ValueError, match="run at `rows`"):
        reward_pipeline(cfg)
