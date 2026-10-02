"""The rollouts and the losses of PPO and A2C: bootstrapped rewards, value loss, advantage normalization, buffer
size."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning import Fabric

from sheeprl.algos.a2c.a2c import A2C
from sheeprl.algos.a2c.a2c import RolloutPlayer as A2CRolloutPlayer
from sheeprl.algos.ppo.agent import PPOPlayer
from sheeprl.algos.ppo.loss import value_loss
from sheeprl.algos.ppo.ppo import PPO
from sheeprl.algos.ppo.ppo import RolloutPlayer as PPORolloutPlayer
from sheeprl.core import EnvStep, Rollout
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.utils import dotdict, normalize_tensor
from tests.test_algos.test_ppo_agent import build_agent


def config(clip_rewards: bool = False, buffer_size: int = 4) -> dotdict:
    return dotdict(
        {
            "algo": {
                "cnn_keys": {"encoder": []},
                "mlp_keys": {"encoder": ["state"]},
                "gamma": 0.9,
                "rollout_steps": 4,
            },
            "env": {"clip_rewards": clip_rewards, "wrapper": {"_target_": "gymnasium.make"}},
            "buffer": {"size": buffer_size, "memmap": False, "validate_args": False},
        }
    )


@pytest.mark.parametrize("player_cls", [PPORolloutPlayer, A2CRolloutPlayer])
@pytest.mark.parametrize("clip_rewards", [False, True])
def test_rollout_bootstraps_only_the_truncated_episodes(player_cls, clip_rewards):
    # Three envs end their episode with reward 3: truncated by the time limit, truncated and terminated in the same
    # step, terminated. Only the first one goes on in the MDP: its reward gets gamma times the value (2) of its final
    # observation, added after the clipping of the reward (PPO; A2C doesn't clip)
    num_envs = 3
    obs = {"state": np.zeros((num_envs, 8), dtype=np.float32)}
    step = EnvStep(
        obs=obs,
        next_obs={"state": np.ones((num_envs, 8), dtype=np.float32)},
        rewards=np.full(num_envs, 3.0),
        terminated=np.array([False, True, True]),
        truncated=np.array([True, True, False]),
        info={"final_obs": np.array([{"state": np.full(8, 5, dtype=np.float32)}] * num_envs, dtype=object)},
    )
    env = SimpleNamespace(num_envs=num_envs, obs=obs, step=lambda actions: step)
    agent = build_agent(ortho_init=False)
    policy = PPOPlayer(agent.feature_extractor, agent.actor, agent.critic)
    policy.get_values = lambda obs: torch.full((len(obs["state"]), 1), 2.0)
    cfg = config(clip_rewards=clip_rewards)
    rollout = Rollout(ReplayBuffer(1, num_envs, memmap=False, obs_keys=["state"]))
    with torch.no_grad():
        player_cls(Fabric(accelerator="cpu", devices=1), cfg, policy).step(env, rollout)

    reward = np.tanh(3.0) if clip_rewards and player_cls is PPORolloutPlayer else 3.0
    np.testing.assert_allclose(rollout.buffer["rewards"][0, :, 0], [reward + 0.9 * 2.0, reward, reward], rtol=1e-6)
    np.testing.assert_array_equal(rollout.buffer["dones"][0, :, 0], [1, 1, 1])


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


@pytest.mark.parametrize("algo_cls", [PPO, A2C])
def test_the_buffer_holds_one_rollout(algo_cls):
    # A larger buffer was accepted: its rows beyond the rollout were trained on, and the returns computed on the wrong
    # rows from the second iteration
    fabric = Fabric(accelerator="cpu", devices=1)
    algo_cls(fabric, config(buffer_size=4))
    with pytest.raises(ValueError, match="must be equal to the rollout steps"):
        algo_cls(fabric, config(buffer_size=8))
