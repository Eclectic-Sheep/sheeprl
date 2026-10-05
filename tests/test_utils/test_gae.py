import pytest
import torch

from sheeprl.utils.utils import gae, gae_function, log_scan_gae

GAMMA = 0.99
GAE_LAMBDA = 0.95


def step_by_step_gae(rewards, values, dones, next_value, num_steps, gamma, gae_lambda):
    """GAE as it was computed with PyTorch, one step at a time."""
    lastgaelam = 0
    nextvalues = next_value
    not_dones = torch.logical_not(dones)
    nextnonterminal = not_dones[-1]
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(num_steps)):
        if t < num_steps - 1:
            nextnonterminal = not_dones[t]
            nextvalues = values[t + 1]
        delta = rewards[t] + nextvalues * nextnonterminal * gamma - values[t]
        advantages[t] = lastgaelam = delta + nextnonterminal * lastgaelam * gamma * gae_lambda
    returns = advantages + values
    return returns, advantages


def rollout(num_steps, num_envs, dtype=torch.float32, next_value_shape=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.randn(num_steps, num_envs, 1, generator=generator, dtype=dtype)
    values = torch.randn(num_steps, num_envs, 1, generator=generator, dtype=dtype)
    dones = (torch.rand(num_steps, num_envs, 1, generator=generator) < 0.1).to(torch.uint8)
    next_value = torch.randn(*(next_value_shape or (num_envs, 1)), generator=generator, dtype=dtype)
    return rewards, values, dones, next_value


@pytest.mark.parametrize("num_steps", [1, 2, 7, 128, 129])
@pytest.mark.parametrize("next_value_shape", [None, (1, 4, 1)])
def test_gae_equals_step_by_step(num_steps, next_value_shape):
    data = rollout(num_steps, 4, next_value_shape=next_value_shape)
    returns, advantages = gae(*data, num_steps, GAMMA, GAE_LAMBDA)
    expected_returns, expected_advantages = step_by_step_gae(*data, num_steps, GAMMA, GAE_LAMBDA)
    assert advantages.dtype == expected_advantages.dtype
    assert torch.equal(advantages, expected_advantages.reshape(advantages.shape))
    assert torch.equal(returns, expected_returns.reshape(returns.shape))


def test_gae_equals_step_by_step_with_rewards_in_float64():
    # A2C and the recurrent PPO compute GAE with the rewards in float64 and the values in float32
    rewards, values, dones, next_value = rollout(128, 4)
    data = (rewards.to(torch.float64), values, dones, next_value)
    returns, advantages = gae(*data, 128, GAMMA, GAE_LAMBDA)
    expected_returns, expected_advantages = step_by_step_gae(*data, 128, GAMMA, GAE_LAMBDA)
    assert advantages.dtype == expected_advantages.dtype == torch.float64
    assert torch.equal(advantages, expected_advantages)
    assert torch.equal(returns, expected_returns)


@pytest.mark.parametrize("num_steps", [1, 2, 3, 7, 8, 128, 129, 1000])
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_log_scan_gae_matches_gae(num_steps, dtype, atol):
    data = rollout(num_steps, 4, dtype=dtype)
    returns, advantages = log_scan_gae(*data, num_steps, GAMMA, GAE_LAMBDA)
    expected_returns, expected_advantages = gae(*data, num_steps, GAMMA, GAE_LAMBDA)
    assert advantages.dtype == expected_advantages.dtype
    torch.testing.assert_close(advantages, expected_advantages, rtol=0, atol=atol)
    torch.testing.assert_close(returns, expected_returns, rtol=0, atol=atol)


def test_log_scan_gae_resets_at_dones():
    # Without discount nor TD errors after the end of an episode, the advantage of a step is the sum of the TD errors
    # until the end of its episode
    rewards = torch.ones(5, 1, 1)
    values = torch.zeros(5, 1, 1)
    dones = torch.tensor([0, 1, 0, 0, 1], dtype=torch.uint8).reshape(5, 1, 1)
    _, advantages = log_scan_gae(rewards, values, dones, torch.zeros(1, 1), 5, 1.0, 1.0)
    assert advantages.flatten().tolist() == [2.0, 1.0, 3.0, 2.0, 1.0]


def test_log_scan_gae_next_value_of_a_sequence():
    # The recurrent PPO bootstraps from the values of a sequence of one step
    data = rollout(16, 4, next_value_shape=(1, 4, 1))
    returns, advantages = log_scan_gae(*data, 16, GAMMA, GAE_LAMBDA)
    expected_returns, expected_advantages = gae(*data, 16, GAMMA, GAE_LAMBDA)
    torch.testing.assert_close(advantages, expected_advantages, rtol=0, atol=1e-5)
    torch.testing.assert_close(returns, expected_returns, rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_log_scan_gae_on_device():
    data = rollout(128, 4)
    returns, advantages = log_scan_gae(*(x.cuda() for x in data), 128, GAMMA, GAE_LAMBDA)
    expected_returns, expected_advantages = gae(*data, 128, GAMMA, GAE_LAMBDA)
    assert advantages.is_cuda and returns.is_cuda
    torch.testing.assert_close(advantages.cpu(), expected_advantages, rtol=0, atol=1e-5)
    torch.testing.assert_close(returns.cpu(), expected_returns, rtol=0, atol=1e-5)


def test_gae_function():
    assert gae_function("loop") is gae
    assert gae_function("log_scan") is log_scan_gae
    with pytest.raises(ValueError, match="algo.gae_method"):
        gae_function("conv")
