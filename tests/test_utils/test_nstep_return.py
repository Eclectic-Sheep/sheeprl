import pytest
import torch

from sheeprl.utils.utils import nstep_returns


@pytest.fixture()
def n_steps():
    return 5


@pytest.fixture()
def gamma():
    return 0.99


@pytest.fixture()
def rewards_long():
    return torch.ones(10, 1, 1)


@pytest.fixture()
def rewards_short():
    return torch.ones(4, 1, 1)


@pytest.fixture()
def values_long():
    return torch.ones(10, 1, 1) * 2


@pytest.fixture()
def values_short():
    return torch.ones(4, 1, 1) * 2


@pytest.fixture()
def dones_long():
    dones = torch.zeros(10, 1, 1)
    dones[9] = 1
    return dones.to(dtype=torch.bool)


@pytest.fixture()
def dones_short():
    dones = torch.zeros(4, 1, 1)
    dones[3] = 1
    return dones.to(dtype=torch.bool)


def test_standard_case(rewards_long, values_long, dones_long, n_steps, gamma):
    result = nstep_returns(rewards_long, values_long, dones_long, n_steps, gamma)
    expected_result_0 = 1.0 + 0.99 + 0.99**2 + 0.99**3 + 0.99**4 + 2 * 0.99**5
    expected_result_8 = 1.0 + 0.99
    expected_result_9 = 1.0
    assert result.shape == torch.Size([10, 1, 1])
    assert torch.allclose(result[0], torch.tensor([expected_result_0]))
    assert torch.allclose(result[8], torch.tensor([expected_result_8]))
    assert torch.allclose(result[9], torch.tensor([expected_result_9]))


def test_short_case(rewards_short, values_short, dones_short, n_steps, gamma):
    result = nstep_returns(rewards_short, values_short, dones_short, n_steps, gamma)
    expected_result_0 = 1.0 + 0.99 + 0.99**2 + 0.99**3
    expected_result_3 = 1.0
    assert result.shape == torch.Size([4, 1, 1])
    assert torch.allclose(result[0], torch.tensor([expected_result_0]))
    assert torch.allclose(result[3], torch.tensor([expected_result_3]))
