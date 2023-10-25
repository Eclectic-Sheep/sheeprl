import numpy as np
import pytest

from sheeprl.utils.utils import nstep_returns


@pytest.fixture()
def n_steps():
    return 5


@pytest.fixture()
def gamma():
    return 0.99


@pytest.fixture()
def rewards_long():
    return np.ones((10, 1, 1))


@pytest.fixture()
def rewards_short():
    return np.ones((4, 1, 1))


@pytest.fixture()
def values_long():
    return np.ones((10, 1, 1)) * 2


@pytest.fixture()
def values_short():
    return np.ones((4, 1, 1)) * 2


@pytest.fixture()
def dones_long():
    dones = np.zeros((10, 1, 1))
    dones[9] = 1
    return dones.astype(np.bool)


@pytest.fixture()
def dones_short():
    dones = np.zeros((4, 1, 1))
    dones[3] = 1
    return dones.astype(np.bool)


def test_standard_case(rewards_long, values_long, dones_long, n_steps, gamma):
    result = nstep_returns(rewards_long, values_long, dones_long, n_steps, gamma)
    expected_result_0 = 1.0 + 0.99 + 0.99**2 + 0.99**3 + 0.99**4 + 2 * 0.99**5
    expected_result_8 = 1.0 + 0.99
    expected_result_9 = 1.0
    assert result.shape == (10, 1, 1)
    assert result[0] == expected_result_0
    assert result[8] == expected_result_8
    assert result[9] == expected_result_9


def test_short_case(rewards_short, values_short, dones_short, n_steps, gamma):
    result = nstep_returns(rewards_short, values_short, dones_short, n_steps, gamma)
    expected_result_0 = 1.0 + 0.99 + 0.99**2 + 0.99**3
    expected_result_3 = 1.0
    assert result.shape == (4, 1, 1)
    assert result[0] == expected_result_0
    assert result[3] == expected_result_3
