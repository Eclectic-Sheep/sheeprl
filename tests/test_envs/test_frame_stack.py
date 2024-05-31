import numpy as np
import pytest

from sheeprl.envs.dummy import ContinuousDummyEnv, DiscreteDummyEnv, MultiDiscreteDummyEnv
from sheeprl.envs.wrappers import FrameStack

ENVIRONMENTS = {
    "discrete_dummy": DiscreteDummyEnv,
    "multidiscrete_dummy": MultiDiscreteDummyEnv,
    "continuous_dummy": ContinuousDummyEnv,
}


@pytest.mark.parametrize("dilation", [1, 2, 4])
@pytest.mark.parametrize("num_stack", [1, 2, 3])
@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
def test_valid_initialization(env_id, num_stack, dilation):
    env = ENVIRONMENTS[env_id]()

    env = FrameStack(env, num_stack=num_stack, cnn_keys=["rgb"], dilation=dilation)
    assert env._num_stack == num_stack
    assert env._dilation == dilation
    assert "rgb" in env._cnn_keys
    assert "rgb" in env._frames


@pytest.mark.parametrize("num_stack", [-2.4, -1, 0])
@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
def test_invalid_num_stack(env_id, num_stack):
    env = ENVIRONMENTS[env_id]()

    with pytest.raises(ValueError, match="Invalid value for num_stack, expected a value greater"):
        FrameStack(env, num_stack=num_stack, cnn_keys=["rgb"], dilation=2)


@pytest.mark.parametrize("num_stack", [1, 3, 7])
@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
def test_invalid_observation_space(env_id, num_stack):
    env = ENVIRONMENTS[env_id](dict_obs_space=False)

    with pytest.raises(RuntimeError, match="Expected an observation space of type gym.spaces.Dict"):
        FrameStack(env, num_stack=num_stack, cnn_keys=["rgb"], dilation=2)


@pytest.mark.parametrize("cnn_keys", [[], None])
@pytest.mark.parametrize("num_stack", [1, 3, 7])
@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
def test_invalid_cnn_keys(env_id, num_stack, cnn_keys):
    env = ENVIRONMENTS[env_id]()

    with pytest.raises(RuntimeError, match="Specify at least one valid cnn key"):
        FrameStack(env, num_stack=num_stack, cnn_keys=cnn_keys, dilation=2)


@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
@pytest.mark.parametrize("num_stack", [1, 3, 7])
def test_reset_method(env_id, num_stack):
    env = ENVIRONMENTS[env_id]()

    wrapper = FrameStack(env, num_stack=num_stack, cnn_keys=["rgb"])
    obs, _ = wrapper.reset()

    assert "rgb" in obs
    assert obs["rgb"].shape == (num_stack, *env.observation_space["rgb"].shape)


@pytest.mark.parametrize("num_stack", [1, 2, 5])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_framestack(num_stack, dilation):
    env = DiscreteDummyEnv()
    env = FrameStack(env, num_stack, cnn_keys=["rgb"], dilation=dilation)

    # Reset the environment to initialize the frame stack
    obs, _ = env.reset()

    for step in range(1, 64):
        obs = env.step(None)[0]

        expected_frame = np.stack(
            [
                np.full(
                    env.env.observation_space["rgb"].shape,
                    max(0, (step - dilation * (num_stack - i - 1))) % 256,
                    dtype=np.uint8,
                )
                for i in range(num_stack)
            ],
            axis=0,
        )
        np.testing.assert_array_equal(obs["rgb"], expected_frame)


@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
@pytest.mark.parametrize("num_stack", [1, 3, 7])
def test_step_method(env_id, num_stack):
    env = ENVIRONMENTS[env_id]()
    wrapper = FrameStack(env, num_stack=num_stack, cnn_keys=["rgb"])
    wrapper.reset()
    action = wrapper.action_space.sample()
    obs = wrapper.step(action)[0]
    assert "rgb" in obs
    assert obs["rgb"].shape == (num_stack, *env.observation_space["rgb"].shape)
