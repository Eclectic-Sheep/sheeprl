import gymnasium as gym
import numpy as np
import pytest

from sheeprl.envs.dummy import ContinuousDummyEnv, DiscreteDummyEnv, MultiDiscreteDummyEnv
from sheeprl.envs.wrappers import ActionRepeat, MaskVelocityWrapper, RewardAsObservationWrapper

ENVIRONMENTS = {
    "discrete_dummy": DiscreteDummyEnv,
    "multidiscrete_dummy": MultiDiscreteDummyEnv,
    "continuous_dummy": ContinuousDummyEnv,
}


def test_mask_velocities_fail():
    with pytest.raises(NotImplementedError):
        env = gym.make("CarRacing-v2")
        env = MaskVelocityWrapper(env)


@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
@pytest.mark.parametrize("dict_obs_space", [True, False])
def test_rewards_as_observation_wrapper_initialization(env_id, dict_obs_space):
    env = ENVIRONMENTS[env_id](dict_obs_space=dict_obs_space)
    wrapped_env = RewardAsObservationWrapper(env)

    if dict_obs_space:
        assert "reward" in wrapped_env.observation_space.spaces
        assert isinstance(wrapped_env.observation_space.spaces["reward"], gym.spaces.Box)
    else:
        assert isinstance(wrapped_env.observation_space, gym.spaces.Dict)
        assert "obs" in wrapped_env.observation_space.spaces
        assert "reward" in wrapped_env.observation_space.spaces
        assert isinstance(wrapped_env.observation_space.spaces["reward"], gym.spaces.Box)


@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
@pytest.mark.parametrize("dict_obs_space", [True, False])
def test_rewards_as_observation_wrapper_step_method(env_id, dict_obs_space):
    env = ENVIRONMENTS[env_id](dict_obs_space=dict_obs_space)
    wrapped_env = RewardAsObservationWrapper(env)

    obs = wrapped_env.step(env.action_space.sample())[0]
    if dict_obs_space:
        assert "rgb" in obs
        assert "state" in obs
        assert "reward" in obs
    else:
        assert "obs" in obs
        assert "reward" in obs
    np.testing.assert_array_equal(obs["reward"], np.array([0.0]))


@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
@pytest.mark.parametrize("dict_obs_space", [True, False])
def test_rewards_as_observation_wrapper_reset_method(env_id, dict_obs_space):
    env = ENVIRONMENTS[env_id](dict_obs_space=dict_obs_space)
    wrapped_env = RewardAsObservationWrapper(env)

    obs = wrapped_env.reset()[0]
    if dict_obs_space:
        assert "rgb" in obs
        assert "state" in obs
        assert "reward" in obs
    else:
        assert "obs" in obs
        assert "reward" in obs
    np.testing.assert_array_equal(obs["reward"], np.array([0.0]))


@pytest.mark.parametrize("amount", [-1.3, -1, 0])
@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
def test_action_repeat_non_positive_amount(env_id, amount):
    env = ENVIRONMENTS[env_id]()
    with pytest.raises(ValueError, match="`amount` should be a positive integer"):
        env = ActionRepeat(env, amount)


@pytest.mark.parametrize("amount", [1, 2, 3, 7, 10])
@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
def test_action_repeat(env_id: str, amount):
    env = ENVIRONMENTS[env_id]()
    env = ActionRepeat(env, amount)

    env.reset()

    assert env.action_repeat == amount
    for i in range(amount * 10):
        _, _, done, _, _ = env.step(env.action_space.sample())
        step = env.__getattr__("_current_step")
        if not done:
            assert amount * (i + 1) == step
        else:
            assert amount * i < step <= amount * (i + 1)
            break


@pytest.mark.parametrize("env_id", ENVIRONMENTS.keys())
def test_reset_method(env_id):
    env = ENVIRONMENTS[env_id]()
    env = ActionRepeat(env, amount=3)

    obs = env.reset()[0]
    assert obs is not None
