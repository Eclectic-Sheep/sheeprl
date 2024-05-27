import gymnasium as gym
import numpy as np
import pytest

from sheeprl.envs.dummy import ContinuousDummyEnv, DiscreteDummyEnv, MultiDiscreteDummyEnv
from sheeprl.envs.wrappers import ActionsAsObservationWrapper, MaskVelocityWrapper

ENVIRONMENTS = {
    "discrete_dummy": DiscreteDummyEnv,
    "multidiscrete_dummy": MultiDiscreteDummyEnv,
    "continuous_dummy": ContinuousDummyEnv,
}


def test_mask_velocities_fail():
    with pytest.raises(NotImplementedError):
        env = gym.make("CarRacing-v2")
        env = MaskVelocityWrapper(env)


@pytest.mark.parametrize("num_stack", [1, 4, 8])
@pytest.mark.parametrize("dilation", [1, 2, 4])
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_actions_as_observations_wrapper(env_id: str, num_stack, dilation):
    env = ENVIRONMENTS[env_id]()
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        noop = [0, 0]
    else:
        noop = 0
    env = ActionsAsObservationWrapper(env, num_stack=num_stack, noop=noop, dilation=dilation)

    o = env.reset()[0]
    assert len(o["action_stack"].shape) == len(env.observation_space["action_stack"].shape)
    for d1, d2 in zip(o["action_stack"].shape, env.observation_space["action_stack"].shape):
        assert d1 == d2

    for _ in range(64):
        o = env.step(env.action_space.sample())[0]
        assert len(o["action_stack"].shape) == len(env.observation_space["action_stack"].shape)
        for d1, d2 in zip(o["action_stack"].shape, env.observation_space["action_stack"].shape):
            assert d1 == d2


@pytest.mark.parametrize("num_stack", [-1, 0])
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_actions_as_observations_wrapper_invalid_num_stack(env_id, num_stack):
    env = ENVIRONMENTS[env_id]()
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        noop = [0, 0]
    else:
        noop = 0
    with pytest.raises(ValueError, match="The number of actions to the"):
        env = ActionsAsObservationWrapper(env, num_stack=num_stack, noop=noop, dilation=3)


@pytest.mark.parametrize("dilation", [-1, 0])
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_actions_as_observations_wrapper_invalid_dilation(env_id, dilation):
    env = ENVIRONMENTS[env_id]()
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        noop = [0, 0]
    else:
        noop = 0
    with pytest.raises(ValueError, match="The actions stack dilation argument must be greater than zero"):
        env = ActionsAsObservationWrapper(env, num_stack=3, noop=noop, dilation=dilation)


@pytest.mark.parametrize("noop", [set([0, 0, 0]), "this is an invalid type", np.array([0, 0, 0])])
@pytest.mark.parametrize("env_id", ["discrete_dummy", "multidiscrete_dummy", "continuous_dummy"])
def test_actions_as_observations_wrapper_invalid_noop_type(env_id, noop):
    env = ENVIRONMENTS[env_id]()
    with pytest.raises(ValueError, match="The noop action must be an integer or float or list"):
        env = ActionsAsObservationWrapper(env, num_stack=3, noop=noop, dilation=2)


def test_actions_as_observations_wrapper_invalid_noop_continuous_type():
    env = ContinuousDummyEnv()
    with pytest.raises(ValueError, match="The noop actions must be a float for continuous action spaces"):
        env = ActionsAsObservationWrapper(env, num_stack=3, noop=[0, 0, 0], dilation=2)


@pytest.mark.parametrize("noop", [[0, 0, 0], 0.0])
def test_actions_as_observations_wrapper_invalid_noop_discrete_type(noop):
    env = DiscreteDummyEnv()
    with pytest.raises(ValueError, match="The noop actions must be an integer for discrete action spaces"):
        env = ActionsAsObservationWrapper(env, num_stack=3, noop=noop, dilation=2)


@pytest.mark.parametrize("noop", [0, 0.0])
def test_actions_as_observations_wrapper_invalid_noop_multidiscrete_type(noop):
    env = MultiDiscreteDummyEnv()
    with pytest.raises(ValueError, match="The noop actions must be a list for multi-discrete action spaces"):
        env = ActionsAsObservationWrapper(env, num_stack=3, noop=noop, dilation=2)


@pytest.mark.parametrize("noop", [[0], [0, 0, 0]])
def test_actions_as_observations_wrapper_invalid_noop_multidiscrete_n_actions(noop):
    env = MultiDiscreteDummyEnv()
    with pytest.raises(
        RuntimeError, match="The number of noop actions must be equal to the number of actions of the environment"
    ):
        env = ActionsAsObservationWrapper(env, num_stack=3, noop=noop, dilation=2)
