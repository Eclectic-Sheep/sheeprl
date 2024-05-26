import gymnasium as gym
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
