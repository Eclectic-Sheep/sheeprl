import pytest

from sheeprl.envs.dummy import ContinuousDummyEnv, DiscreteDummyEnv, MultiDiscreteDummyEnv
from sheeprl.envs.wrappers import ActionRepeat

ENVIRONMENTS = {
    "discrete_dummy": DiscreteDummyEnv,
    "multidiscrete_dummy": MultiDiscreteDummyEnv,
    "continuous_dummy": ContinuousDummyEnv,
}


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
        o, r, done, t, info = env.step(env.action_space.sample())
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

    obs = env.reset()
    assert obs is not None
