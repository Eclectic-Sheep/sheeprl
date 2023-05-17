import pytest
from sheeprl.envs.wrappers import MaskVelocityWrapper
import gymnasium as gym


def test_mask_velocities_fail():
    with pytest.raises(NotImplementedError):
        env = gym.make("CarRacing-v2")
        env = MaskVelocityWrapper(env)
