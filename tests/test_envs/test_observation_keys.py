"""The observation keys of the environments (`make_env`): the keys the inputs and `algo.extra_keys` declare, each
through the `frame` stage of its transforms, the others dropped; a bare observation is the key `state`, the render the
key `env.render_key`."""

from typing import List

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict


def config(overrides: List[str]) -> dotdict:
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["env.capture_video=False", *overrides])
    return dotdict(OmegaConf.to_container(cfg, resolve=True))


def first_obs(cfg: dotdict):
    env = make_env(cfg, seed=0, rank=0)()
    obs, _ = env.reset(seed=0)
    space = env.observation_space
    env.close()
    return space, obs


def test_a_bare_observation_is_the_key_state():
    space, obs = first_obs(config(["exp=ppo", "env.id=CartPole-v1", "algo.inputs.vector.keys=[state]"]))
    assert list(space.spaces) == ["state"] and obs["state"].shape == (4,)


def test_the_render_is_the_key_of_env_render_key():
    cfg = config(
        [
            "exp=ppo",
            "env.id=CartPole-v1",
            "env.render_key=rgb",
            "env.screen_size=32",
            "algo.inputs.image.keys=[rgb]",
            "algo.inputs.vector.keys=[]",
        ]
    )
    space, obs = first_obs(cfg)
    # The frame of CartPole resized by the `Image` transform of the input; `state` is declared by no input: dropped
    assert list(space.spaces) == ["rgb"]
    assert space["rgb"].shape == obs["rgb"].shape == (3, 32, 32) and obs["rgb"].dtype == np.uint8
    with pytest.raises(ValueError, match="not observations of the environment"):
        first_obs(config(["exp=ppo", "env.id=CartPole-v1", "algo.inputs.image.keys=[rgb]"]))


def test_the_keys_nobody_declares_are_dropped_and_the_extra_keys_kept_as_they_are():
    dummy = ["exp=ppo", "env=dummy", "env.id=discrete_dummy", "env.screen_size=16"]
    space, _ = first_obs(config([*dummy, "algo.inputs.vector.keys=[state]"]))
    assert list(space.spaces) == ["state"]
    space, obs = first_obs(config([*dummy, "algo.inputs.vector.keys=[state]", "algo.extra_keys=[rgb]"]))
    # An extra key has no transforms: the image of the environment, not resized
    assert set(space.spaces) == {"state", "rgb"} and obs["rgb"].shape == (3, 64, 64)


def test_a_key_the_environment_does_not_have_is_an_error_listing_its_keys():
    with pytest.raises(ValueError, match=r"\['rgb', 'state'\]"):
        first_obs(config(["exp=ppo", "env=dummy", "env.id=discrete_dummy", "algo.inputs.vector.keys=[position]"]))


def test_a_key_of_two_inputs_is_stored_once_with_the_same_stored_stages():
    image = "{_target_:sheeprl.transforms.Image,size:16,grayscale:false}"
    two_inputs = [
        "exp=ppo",
        "env=dummy",
        "env.id=discrete_dummy",
        "algo.inputs.image.keys=[rgb]",
        "algo.inputs.vector.keys=[]",
        f"algo.inputs.image.transforms=[{image},{{_target_:sheeprl.transforms.Scale}}]",
        "+algo.inputs.raw.keys=[rgb]",
    ]
    # Only their `batch` transforms differ: stored once, resized
    space, _ = first_obs(config([*two_inputs, f"+algo.inputs.raw.transforms=[{image}]"]))
    assert list(space.spaces) == ["rgb"] and space["rgb"].shape == (3, 16, 16)
    with pytest.raises(ValueError, match="different `frame` or `envs` transforms"):
        first_obs(config([*two_inputs, "+algo.inputs.raw.transforms=[]"]))
