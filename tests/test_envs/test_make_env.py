from typing import List
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl.algos.p2e_dv1.p2e_dv1_finetuning import P2EDV1Finetuning
from sheeprl.algos.p2e_dv2.p2e_dv2_finetuning import P2EDV2Finetuning
from sheeprl.algos.p2e_dv3.p2e_dv3_finetuning import P2EDV3Finetuning
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict


def config(overrides: List[str]) -> dotdict:
    """The configuration the CLI gives to the algorithms (without checking the mandatory values)."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        return dotdict(OmegaConf.to_container(compose(config_name="config", overrides=overrides), resolve=True))


def played_actions(cfg: dotdict, actions: List[np.ndarray]) -> List[np.ndarray]:
    """The actions Pendulum receives when the environment of `cfg` plays `actions`."""
    played = []
    pendulum_step = gym.envs.classic_control.PendulumEnv.step

    def recording_step(self, u):
        played.append(np.array(u))
        return pendulum_step(self, u)

    env = make_env(cfg, seed=0, rank=0)()
    env.reset(seed=0)
    with mock.patch.object(gym.envs.classic_control.PendulumEnv, "step", recording_step):
        for action in actions:
            env.step(action)
    env.close()
    return played


@pytest.mark.parametrize(
    "exp",
    [
        "dreamer_v3",
        "p2e_dv3_exploration",
        "p2e_dv3_finetuning",
        "dreamer_v2",
        "p2e_dv2_exploration",
        "p2e_dv2_finetuning",
        "dreamer_v1",
        "p2e_dv1_exploration",
        "p2e_dv1_finetuning",
    ],
)
def test_dreamers_play_normalized_actions(exp):
    # The actors of the Dreamers act in [-1, 1]: Pendulum (±2) gets twice their actions, and the random ones are
    # sampled in [-1, 1] too (they were sampled in ±2, and the policy could never reach the bounds)
    # Vector observations only: rendering Pendulum needs the PNG support of pygame, which some installations miss
    cfg = config(
        [f"exp={exp}", "env=gym", "env.id=Pendulum-v1", "algo.mlp_keys.encoder=[state]", "algo.cnn_keys.encoder=[]"]
    )
    env = make_env(cfg, seed=0, rank=0)()
    assert env.action_space == gym.spaces.Box(-1, 1, (1,), np.float32)
    env.close()
    played = played_actions(cfg, [np.array([1.0], np.float32), np.array([-0.25], np.float32)])
    np.testing.assert_allclose(played, [[2.0], [-0.5]])


def test_other_algorithms_and_old_configs_play_the_actions_as_they_are():
    sac = config(["exp=sac", "env.id=Pendulum-v1"])
    assert "normalize_actions" not in sac.algo
    # A DreamerV3 configuration saved before the option: its checkpoints play as they were trained
    dreamer = config(
        ["exp=dreamer_v3", "env=gym", "env.id=Pendulum-v1", "algo.mlp_keys.encoder=[state]", "algo.cnn_keys.encoder=[]"]
    )
    del dreamer.algo["normalize_actions"]
    for cfg in (sac, dreamer):
        env = make_env(cfg, seed=0, rank=0)()
        assert env.action_space == gym.spaces.Box(-2, 2, (1,), np.float32)
        env.close()
        np.testing.assert_allclose(played_actions(cfg, [np.array([1.5], np.float32)]), [[1.5]])


@pytest.mark.parametrize("algo", ["p2e_dv3", "p2e_dv2", "p2e_dv1"])
@pytest.mark.parametrize("explored_with", [True, False, None])
def test_p2e_finetuning_normalizes_the_actions_as_the_exploration(explored_with, algo):
    overrides = ["env=gym", "env.id=Pendulum-v1", "algo.mlp_keys.encoder=[state]"]
    exploration_cfg = config([f"exp={algo}_exploration", *overrides])
    if explored_with is None:
        # An exploration saved before the option didn't normalize the actions
        del exploration_cfg.algo["normalize_actions"]
    else:
        exploration_cfg.algo.normalize_actions = explored_with
    cfg = config([f"exp={algo}_finetuning", *overrides])
    finetuning_cls = {"p2e_dv3": P2EDV3Finetuning, "p2e_dv2": P2EDV2Finetuning, "p2e_dv1": P2EDV1Finetuning}[algo]
    finetuning_cls(Fabric(accelerator="cpu", devices=1), cfg, exploration_cfg)
    assert cfg.algo.normalize_actions is bool(explored_with)
