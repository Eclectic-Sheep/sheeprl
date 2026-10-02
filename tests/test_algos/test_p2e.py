"""Plan2Explore (P2E-DV1, P2E-DV2, P2E-DV3): the optimizer of the ensembles, the slow critic of a finetuning."""

from typing import List

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl.algos.p2e_dv1.p2e_dv1_exploration import P2EDV1Exploration
from sheeprl.algos.p2e_dv2.p2e_dv2_exploration import P2EDV2Exploration
from sheeprl.algos.p2e_dv3 import p2e_dv3_finetuning
from sheeprl.algos.p2e_dv3.p2e_dv3_exploration import P2EDV3Exploration
from sheeprl.algos.p2e_dv3.p2e_dv3_finetuning import P2EDV3Finetuning
from sheeprl.core import TrainSchedule
from sheeprl.utils.utils import dotdict

# Small models, the observations of the dummy environment
SMALL_ARGS = [
    "env=dummy",
    "algo.cnn_keys.encoder=[rgb]",
    "algo.cnn_keys.decoder=[rgb]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
    "algo.dense_units=8",
    "algo.world_model.encoder.cnn_channels_multiplier=2",
    "algo.world_model.recurrent_model.recurrent_state_size=8",
    "algo.world_model.representation_model.hidden_size=8",
    "algo.world_model.transition_model.hidden_size=8",
    "metric.log_level=0",
]
OBS_SPACE = gym.spaces.Dict(
    {
        "rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
        "state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32),
    }
)


def config(overrides: List[str]) -> dotdict:
    """The configuration the CLI gives to the algorithms (without checking the mandatory values)."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        return dotdict(OmegaConf.to_container(compose(config_name="config", overrides=overrides), resolve=True))


def build(algo_cls, overrides: List[str], log_dir: str):
    """The algorithm `algo_cls` and its training state, built from the configuration with `overrides`."""
    cfg = config(SMALL_ARGS + overrides)
    algo = algo_cls(Fabric(accelerator="cpu", devices=1), cfg)
    schedule = TrainSchedule(cfg, 1, algo.steps_per_iteration, off_policy=algo.off_policy)
    state, _ = algo.build(OBS_SPACE, gym.spaces.Discrete(3), schedule, log_dir)
    return algo, state


@pytest.mark.parametrize(
    "algo_cls,exp",
    [
        (P2EDV1Exploration, "p2e_dv1_exploration"),
        (P2EDV2Exploration, "p2e_dv2_exploration"),
        (P2EDV3Exploration, "p2e_dv3_exploration"),
    ],
)
def test_the_ensembles_are_trained_with_their_own_optimizer(algo_cls, exp, tmp_path):
    # Their optimizer was the one of the critic (P2E-DV2, P2E-DV3) or of the world model (P2E-DV1)
    _, state = build(
        algo_cls,
        [f"exp={exp}", "algo.ensembles.optimizer.lr=0.123", "algo.ensembles.optimizer.weight_decay=0.01"],
        str(tmp_path),
    )
    (group,) = state.ensemble_optimizer.param_groups
    assert group["lr"] == 0.123
    assert group["weight_decay"] == 0.01


def test_a_finetuning_keeps_the_slow_critic_of_the_exploration(tmp_path, monkeypatch):
    # Its first gradient step copied the critic into the target critic: the slow critic the exploration had trained
    # was lost
    algo, state = build(P2EDV3Finetuning, ["exp=p2e_dv3_finetuning"], str(tmp_path))
    with torch.no_grad():
        # The slow critic of the exploration lags behind its critic
        for p in state.target_critic_task.parameters():
            p.add_(1.0)
    critic = [p.detach().clone() for p in state.critic_task.parameters()]
    target = [p.detach().clone() for p in state.target_critic_task.parameters()]
    # Only the update of the target critic is tested
    monkeypatch.setattr(p2e_dv3_finetuning, "world_model_learning", lambda *args: (None, None, {}))
    monkeypatch.setattr(
        p2e_dv3_finetuning,
        "behaviour_learning",
        lambda *args: {"policy_loss": 0, "value_loss": 0, "actor_grads": None, "critic_grads": None},
    )
    algo.train_step(state, {"terminated": None}, step=0)
    tau = algo.cfg.algo.critic.tau
    assert 0 < tau < 1
    for c, t, updated in zip(critic, target, state.target_critic_task.parameters()):
        torch.testing.assert_close(updated.detach(), tau * c + (1 - tau) * t)
