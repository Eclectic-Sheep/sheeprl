"""DreamerV1 (and P2E-DV1): the continue loss, the exploration noise, the player."""

import os
import shutil
import sys
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Bernoulli, Independent, Normal

from sheeprl import ROOT_DIR
from sheeprl.algos.dreamer_v1 import agent
from sheeprl.algos.dreamer_v1.agent import PlayerDV1, RecurrentModel
from sheeprl.algos.dreamer_v1.loss import reconstruction_loss
from sheeprl.algos.dreamer_v2.agent import Actor

DREAMER_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "dry_run=True",
    "env=dummy",
    "env.num_envs=2",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.devices=1",
    "fabric.accelerator=cpu",
    "metric.log_level=0",
    "checkpoint.save_last=False",
    "algo.run_test=False",
    "algo.cnn_keys.encoder=[]",
    "algo.cnn_keys.decoder=[]",
    "algo.mlp_keys.encoder=[state]",
    "algo.mlp_keys.decoder=[state]",
    "algo.dense_units=8",
    "algo.world_model.recurrent_model.recurrent_state_size=8",
    "algo.world_model.representation_model.hidden_size=8",
    "algo.world_model.transition_model.hidden_size=8",
    "algo.horizon=4",
    "algo.per_rank_batch_size=1",
    "algo.per_rank_sequence_length=2",
    "algo.learning_starts=0",
    "algo.replay_ratio=1",
    "buffer.size=10",
]


def run_dreamer_v1(args, root_dir):
    from sheeprl.cli import run

    argv = [os.path.join(ROOT_DIR, "__main__.py"), *DREAMER_ARGS, *args, f"root_dir={root_dir}"]
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(sys, "argv", argv):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)


def test_the_continue_loss_is_the_mean_negative_log_likelihood():
    # It was the log-likelihood itself, one per step: the loss wasn't a scalar (the training crashed) and its sign
    # trained the continue model away from the targets
    torch.manual_seed(0)
    T, B = 4, 3
    # The targets are the discount, not 0 or 1: the training doesn't validate the arguments of the distributions
    qc = Independent(Bernoulli(logits=torch.randn(T, B, 1), validate_args=False), 1)
    targets = (torch.rand(T, B, 1) > 0.3).float() * 0.99
    qo = {"state": Independent(Normal(torch.zeros(T, B, 2), 1), 1)}
    qr = Independent(Normal(torch.zeros(T, B, 1), 1), 1)
    states = Independent(Normal(torch.zeros(T, B, 5), 1), 1)
    rec_loss, *_, continue_loss = reconstruction_loss(
        qo, {"state": torch.randn(T, B, 2)}, qr, torch.zeros(T, B, 1), states, states, 3.0, 1.0, qc, targets, 2.0
    )
    assert rec_loss.shape == continue_loss.shape == ()
    torch.testing.assert_close(continue_loss, -2.0 * qc.log_prob(targets).mean())
    assert continue_loss > 0


@pytest.mark.parametrize("exp", ["dreamer_v1", "p2e_dv1_exploration"])
def test_the_continue_model_is_trained(exp):
    run_dreamer_v1(
        [f"exp={exp}", "env.id=discrete_dummy", "algo.world_model.use_continues=True"], f"pytest_{exp}_continues"
    )


def test_the_exploration_noise_halves_every_decay_steps():
    # The amount was `0.5 ** step / decay`: `expl_min` from the first steps
    actor = SimpleNamespace(_expl_amount=0.4, _expl_decay=1000, _expl_min=0.05)
    for step, amount in ((0, 0.4), (1000, 0.2), (2000, 0.1), (4000, 0.05)):
        assert Actor._get_expl_amount(actor, step) == pytest.approx(amount)


def test_each_environment_explores_on_its_own():
    # One random draw decided for all the environments whether they played a random action
    torch.manual_seed(0)
    num_envs = 1000
    actions = torch.nn.functional.one_hot(torch.zeros(1, num_envs, dtype=torch.long), 4).float()
    actor = SimpleNamespace(is_continuous=False, _get_expl_amount=lambda step: 0.5)
    (explored,) = Actor.add_exploration_noise(actor, [actions])
    assert explored.shape == actions.shape
    changed = (explored != actions).any(-1).sum().item()
    # About half of the envs explore, and 3/4 of the random actions differ from the played one
    assert 300 < changed < 450, changed


def test_the_player_samples_the_posterior_with_the_minimum_std_of_the_world_model(monkeypatch):
    # The player used the default minimum std (0.1) whatever `algo.world_model.min_std`
    min_stds = []

    def compute_stochastic_state(state_information, event_shape=1, min_std=0.1):
        min_stds.append(min_std)
        return (None, None), torch.zeros(*state_information.shape[:-1], 4)

    monkeypatch.setattr(agent, "compute_stochastic_state", compute_stochastic_state)
    player = PlayerDV1(
        encoder=lambda obs: torch.zeros(1, 2, 5),
        recurrent_model=RecurrentModel(4 + 3, 8),
        representation_model=nn.Linear(8 + 5, 8),
        actor=lambda latent_state, greedy, mask: ([torch.zeros(1, 2, 3)], None),
        actions_dim=[3],
        num_envs=2,
        stochastic_size=4,
        recurrent_state_size=8,
        device="cpu",
        min_std=0.5,
    )
    player.init_states()
    player.get_actions({})
    assert min_stds == [0.5]


@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"))],
)
def test_the_player_follows_the_updates_of_the_agent(accelerator):
    # The player was a copy of the agent with the weights tied: on CUDA its GRU moved them into a new buffer at its
    # first forward (`flatten_parameters`), and the player played with the initial recurrent model for the whole
    # training. It also takes the minimum std of the world model
    from hydra import compose, initialize_config_module
    from lightning import Fabric
    from omegaconf import OmegaConf

    from sheeprl.algos.dreamer_v1.agent import build_agent
    from sheeprl.utils.utils import dotdict

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v1",
                "env.num_envs=2",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.mlp_keys.decoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.min_std=0.3",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    fabric = Fabric(accelerator=accelerator, devices=1)
    world_model, actor, _, player = build_agent(fabric, [3], False, cfg, obs_space)
    assert player.min_std == 0.3
    player.init_states()
    with torch.no_grad():
        player.get_actions({"state": torch.randn(1, 2, 5, device=fabric.device)})
        for p in [*world_model.parameters(), *actor.parameters()]:
            p.add_(1.0)
    for agent_module, player_module in (
        (world_model.encoder, player.encoder),
        (world_model.rssm.recurrent_model, player.recurrent_model),
        (world_model.rssm.representation_model, player.representation_model),
        (actor, player.actor),
    ):
        agent_params, player_params = list(agent_module.parameters()), list(player_module.parameters())
        assert len(agent_params) == len(player_params) > 0
        for agent_p, player_p in zip(agent_params, player_params):
            assert torch.equal(agent_p, player_p)
