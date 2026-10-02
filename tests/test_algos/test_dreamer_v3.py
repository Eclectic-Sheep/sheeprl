"""DreamerV3 (and P2E-DV3): the initialization of the weights, the normalization layers of the decoder."""

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf
from torch import nn

from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.utils import init_weights
from sheeprl.utils.utils import dotdict


@pytest.mark.parametrize(
    "layer,fan_in,fan_out",
    [
        (nn.Linear(256, 512), 256, 512),
        (nn.Conv2d(32, 64, kernel_size=4), 32 * 16, 64 * 16),
        (nn.ConvTranspose2d(64, 32, kernel_size=4), 64 * 16, 32 * 16),
    ],
)
def test_the_weights_have_the_variance_of_the_average_fan(layer, fan_in, fan_out):
    # The weights of the convolutions were truncated at -2 and 2 instead of 2 standard deviations: the division by the
    # standard deviation of the truncated normal (0.8796) made them 14% larger than the ones of DreamerV3
    torch.manual_seed(0)
    layer.apply(init_weights)
    std = math.sqrt(2 / (fan_in + fan_out))
    weight = layer.weight.detach()
    assert weight.std().item() == pytest.approx(std, rel=0.03)
    assert weight.abs().max().item() <= 2 * std / 0.87962566103423978 + 1e-6


def test_the_cnn_decoder_is_normalized_as_configured():
    # Its normalization layers took the arguments of the ones of the MLP decoder (`observation_model.mlp_layer_norm`)
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v3",
                "env=dummy",
                "algo.cnn_keys.encoder=[rgb]",
                "algo.mlp_keys.encoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.encoder.cnn_channels_multiplier=2",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
                "algo.cnn_layer_norm.kw.eps=0.123",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict(
        {
            "rgb": gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
            "state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32),
        }
    )
    world_model, *_ = build_agent(Fabric(accelerator="cpu", devices=1), [3], False, cfg, obs_space)
    cnn_decoder, mlp_decoder = world_model.observation_model.cnn_decoder, world_model.observation_model.mlp_decoder
    assert {m.eps for m in cnn_decoder.modules() if isinstance(m, nn.LayerNorm)} == {0.123}
    assert {m.eps for m in mlp_decoder.modules() if isinstance(m, nn.LayerNorm)} == {1e-3}


def test_the_player_resets_the_environments_one_by_one(recwarn):
    # After a full reset, the recurrent state was the initial one expanded to the environments, whose rows share the
    # memory: resetting one environment (e.g. at the end of an episode during the random actions, when the player
    # computes no new state) wrote in the expanded tensor (a deprecated `index_put_`, with a warning)
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v3",
                "env=dummy",
                "env.num_envs=3",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    *_, player = build_agent(Fabric(accelerator="cpu", devices=1), [3], False, cfg, obs_space)
    player.init_states()
    initial = player.recurrent_state.clone()
    player.recurrent_state[:, 1] += 1.0
    torch.testing.assert_close(player.recurrent_state[:, [0, 2]], initial[:, [0, 2]])
    player.init_states([1])
    torch.testing.assert_close(player.recurrent_state, initial)
    assert not [w for w in recwarn if "expanded tensors" in str(w.message)]


def initial_states_after_a_gradient_step(fabric: Fabric) -> bool:
    """One gradient step of DreamerV3 in every process, each on its own batch: whether the processes then have the same
    learnable initial recurrent state."""
    from sheeprl.algos.dreamer_v3.dreamer_v3 import train
    from sheeprl.algos.dreamer_v3.utils import Moments

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=dreamer_v3",
                "env=dummy",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.mlp_keys.decoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
                "algo.horizon=3",
                "algo.per_rank_batch_size=2",
                "algo.per_rank_sequence_length=4",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    fabric.seed_everything(0)
    world_model, actor, critic, target_critic, _ = build_agent(fabric, [3], False, cfg, obs_space)
    optimizers = fabric.setup_optimizers(
        *[torch.optim.Adam(m.parameters(), lr=1e-2) for m in (world_model, actor, critic)]
    )
    moments = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    # A different batch in every process
    generator = torch.Generator().manual_seed(fabric.global_rank)
    T, B = 4, 2
    data = {
        "state": torch.randn(T, B, 5, generator=generator),
        "actions": torch.nn.functional.one_hot(torch.randint(3, (T, B), generator=generator), 3).float(),
        "rewards": torch.randn(T, B, 1, generator=generator),
        "terminated": torch.zeros(T, B, 1),
        "truncated": torch.zeros(T, B, 1),
        "is_first": torch.zeros(T, B, 1),
    }
    train(fabric, world_model, actor, critic, target_critic, *optimizers, data, None, cfg, False, [3], moments)
    initial_states = fabric.all_gather(world_model.rssm.initial_recurrent_state.detach())
    return torch.equal(initial_states[0], initial_states[1])


def test_the_processes_learn_the_same_initial_recurrent_state():
    # The learnable initial recurrent state is in no module wrapped by DDP: its gradient wasn't averaged over the
    # processes, and each one trained its own
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    assert fabric.launch(initial_states_after_a_gradient_step)


@pytest.mark.parametrize(
    "actor_cls", ["sheeprl.algos.dreamer_v3.agent.Actor", "sheeprl.algos.dreamer_v3.agent.MinedojoActor"]
)
def test_the_actors_of_dreamer_v3_and_p2e_dv3_follow_the_configuration_of_the_actor(actor_cls):
    # The actors ignored `algo.actor.max_std` and `algo.actor.unimix` (the one of the world model, `algo.unimix`, was
    # used), and the exploration actor of P2E-DV3 `algo.actor.action_clip`. The task actor is built as the actor of
    # DreamerV3
    from sheeprl.algos.p2e_dv3.agent import build_agent as build_p2e_agent

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=p2e_dv3_exploration",
                "env=dummy",
                "algo.cnn_keys.encoder=[]",
                "algo.cnn_keys.decoder=[]",
                "algo.mlp_keys.encoder=[state]",
                "algo.mlp_keys.decoder=[state]",
                "algo.dense_units=8",
                "algo.world_model.recurrent_model.recurrent_state_size=8",
                "algo.world_model.representation_model.hidden_size=8",
                "algo.world_model.transition_model.hidden_size=8",
                f"algo.actor.cls={actor_cls}",
                "algo.actor.max_std=0.7",
                "algo.actor.unimix=0.2",
                "algo.actor.action_clip=0.3",
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-20, 20, shape=(5,), dtype=np.float32)})
    _, _, actor_task, _, _, actor_exploration, *_ = build_p2e_agent(
        Fabric(accelerator="cpu", devices=1), [3], False, cfg, obs_space
    )
    for actor in (actor_task.module, actor_exploration.module):
        assert actor.max_std == 0.7
        assert actor._unimix == 0.2
        assert actor._action_clip == 0.3
